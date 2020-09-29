import os
import time

import numpy as np
from gpflow.monitor import Monitor, MonitorTaskGroup, ScalarToTensorBoard, ImageToTensorBoard, ModelToTensorBoard
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tqdm import tqdm

from experiments import Experiment
from utils.plot_utils import plot_synthetic_samples, plot_decoder, plot_encoded_samples, plot_drift_predictions
from utils.math import build_delay_space
from utils.train import (tbptt_chunks_generator, optimize_sde_with_nat_grad, optimize_sde_standard_grad,
                         optimize_nnets_and_hpars, get_breaked_loss, draw_fast_samples)
from VAE import VAE
from SdeModel import SdeModel
from vaele_config import tf_floatx


class Trainer:
    def __init__(self, model: VAE, experiment: Experiment):
        self.model = model
        self.experiment = experiment

    def configure_tensorboard_monitor(self, scalar_period, imgs_period, nb_images=1, do_phase_space=None):
        if do_phase_space is None:
            do_phase_space = self.model.phase_space_dim == 2
        if self.experiment.tensorboard_dir is None or scalar_period < 1:
            return None

        def create_bloss_tasks(directory):
            bloss_names = ['-ly', '-lx', 'penalty_term', 'alpha_term', '-H', '+KL']
            bloss_tasks = []
            def create_lambda(i):
                return lambda train_bloss=None, **kwargs: train_bloss[i]
            for i, name in enumerate(bloss_names):
                    bloss_tasks.append(
                        ScalarToTensorBoard(directory, create_lambda(i), 'bloss/' + name)
                    )
            return bloss_tasks

        train_dir = os.path.join(self.experiment.tensorboard_dir, 'train')
        test_dir = os.path.join(self.experiment.tensorboard_dir, 'test')

        # diff_task = ModelToTensorBoard(train_dir, self.model.sde_model.diffusion)
        # drift_task = ModelToTensorBoard(train_dir, self.model.sde_model.drift_svgp)
        diff_task = []
        drift_task = []

        train_loss = ScalarToTensorBoard(train_dir, lambda train_loss=None, **kwargs: train_loss, 'loss')
        test_loss = ScalarToTensorBoard(test_dir,
                                        lambda epoch=None, kl_scheduler=None, **kwargs: self.test_loss(epoch, kl_scheduler),
                                        'loss')

        train_bloss_list = create_bloss_tasks(train_dir)

        # train_bloss_list = []  # TODO: remove or add

        generator = self.experiment.test_dataset if self.experiment.has_test else self.experiment.train_dataset
        y_inputs = []
        y_targets = []
        for y in generator.take(1):
            for y_input, y_target in self.tbptt_chunks_generator(y):
                break
        #         y_inputs.append(y_input)
        #         y_targets.append(y_target)
        # y_input = tf.concat(y_inputs, axis=1)
        # y_target = tf.concat(y_targets, axis=1)

        def calc_drift_error(**kwargs):
            samples, entropies, encoded_dist, q0_stats, states = draw_fast_samples(self.model, None, y_input)
            fx, var_fx = self.model.sde_model.drift_svgp.predict_f(tf.reshape(samples, (-1, samples.shape[-1])))
            fx = tf.reshape(fx, samples.shape)
            return tf.reduce_mean(tf.square(samples[..., 1:, :] - samples[..., :-1, :] - fx[..., :-1, :]))
        drift_error = ScalarToTensorBoard(train_dir, calc_drift_error, 'drift_error')
        beta_alpha = ScalarToTensorBoard(train_dir, lambda **kwargs: tf.reduce_mean(self.model.sde_model.diffusion.expected_diffusion()),
                                         'beta_div_alpha')
        if imgs_period > 0:
            print('Creating image callbacks')
            images_dir = os.path.join(self.experiment.tensorboard_dir, 'images')

            nrows = 2 if self.model.phase_space_dim > 3 else 1
            encoded_samples = ImageToTensorBoard(
                images_dir, lambda f, a: plot_encoded_samples(f, a, self.model, y_input), 'encoded_samples',
                fig_kw={'figsize': (12, 12)},
                subplots_kw={'nrows': nrows, 'ncols': np.ceil(5 / 2).astype(int)}
            )

            def plot_synth(fig, axes):
                plot_synthetic_samples(fig, axes, self.model, y_input, y_target,
                                       simulation_steps=y.shape[-2])

            nrows = 2 if do_phase_space else 1
            synthetic_samples = ImageToTensorBoard(
                images_dir, plot_synth, 'synthetic_samples',
                fig_kw={'figsize': (12, 12)},
                subplots_kw={'nrows': nrows, 'ncols': nb_images}
            )

            def plot_dec(fig, axes):
                plot_decoder(fig, axes, self.model, y_input, y_target)

            nrows = 2 if self.experiment.batch_size > 1 else 1
            dec_images = ImageToTensorBoard(
                images_dir, plot_dec, 'decoder',
                fig_kw={'figsize': (12, 12)},
                subplots_kw={'nrows': nrows, 'ncols': min(self.experiment.batch_size // nrows, 2)}
            )
            drift_images = ImageToTensorBoard(
                images_dir, lambda fig, axes: plot_drift_predictions(fig, axes, self.model, y_input), 'drift',
                fig_kw={'figsize': (12, 12)},
                subplots_kw={'nrows': nrows, 'ncols': self.model.sde_model.dimension}
            )

            monitor = Monitor(
                MonitorTaskGroup([train_loss, test_loss] + train_bloss_list, period=scalar_period),
                # MonitorTaskGroup([drift_error, beta_alpha], period=scalar_period),
                MonitorTaskGroup([
                    synthetic_samples,
                    dec_images,
                    encoded_samples,
                    drift_images
                ], period=imgs_period)
            )
            print('done')
        else:
            monitor = Monitor(
                MonitorTaskGroup([train_loss, test_loss] + train_bloss_list, period=scalar_period),
                MonitorTaskGroup([drift_error, beta_alpha], period=scalar_period),
            )
        return monitor

    # def configure_pretraining_monitor(self, scalar_period, imgs_period, nb_images=1, do_phase_space=None):
    #     if self.experiment.tensorboard_dir is None or scalar_period < 1:
    #         return None
    #
    #     def create_bloss_tasks(directory):
    #         bloss_names = ['-ly', '-lx', 'penalty_term', 'alpha_term', '-H', '+KL']
    #         bloss_tasks = []
    #         def create_lambda(i):
    #             return lambda train_bloss=None, **kwargs: train_bloss[i]
    #         for i, name in enumerate(bloss_names):
    #                 bloss_tasks.append(
    #                     ScalarToTensorBoard(directory, create_lambda(i), 'bloss/' + name)
    #                 )
    #         return bloss_tasks
    #
    #     train_dir = os.path.join(self.experiment.tensorboard_dir, 'train')
    #     test_dir = os.path.join(self.experiment.tensorboard_dir, 'test')
    #     train_loss = ScalarToTensorBoard(train_dir, lambda train_loss=None, **kwargs: train_loss, 'loss')
    #     train_bloss_list = create_bloss_tasks(train_dir)
    #
    #     generator = self.experiment.test_dataset if self.experiment.has_test else self.experiment.train_dataset
    #     for y in generator.take(1):
    #         for y_input, y_target in self.tbptt_chunks_generator(y):
    #             break
    #
    #     def drift_error(**kwargs):
    #         ((samples, _, _), _, _, _) = self.model._encode_and_decode(y, False, None)
    #         fx, var_fx = self.model.sde_model.drift_svgp.predict_f(tf.reshape(samples, (-1, samples.shape[-1])))
    #         fx = tf.reshape(fx, samples.shape)
    #         return tf.reduce_mean(tf.square(samples[..., 1:, :] - samples[..., :-1, :] - fx[..., :-1, :]))
    #
    #     drift_error = ScalarToTensorBoard(train_dir, drift_error, 'drift_error')
    #     beta_alpha = ScalarToTensorBoard(train_dir, lambda **kwargs: tf.reduce_mean(self.model.sde_model.diffusion.expected_diffusion()),
    #                                      'beta_div_alpha')
    #     if imgs_period > 0:
    #         images_dir = os.path.join(self.experiment.tensorboard_dir, 'images')
    #
    #         nrows = 2 if self.model.phase_space_dim > 3 else 1
    #
    #         def plot_fast_samples(fig, axes, svae, y):
    #             samples, entropies, encoded_dist, q0_stats, _ = draw_fast_samples(svae, None, y)
    #             encoded_means, encoded_scales = encoded_dist
    #             # Focus on first batch, first sample...
    #             samples = samples[0, 0, ...]
    #             ms = encoded_means[0, ...]
    #             ss = encoded_scales[0, ...]
    #             if isinstance(axes, np.ndarray):
    #                 axes = axes.flatten()
    #             for ax, sample, m, s in zip(axes, tf.transpose(samples), tf.transpose(ms), tf.transpose(ss)):
    #                 ax.plot(m, label='mean', color='blue')
    #                 ax.fill_between(np.arange(m.shape[0]), m - s, m + s, color='blue', alpha=0.2)
    #                 ax.plot(sample, label='sample', color='orange')
    #             ax.legend()
    #
    #         encoded_samples = ImageToTensorBoard(
    #             images_dir, lambda f, a: plot_fast_samples(f, a, self.model, y), 'encoded_samples',
    #             fig_kw={'figsize': (12, 12)},
    #             subplots_kw={'nrows': nrows, 'ncols': np.ceil(5 / 2).astype(int)}
    #         )
    #
    #         def plot_dec(fig, axes):
    #             batches = []
    #             samples, entropies, encoded_dist, q0_stats, states = draw_fast_samples(self.model, None, y)
    #             decoded_means, decoded_scales_diag = tf.map_fn(lambda x: self.model.decoder(x), samples,
    #                                                            dtype=(tf_floatx(), tf_floatx()))
    #             # Focus on first sample
    #             means = decoded_means[0]
    #             scales = decoded_scales_diag[0]
    #             drop = 0
    #             if isinstance(axes, np.ndarray):
    #                 axes = axes.flatten()
    #             for ax, yy, mean, scale in zip(axes, y, means, scales):
    #                 ax.plot(yy[drop:, 0])
    #                 ax.plot(mean[..., 0])
    #                 ax.fill_between(np.arange(len(mean[..., 0])),
    #                                 mean[..., 0] - 2 * scale[..., 0], mean[..., 0] + 2 * scale[..., 0],
    #                                 color='orange', alpha=0.2
    #                                 )
    #         nrows = 2 if self.experiment.batch_size > 1 else 1
    #         dec_images = ImageToTensorBoard(
    #             images_dir, plot_dec, 'decoder',
    #             fig_kw={'figsize': (12, 12)},
    #             subplots_kw={'nrows': nrows, 'ncols': min(self.experiment.batch_size // nrows, 2)}
    #         )
    #
    #         def pppp(fig, axes):
    #             samples, entropies, encoded_dist, q0_stats, states = draw_fast_samples(self.model, None, y)
    #             x = tf.reshape(samples, (-1, samples.shape[-1]))
    #             fx, var_fx = self.model.sde_model.drift_svgp.predict_f(x)
    #             predicted_mean = tf.reshape(x + fx, samples.shape)
    #             fxs = tf.reshape(fx, samples.shape)
    #             predicted_scale = tf.reshape(tf.sqrt(var_fx), samples.shape)
    #             # Pick only first sample
    #             samples = samples[0]
    #             predicted_mean = predicted_mean[0]
    #             fxs = fxs[0]
    #             predicted_scale = predicted_scale[0]
    #
    #             if not isinstance(axes, np.ndarray) or axes.ndim != 2:
    #                 axes = axes.reshape((1, -1))
    #             for ax_row, yy, mean, fx, scale in zip(axes, samples, predicted_mean, fxs, predicted_scale):
    #                 for i, ax in enumerate(ax_row):
    #                     ax.plot(yy[..., i])
    #                     # predictions are delayed by 1 sample with respect to yy
    #                     indexes = np.arange(1, len(mean[..., i]) + 1)
    #                     ax.plot(indexes, mean[..., i])
    #                     ax.fill_between(
    #                         indexes,
    #                         mean[..., i] - 2 * scale[..., i], mean[..., i] + 2 * scale[..., i],
    #                         color='orange', alpha=0.2
    #                     )
    #                     ax.plot(indexes, fx[..., i])
    #             ax.legend(['samples', 'predictions', 'fx'])
    #
    #         drift_images = ImageToTensorBoard(
    #             images_dir, pppp, 'drift',
    #             fig_kw={'figsize': (12, 12)},
    #             subplots_kw={'nrows': nrows, 'ncols': self.model.sde_model.dimension}
    #         )
    #
    #         monitor = Monitor(
    #             MonitorTaskGroup([train_loss] + train_bloss_list, period=scalar_period),
    #             MonitorTaskGroup([drift_error, beta_alpha], period=scalar_period),
    #             MonitorTaskGroup([dec_images, encoded_samples, drift_images], period=imgs_period)
    #         )
    #     else:
    #         monitor = Monitor(
    #             MonitorTaskGroup([train_loss] + train_bloss_list, period=scalar_period),
    #             MonitorTaskGroup([drift_error, beta_alpha], period=scalar_period),
    #         )
    #     return monitor

    def configure_checkpoints(self):
        ckpt = None,
        manager = None
        if self.experiment.checkpoint_dir is not None:
            ckpt = tf.train.Checkpoint(step=tf.Variable(1), graphicalModel=self.model)
            manager = tf.train.CheckpointManager(ckpt, self.experiment.checkpoint_dir, max_to_keep=100) # TODO
        return ckpt, manager

    def default_optimizer(self):
        return tf.keras.optimizers.Adam(3e-4)

    def tbptt_chunks_generator(self, y):
        return tbptt_chunks_generator(
            y, self.experiment.len_tbptt,
            self.experiment.time_lag,
            self.model.encoder.kernel_size,
            self.model.encoder.dilation_rate
        )

    # @tf.function
    # def _pretrain_step(self, y_input, y_target, optimizer: tf.keras.optimizers.Optimizer, effective_len, gamma=0.1,
    #                    kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx()), initial_state=None,
    #                    clip_natural_grads=100):
    #     samples, entropies, encoded_dist, q0_stats, _ = draw_fast_samples(self.model, initial_state, y_input)
    #     sde_nat_grads = self.model.sde_nat_grads(samples, effective_len,
    #                                              kl_weight=tf.convert_to_tensor(1., dtype=tf_floatx()))
    #     thetas = SdeModel.standard_to_natural_params([
    #         self.model.sde_model.diffusion._alphas.value(), self.model.sde_model.diffusion._betas.value(),
    #         self.model.sde_model.drift_svgp.q_mu.value(), self.model.sde_model.drift_svgp.q_sqrt.value()
    #     ])
    #     if clip_natural_grads:
    #         sde_nat_grads = [tf.clip_by_value(grad, -clip_natural_grads, clip_natural_grads) for grad in sde_nat_grads]
    #     new_xis = SdeModel.natural_to_standard_params(
    #         (thetas[0] - gamma * sde_nat_grads[0],
    #          thetas[1] - gamma * sde_nat_grads[1],
    #          thetas[2] - gamma * sde_nat_grads[2],
    #          thetas[3] - gamma * sde_nat_grads[3])
    #     )
    #     self.model.sde_model.diffusion._alphas.assign(new_xis[0])
    #     self.model.sde_model.diffusion._betas.assign(new_xis[1])
    #     self.model.sde_model.drift_svgp.q_mu.assign(new_xis[2])
    #     self.model.sde_model.drift_svgp.q_sqrt.assign(new_xis[3])
    #
    #     vars = self.model.encoder.trainable_variables + self.model.decoder.trainable_variables + self.model.sde_model.hyperpars
    #     with tf.GradientTape(watch_accessed_variables=False) as tape:
    #         tape.watch(vars)
    #         samples, entropies, encoded_dist, q0_stats, states = draw_fast_samples(self.model, initial_state, y_input)
    #         decoded_mean, decoded_scale = tf.map_fn(lambda x: self.model.decoder(x), samples,
    #                                                 dtype=(tf_floatx(), tf_floatx()))
    #         breaked_loss = self.model._breaked_loss(y_input, y_target, samples, entropies, encoded_dist, (decoded_mean, decoded_scale),
    #                                                initial_state, effective_len, kl_weight)
    #         loss = tf.reduce_sum(breaked_loss)
    #     grad = tape.gradient(loss, vars)
    #     optimizer.apply_gradients(zip(grad, vars))
    #
    #     def var_to_prec_mat(vars):
    #         # vars [batch, dim]
    #         return tf.map_fn(tf.linalg.diag, 1 / vars)
    #
    #     return loss, breaked_loss, [states, encoded_dist[0][:, -1, :], var_to_prec_mat(tf.square(encoded_dist[1][:, -1, :]))]

    # def pretrain(self, epochs, optimizer, kl_scheduler, verbose=10, monitor=None, epoch_0=1):
    #     avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf_floatx())
    #     avg_breaked_loss = tf.keras.metrics.MeanTensor(name='breaked_loss', dtype=tf_floatx())
    #     for epoch in range(epoch_0, epoch_0 + epochs + 1):
    #         kl_weight = kl_scheduler(epoch)
    #         for y in self.experiment.train_dataset:
    #             state = None
    #             for y_input, y_target in self.tbptt_chunks_generator(y):
    #                 loss, breaked_loss, state = self._pretrain_step(
    #                     y_input, y_target, optimizer, effective_len=self.experiment.effective_len,
    #                     kl_weight=kl_weight, initial_state=state
    #                 )
    #                 avg_loss.update_state(loss)
    #                 avg_breaked_loss.update_state(breaked_loss)
    #         if epoch % verbose == 0:
    #             print(f'Epoch {epoch}, loss={avg_loss.result()}')
    #         if monitor:
    #             try:
    #                 monitor(epoch, epoch=epoch, kl_scheduler=kl_scheduler,
    #                         train_loss=avg_loss.result(), train_bloss=avg_breaked_loss.result())
    #                 avg_loss.reset_states()
    #                 avg_breaked_loss.reset_states()
    #             except Exception as e:
    #                 # TODO
    #                 print("============ Exception in monitor ================")
    #                 print(e)
    # def pretrain_phase_space(self, epochs, optimizer, verbose=0):
    #     for epoch in range(epochs):
    #         for y in self.experiment.train_dataset:
    #             y, Y = build_delay_space(y, self.model.sde_model.dimension, self.experiment.time_lag)
    #             initial_state = None
    #             drop = ((self.model.encoder.kernel_size - 1) * self.model.encoder.dilation_rate)
    #             loss, initial_state = self._pretrain_phase_space_step(y, Y[:, drop:, :], optimizer, initial_state)
    #         if verbose > 0 and epoch % verbose == 0:
    #             print(f'Pretraining, Epoch {epoch}: {loss}')
    #
    # def _pretrain_phase_space_step(self, x_chunk, y_chunk, optimizer: tf.keras.optimizers.Optimizer, initial_state=None):
    #     with tf.GradientTape() as tape:
    #         mean, scales, states = self.model.encoder(x_chunk, training=True, initial_state=initial_state)
    #         loss = -tf.reduce_mean(
    #             tfd.MultivariateNormalDiag(mean, scales).log_prob(y_chunk)
    #         )
    #     grad = tape.gradient(loss, self.model.encoder.trainable_variables)
    #     optimizer.apply_gradients(zip(grad, self.model.encoder.trainable_variables))
    #     return loss, states

    # def pretrain_ae(self, epochs, optimizer, use_sampling=True, decoder_only=False, verbose=0):
    #     """
    #     Pretrain the model focusing on the Encoder and Decoder parts of the SVAE: 1) the graphical
    #     model connecting the SDE with the encoding potentials is not used, 2) The loss is just a
    #     reconstruction loss.
    #     """
    #     for epoch in range(epochs):
    #         tloss = 0
    #         for y in self.experiment.train_dataset:
    #             initial_state = None
    #             for x_chunk, y_chunk in self.tbptt_chunks_generator(y):
    #                 loss, initial_state = self._pretrain_ae_step(
    #                     x_chunk, y_chunk, optimizer, use_sampling=use_sampling, decoder_only=decoder_only,
    #                     initial_state=initial_state
    #                 )
    #                 tloss += loss
    #         if verbose > 0 and epoch % verbose == 0:
    #             print(f'Pretraining, Epoch {epoch}: {tloss}')
    #
    # @tf.function
    # def _pretrain_ae_step(self, x_chunk, y_chunk, optimizer: tf.keras.optimizers.Optimizer, use_sampling, decoder_only, initial_state=None):
    #     with tf.GradientTape(watch_accessed_variables=False) as tape:
    #         if decoder_only:
    #             vars = self.model.decoder.trainable_variables
    #         else:
    #             vars = self.model.encoder.trainable_variables + self.model.decoder.trainable_variables
    #         tape.watch(vars)
    #         mean, scales, states = self.model.encoder(x_chunk, training=True, initial_state=initial_state)
    #         if use_sampling:
    #             samples = tfd.MultivariateNormalDiag(mean, scales).sample(self.model.mpa.nb_samples)
    #         else:
    #             samples = tf.repeat(mean[tf.newaxis, ...], self.model.mpa.nb_samples, axis=0)
    #         decoded_mean, decoded_scale = tf.map_fn(lambda x: self.model.decoder(x), samples, dtype=(tf_floatx(), tf_floatx()))
    #         loss = -tf.reduce_mean(
    #             # tf.reduce_sum(tf.keras.losses.mean_squared_error(decoded, y_chunk[tf.newaxis, ...]), axis=-1)
    #             tfd.MultivariateNormalDiag(decoded_mean, decoded_scale).log_prob(y_chunk[tf.newaxis, ...])
    #         )
    #     grad = tape.gradient(loss, vars)
    #     optimizer.apply_gradients(zip(grad, vars))
    #     return loss, states

    def train(self, epochs, hpars_optimizer, lr_scheduler, kl_scheduler, monitor: Monitor,
              ckpt_manager: tf.train.CheckpointManager, ckpt_period=1, use_natural=True,
              clip_natgrad_value=None, first_epoch=1):
        bar = tqdm(total=self.experiment.len_train_data)
        avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf_floatx())
        avg_breaked_loss = tf.keras.metrics.MeanTensor(name='breaked_loss', dtype=tf_floatx())
        optimizer = None if use_natural else self.default_optimizer()
        self._optimizer_ = optimizer

        # TODO: initial monitoring
        if monitor:
            try:
                epoch = 0
                kl_weight = kl_scheduler(epoch)
                for y in self.experiment.train_dataset:
                    loss, breaked_loss = self._initial_monitor_on_batch(y, epoch, hpars_optimizer, kl_weight, lr_scheduler)
                    avg_loss.update_state(loss)
                    avg_breaked_loss.update_state(breaked_loss)
                    bar.update()

                monitor(epoch, epoch=epoch, kl_scheduler=kl_scheduler,
                        train_loss=avg_loss.result(), train_bloss=avg_breaked_loss.result())
                avg_loss.reset_states()
                avg_breaked_loss.reset_states()
            except:
                pass

        for epoch in tf.range(first_epoch, epochs + 1, dtype=tf_floatx()):
            bar.reset()
            bar.set_description(f'Epoch {epoch}')
            kl_weight = kl_scheduler(epoch)
            for y in self.experiment.train_dataset:
                loss, breaked_loss = self._train_on_batch(
                    y, epoch, hpars_optimizer, kl_weight, lr_scheduler,
                    use_natural, optimizer, clip_value=clip_natgrad_value
                )
                avg_loss.update_state(loss)
                avg_breaked_loss.update_state(breaked_loss)
                bar.update()

            if monitor:
                try:
                    monitor(epoch, epoch=epoch, kl_scheduler=kl_scheduler,
                            train_loss=avg_loss.result(), train_bloss=avg_breaked_loss.result())
                except:
                    pass
            avg_loss.reset_states()
            avg_breaked_loss.reset_states()

            if ckpt_manager and epoch % ckpt_period == 0:
                ckpt_manager.save()

    def _initial_monitor_on_batch(self, y, epoch, hpars_optimizer, kl_weight, lr_scheduler):
            batch_loss = 0.0
            breaked_loss_batch = 0.0
            initial_state = None
            for x_chunk, y_chunk in self.tbptt_chunks_generator(y):
                loss, breaked_loss, final_state = get_breaked_loss(
                    x_chunk, y_chunk, self.model, lr_scheduler(epoch), initial_state,
                    self.experiment.effective_len, kl_weight
                )
                initial_state = final_state

                breaked_loss_batch += breaked_loss
                batch_loss += loss
            return loss, breaked_loss_batch

    def _train_on_batch(self, y, epoch, hpars_optimizer, kl_weight, lr_scheduler, use_natural,
                        optimizer, clip_value=None):
        batch_loss = 0.0
        breaked_loss_batch = 0.0
        initial_state = None
        for x_chunk, y_chunk in self.tbptt_chunks_generator(y):
            if use_natural:
                _, breaked_loss, _ = optimize_sde_with_nat_grad(
                    x_chunk, y_chunk, self.model, lr_scheduler(epoch), initial_state,
                    self.experiment.effective_len, kl_weight, clip_value
                )
            else:
                _, breaked_loss, _ = optimize_sde_standard_grad(
                    x_chunk, y_chunk, self.model, optimizer, initial_state,
                    self.experiment.effective_len, kl_weight
                )
            loss, final_state = optimize_nnets_and_hpars(
                x_chunk, y_chunk, hpars_optimizer,
                self.model, initial_state,
                self.experiment.effective_len,
                kl_weight
            )
            initial_state = final_state

            breaked_loss_batch += breaked_loss
            batch_loss += loss
        return loss, breaked_loss_batch

    def test_loss(self, epoch, kl_scheduler):
        if not self.experiment.has_test:
            return 0
        avg_loss = tf.keras.metrics.Mean(name='loss', dtype=tf_floatx())
        kl_weight = kl_scheduler(epoch)
        for y in self.experiment.test_dataset:
            batch_loss = 0
            initial_state = None
            for x_chunk, y_chunk in self.tbptt_chunks_generator(y):
                chunk_loss, initial_state = self.model.loss(
                    x_chunk, y_chunk, training=False,
                    initial_state=initial_state,
                    effective_nb_of_timesteps=self.experiment.effective_len,
                    kl_weight=kl_weight
                )
                batch_loss += chunk_loss
            avg_loss.update_state(batch_loss)
        return avg_loss.result()


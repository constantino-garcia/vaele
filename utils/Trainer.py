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


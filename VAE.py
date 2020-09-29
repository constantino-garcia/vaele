from nnets.models import Decoder, Encoder, RnnEncoder
from vaele_config import tf_floatx, vaele_jitter
from MessagePassingAlgorithm import MessagePassingAlgorithm
import numpy as np
from SdeModel import SdeModel, DriftParameters, DiffParameters
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
triangular_solve = tf.linalg.triangular_solve

from magic_numbers import INITIAL_PREC
from ParticleFilter import VaeleParticleFilter


class VAE(tf.Module):
    """This is a stateful implementation!! See behind initial_mean and initial_prec"""
    def __init__(self, input_dimension: int, output_dimension: int, len_tbptt: int,
                 encoder_type, encoder_hidden_units, encoder_kernel_size, encoder_dilation_rate,
                 phase_space_dimension: int, drift_parameters: DriftParameters, diff_parameters: DiffParameters,
                 pseudo_inputs: np.array, nb_samples: int, initial_prec=INITIAL_PREC):  # TODO:initial_prec
        self.phase_space_dim = phase_space_dimension
        if encoder_type == 'rnn':
            self.encoder = RnnEncoder(encoder_hidden_units, output_dim=phase_space_dimension)
        # TODO
        # elif encoder_type == 'cnn':
        #     self.encoder = Encoder(encoder_hidden_units, kernel_size=encoder_kernel_size,
        #                            dilation_rate=encoder_dilation_rate, output_dim=phase_space_dimension)
        else:
            raise ValueError('Invalid encoder type')
        self.decoder = Decoder(output_dimension)
        self.sde_model = SdeModel(
            tf.convert_to_tensor(pseudo_inputs, dtype=tf_floatx()),
            drift_parameters, diff_parameters
        )
        # TODO: aqui a machete, elegido en base a la longitud de un batch
        # Using -Inf in case you use a improper prior
        self.minimum_nats = tf.convert_to_tensor(-np.Inf, dtype=tf_floatx())
        # Since we are using, TBPTT, we permit the gm to have a 'state', in the sense that the last means and precs
        # of the current chunk can be propagated to the next chunk as the initial_means and precs.
        # We first set the values for state_0 (t=0) and then create two variables for tracking the state

        self.x0_prior = tfd.MultivariateNormalDiag(
            scale_diag=tf.ones(self.phase_space_dim, dtype=tf_floatx())
        )

        # x0_units = InitialStateEncoder.automatic_units_selection(len_tbptt)
        # self.q_distro_x0 = InitialStateEncoder(x0_units, phase_space_dimension)
        # self.q_distro_x0 = RnnInitialStateEncoder(64, phase_space_dimension)
        # self.mpa = MessagePassingAlgorithm(self.sde_model, nb_samples)
        self.mpa = VaeleParticleFilter(self.sde_model, nb_samples)
        assert nb_samples > 1, 'nb_samples should be > 1'

    def _reduce_batch(self, x, axis=None):
        return tf.reduce_mean(x, axis=axis)

    def _reduce_samples_batch_and_time(self, xs, effective_nb_timesteps):
        """
        :param xs: metrics of the simulated x samples with shape [nb_samples, nb_batches, len_time]
        :return: The final likelihood (a number) of the lks using reductions.
        """
        # Reduce across time taking into account the effective nb of timesteps
        time_reduced = effective_nb_timesteps * tf.reduce_mean(xs, 2)
        time_and_batch_reduced = self._reduce_batch(time_reduced, 1)
        # mean reduce across samples and batches
        return tf.reduce_mean(time_and_batch_reduced, 0)

    def _variational_loglikelihood_x(self, x, x0_mean, x0_scale, effective_nb_timesteps):
        """
        :param x: Trajectories from the embedding space with shape [nb_samples, batch, time, dim].
        :return: Equation (4.18) of the thesis
        """
        lx0 = tfd.MultivariateNormalDiag(x0_mean, x0_scale).log_prob(x[:, :, 0, :])
        # All x but the last temporal instant. This is used for feeding the drift
        x_1Tm1 = x[:, :, :-1, :]
        fx, var_fx = self.sde_model.drift_svgp.predict_f(tf.reshape(x_1Tm1, (-1, x_1Tm1.shape[-1])))
        fx = tf.reshape(fx, x_1Tm1.shape)
        var_fx = tf.reshape(var_fx, x_1Tm1.shape)
        lks = tfd.MultivariateNormalDiag(
            loc=fx,
            scale_diag=tf.sqrt(self.sde_model.diffusion.expected_diffusion())
        ).log_prob(
            # tf.stop_gradient(
                x[:, :, 1:, :] - x_1Tm1
            # )
        )

        alphas = self.sde_model.diffusion.alphas()
        return (
            self._reduce_samples_batch_and_time(lks, effective_nb_timesteps - 1) + tf.reduce_mean(lx0), # TODO
            - 0.5 * self._reduce_samples_batch_and_time(tf.reduce_sum(var_fx, -1), effective_nb_timesteps - 1),
            + 0.5 * (effective_nb_timesteps - 1) * tf.reduce_sum(tf.math.digamma(alphas) - tf.math.log(alphas)),
            # More TODO: delete the following
            [lks, lx0]
        )

    def _loglikelihood_y_given_x(self, y, decoding_means, decoding_scale_diag, effective_nb_timesteps):
        lks = tfd.MultivariateNormalDiag(decoding_means, decoding_scale_diag).log_prob(y[tf.newaxis, ...])
        return self._reduce_samples_batch_and_time(lks, effective_nb_timesteps)

    @tf.function
    def __call__(self, y_input, y_target, training=None, initial_state=None, effective_nb_timesteps=None,
                 kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx())):
        (samples, entropies, _), encoded_dist, decoded_dist, states = self._encode_and_decode(
            y_input, training=training, initial_state=initial_state
        )
        loss = self._loss(
            y_input, y_target, samples, entropies, encoded_dist, decoded_dist, initial_state,
            effective_nb_timesteps, kl_weight
        )
        return samples, encoded_dist, decoded_dist, loss, states

    @tf.function
    def synthetize(self, y_input, y_target, simulation_steps):
        samples, encoded_dist, decoded_dist, loss, states = self.__call__(
            y_input, y_target, training=False, initial_state=None,
            effective_nb_timesteps=tf.convert_to_tensor(1.0, dtype=tf_floatx())
        )
        # Use last samples as starting point. We use just the first sample from each batch
        initial_points = samples[0, :, -1, :]
        predicted_samples = self.sde_model.sample_trajectories(initial_points, simulation_steps)
        return self.decoder(predicted_samples), predicted_samples

    def sampling_dist(self, y_input, y_target, training=None, initial_state=None):
        (_, _, sampling_dist), _, _, _ = self._encode_and_decode(
            y_input, training=training, initial_state=initial_state
        )
        return sampling_dist

    @tf.function
    def loss(self, y_input, y_target, training=None, initial_state=None,
             effective_nb_of_timesteps=None,
             kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx())):

        if effective_nb_of_timesteps is None:
            effective_nb_of_timesteps = y_target.shape[1]
        return self.__call__(y_input, y_target, training=training, initial_state=initial_state,
                             effective_nb_timesteps=effective_nb_of_timesteps,
                             kl_weight=kl_weight)[-2:]

    @tf.function
    def _encode_and_decode(self, y_input, training=None, initial_state=None, use_mask=False):
        """
        :param y_input:
        :param training:
        :param initial_state: (encoder_initial_state,  initial_dynamic_mean, initial_dynamic_prec)
        :return:
        """
        encoder_initial_state, x0_stats = self._unzip_initial_state(initial_state)

        encoded_means, encoded_scales, encoder_states = self.encoder(
            y_input, training=training, initial_state=encoder_initial_state
        )
        initial_dynamic_mean, initial_dynamic_scale = self._handle_x0_stats(x0_stats, (encoded_means, encoded_scales))

        # encoded_covs = tfd.MultivariateNormalDiag(scale_diag=encoded_scales).covariance()
        samples, entropies, sampling_distro, final_mean, final_prec = (
            self.mpa.forward_backward(
                # encoded_means, encoded_covs, initial_dynamic_mean, tf.map_fn(lambda x: tf.linalg.diag(1 / x ** 2), initial_dynamic_scale)
                (initial_dynamic_mean, initial_dynamic_scale), (encoded_means, encoded_scales)
            )
        )
        decoded_means, decoded_scales_diag = tf.map_fn(
            lambda x_: self.decoder(x_, training=training),
            samples,
            dtype=(tf_floatx(), tf_floatx())
        )
        return (
            (samples, entropies, sampling_distro),
            (encoded_means, encoded_scales),
            (decoded_means, decoded_scales_diag),
            (encoder_states, final_mean, final_prec)
        )

    def _handle_x0_stats(self, x0_stats, encoded_stats):
        encoded_means, encoded_scales = encoded_stats
        if x0_stats:
            initial_dynamic_mean, initial_dynamic_prec = x0_stats
            initial_dynamic_scale = tf.map_fn(lambda x: tf.sqrt(1 / tf.linalg.diag_part(x)), initial_dynamic_prec)
        else:
            initial_dynamic_mean, initial_dynamic_scale = encoded_means[:, 0, :], encoded_scales[:, 0, :]
        return initial_dynamic_mean, initial_dynamic_scale

    def _unzip_initial_state(self, initial_state):
        if initial_state:
            encoder_initial_state, initial_dynamic_mean, initial_dynamic_prec = initial_state
            x0_stats = (initial_dynamic_mean, initial_dynamic_prec)
        else:
            encoder_initial_state, x0_stats = None, None
        return encoder_initial_state, x0_stats

    def _breaked_loss(self, y_input, y_target, samples, entropies, encoded_dist, decoded_dist, initial_state,
                      effective_nb_timesteps, kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx())):
        # This implements equation (4.17) from the thesis (A part of which is detailed in Eq. (4.18))

        _, x0_stats = self._unzip_initial_state(initial_state)
        x0_mean, x0_scale = self._handle_x0_stats(x0_stats, encoded_dist)

        (decoded_means, decoded_scales_diag) = decoded_dist
        # Reduce all the entropies (one per batch) taking into account the effective_nb_timesteps
        reduced_entropy = (effective_nb_timesteps / y_target.shape[1]) * tf.reduce_mean(entropies)

        ly = self._loglikelihood_y_given_x(y_target, decoded_means, decoded_scales_diag, effective_nb_timesteps)
        lx, mpenalty, alphaterm, [lxs, lx0s] = self._variational_loglikelihood_x(samples,
                                                                                 x0_mean, x0_scale,
                                                                                 effective_nb_timesteps)
        kl = self.sde_model.kullback_leibler(self.minimum_nats)

        # Note the minus so that this is a loss (instead of lower bound!)
        # TODO: kl is 0
        return tf.stack([
            -ly, -kl_weight * lx, -kl_weight * mpenalty, -kl_weight * alphaterm,
            -kl_weight * reduced_entropy, kl_weight * kl
        ])
        # return tf.stack([-ly, -kl_weight * lx, -kl_weight * mpenalty,
        #                  -kl_weight * alphaterm, -kl_weight * reduced_entropy, kl_weight * kl])

    @tf.function
    def _loss(self, y_input, y_target, samples, entropies, encoded_dist, decoded_dist, initial_state,
              effective_nb_timesteps, kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx())):
        # # TODO:
        # loss = -(
        #     ly + lx + reduced_entropy - kl
        # )
        # return loss
        # initial_state = self._handle_initial_state(
        #     initial_state, y_target.shape[0]
        # )
        # if effective_nb_timesteps is None:
        #     effective_nb_timesteps = y_target.shape[1]

        return tf.reduce_sum(
            self._breaked_loss(
                y_input, y_target, samples, entropies, encoded_dist, decoded_dist,
                initial_state, effective_nb_timesteps, kl_weight
            )
        )

    # def nat_grads(self, y_input, y_target, training, effective_nb_timesteps,
    #               kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx())):
    #     vars = [
    #         self.q_distro_x0.q_mu.unconstrained_variable,
    #         self.q_distro_x0.q_diag_var.unconstrained_variable
    #     ]
    #     with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
    #         tape.watch(vars)
    #         expectations = self.q_distro_x0.meanvarsqrt_to_expectation(self.q_distro_x0.q_mu, self.q_distro_x0.q_diag_var)
    #         xis = self.q_distro_x0.expectation_to_meanvarsqrt(*expectations)
    #         (samples, entropies, _), encoded_dist, decoded_dist, states, qx0 = self._encode_and_decode(
    #             y_input, training=training, initial_state=None
    #         )
    #         breaked_loss = self._breaked_loss(
    #             y_target, samples, entropies, encoded_dist, decoded_dist, None, effective_nb_timesteps, kl_weight
    #         )
    #         loss = tf.reduce_sum(breaked_loss)
    #     dL_dxi = tape.gradient(loss, vars)
    #     # dL_dxi[1] = self.q_distro_x0.q_diag_var.transform.forward(dL_dxi[1])
    #     # Apply chain rule to get the natural gradients
    #     x0_nat_grads = tape.gradient(
    #         xis, expectations, output_gradients=dL_dxi
    #     )
    #     del tape
    #     return x0_nat_grads, self.sde_nat_grads(samples, effective_nb_timesteps), breaked_loss, loss, states

    @tf.function
    def nat_grads(self, y_input, y_target, training, initial_state, effective_nb_timesteps,
                  kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx())):
        (samples, entropies, _), encoded_dist, decoded_dist, states = self._encode_and_decode(
            y_input, training=training, initial_state=initial_state
        )
        breaked_loss = self._breaked_loss(
            y_input, y_target, samples, entropies, encoded_dist, decoded_dist, initial_state, effective_nb_timesteps, kl_weight
        )
        loss = tf.reduce_sum(breaked_loss)
        # TODO!
        natgrads = self.sde_nat_grads(samples, effective_nb_timesteps)
        # natgrads = self.automatic_sde_nat_grads(
        #     y_target, samples, entropies, encoded_dist,
        #     decoded_dist, initial_state, effective_nb_timesteps,
        #     kl_weight
        # )
        return natgrads, breaked_loss, loss, states

    @tf.function
    def automatic_sde_nat_grads(self, y_input, y_target, samples, entropies, encoded_dist, decoded_dist,
                                initial_state, effective_nb_timesteps=None,
                                kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx())):
        alphas = self.sde_model.diffusion._alphas
        betas = self.sde_model.diffusion._betas
        q_mu = self.sde_model.drift_svgp.q_mu
        q_sqrt = self.sde_model.drift_svgp.q_sqrt
        vars = [
            alphas.unconstrained_variable, betas.unconstrained_variable,
            q_mu.unconstrained_variable, q_sqrt.unconstrained_variable
        ]
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as tape:
            tape.watch(vars)
            expectations = self.sde_model.expectation_params()
            xis = self.sde_model.expectation_to_xi(expectations)
            loss = self._loss(y_input, y_target, samples, entropies, encoded_dist, decoded_dist, initial_state,
                              effective_nb_timesteps, kl_weight)

        dL_dxi = tape.gradient(loss, vars)
        # Apply chain rule to get the natural gradients
        natural_gradients = tape.gradient(
            xis, expectations, output_gradients=dL_dxi
        )
        del tape
        return natural_gradients

    @tf.function
    def sde_nat_grads(self, samples, effective_nb_timesteps, kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx())):
        # We first compute the gradients of the lower-bound. In the return we will change sign so
        # that they are the gradients of the loss function
        real_nb_timesteps = samples.shape[2]
        # -1 is due to the fact that we loose one sample when dealing with dx = x_1:T - x_0:(T -1)
        # TODO
        weight = (effective_nb_timesteps - 1) / (real_nb_timesteps - 1)
        dsamples = samples[:, :, 1:, :] - samples[:, :, :-1, :]

        alphas = self.sde_model.diffusion._alphas
        betas = self.sde_model.diffusion._betas
        q_mu = self.sde_model.drift_svgp.q_mu
        q_sqrt = self.sde_model.drift_svgp.q_sqrt

        kl = self.sde_model.kullback_leibler_by_dimension(self.minimum_nats)

        theta_0, theta_1, theta_2, theta_3 = SdeModel.standard_to_natural_params(
            [alphas.value(), betas.value(), q_mu.value(), q_sqrt.value()]
        )
        # The relevant priors
        alpha0 = self.sde_model.diffusion.prior_distribution.concentration
        theta_0_prior = alpha0 - 0.5
        beta0 = self.sde_model.diffusion.prior_distribution.rate
        theta_1_prior = -beta0
        if self.sde_model.drift_svgp._vague_prior:
            theta_3_prior = tf.zeros_like(theta_3, dtype=tf_floatx()) + vaele_jitter()
        else:
            theta_3_prior = -0.5 * tf.map_fn(tf.linalg.inv, self.sde_model.drift_svgp.prior_distribution.covariance())
        # Some auxiliar variables
        q_sqrt_T = tf.transpose(q_sqrt, [0, 2, 1])
        Q = tf.einsum('aij,ajk->aik', q_sqrt, q_sqrt_T)
        prec = tf.map_fn(tf.linalg.inv, Q)

        chol_Kmm = self.sde_model._get_Kmm_chol()

        nat_mnat_0 = ((alphas - 0.5) - theta_0_prior)
        nat_mnat_0 = kl_weight * tf.where(kl > self.minimum_nats, nat_mnat_0, tf.zeros_like(nat_mnat_0))
        ngrad_0 = weight * 0.5 * dsamples.shape[2] - nat_mnat_0

        prec_mu = tf.einsum('aij,ja->ai', prec, q_mu)
        mu_prec_mu = tf.einsum('ia,ai->a', q_mu, prec_mu)
        # Since the prior of mu is 0 we don't compute mu_0 Prec_0 mu_0
        nat_mnat_1 = (-betas - 0.5 * mu_prec_mu - theta_1_prior)
        nat_mnat_1 = kl_weight * tf.where(kl > self.minimum_nats, nat_mnat_1, tf.zeros_like(nat_mnat_1))

        ngrad_1 = (
            - weight * 0.5 * tf.reduce_mean(tf.reduce_sum(tf.square(dsamples), axis=2), axis=[0, 1]) +
            -nat_mnat_1
        )

        term_2 = 0.0
        term_3 = 0.0
        for ii in range(samples.shape[0]):
            for jj in range(samples.shape[1]):
                Kmn = self.sde_model.kernel.K(self.sde_model.iv_values(), samples[ii, jj, :-1, :], full_output_cov=False)
                B = triangular_solve(chol_Kmm, Kmn)
                term_2 += tf.einsum('abt,ta->ba', B, dsamples[ii, jj, :, ])
                term_3 += tf.einsum('ait,ajt->aij', B, B)

        term_2 /= (samples.shape[0] * samples.shape[1])
        term_3 /= (samples.shape[0] * samples.shape[1])

        nat_mnat_2 = kl_weight * tf.where((kl > self.minimum_nats)[tf.newaxis, ...],
                                          theta_2, tf.zeros_like(theta_2))
        ngrad_2 = weight * term_2 - nat_mnat_2

        # I = tf.eye(q_sqrt.shape[1], batch_shape=[q_sqrt.shape[0]], dtype=tf_floatx())
        nat_mnat_3 = theta_3 - theta_3_prior

        nat_mnat_3 = kl_weight * tf.where(tf.reshape((kl > self.minimum_nats), [-1, 1, 1]),
                              nat_mnat_3, tf.zeros_like(nat_mnat_3)
        )
        ngrad_3 = -weight * 0.5 * term_3 - nat_mnat_3

        # ngrad_x refers to the gradient of the lower-bound. Therefore, we change here the
        # sign so that it is the gradient of the loss
        return -ngrad_0, -ngrad_1, -ngrad_2, -ngrad_3



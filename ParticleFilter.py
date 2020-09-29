import os

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from nnets.models import RnnEncoder
from SdeModel import SdeModel
from utils.math import multiply_gaussians
from vaele_config import tf_floatx


def forward_resampling(particles, log_weights, alpha):
    log_sampling_weights = tf.math.log(alpha) + log_weights + tf.math.log((1 - alpha) / len(log_weights))
    indices = tfd.Categorical(logits=log_sampling_weights).sample(len(log_weights))
    new_log_weights = log_weights - log_sampling_weights
    new_log_weights = new_log_weights - tf.reduce_logsumexp(new_log_weights)
    return tf.gather(particles, indices), new_log_weights


def backward_sample(particles, log_weights):
    indices = tf.squeeze(tfd.Categorical(logits=log_weights).sample(1), axis=0)
    samples = tf.map_fn(
        lambda x: tf.gather(x[0], x[1]),
        elems=(particles, tf.transpose(indices)),
        dtype=tf_floatx()
    )
    return tf.transpose(samples, [1, 0, 2])


class ParticleFilter(object):
    def __init__(self, proposal_builder, dynamics_fn, weight_fn,
                 nb_samples, n_particles=100, n_eff_threshold=None):
        self.nb_samples = nb_samples
        self.n_particles = n_particles
        self.n_eff_threshold = n_eff_threshold or n_particles // 2
        self.proposal_builder = proposal_builder
        self.dynamics_fn = dynamics_fn
        self.weight_fn = weight_fn
        self.alpha = tf.constant(0.99, dtype=tf_floatx())

    # def forward_pass(self, encoding_potentials):
    #     particles, weights = self._forward_pass(encoding_potentials)
    #     # TODO: repeated code, see forward_backward
    #     return tf.transpose(particles, [1, 2, 0, 3]), tf.transpose(weights, [1, 2, 0])
    #
    # # Shape as [batch, particles, time, ...]
    # def backward_pass(self, particles, weights, nb_samples):
    #     particles, weights, entropy = self._backward_pass(
    #         tf.transpose(particles, [2, 0, 1, 3]),
    #         tf.transpose(weights, [2, 0, 1]),
    #         nb_samples
    #     )
    #     # TODO: repeated code, see forward_backward
    #     return tf.transpose(particles, [1, 2, 0, 3]), tf.transpose(weights, [1, 2, 0, 3]), entropy

    def forward_backward(self, init_stats, encoding_potentials):
        particles, log_weights = self._forward_pass(init_stats, encoding_potentials)
        samples, entropies, blog_weights, final_mean, final_prec = self._backward_pass(particles, log_weights)
        return (
            tf.transpose(samples, [1, 2, 0, 3]),
            entropies,
            (particles, log_weights),
            # tf.transpose(bweights, [1, 2, 0, 3]),
            final_mean,
            final_prec
        )
        # TODO: particle and log_weights do not have proper order. When changing it, change also plot_utils/plot_encoded_samples

    def _init_filter(self, init_stats, init_potentials, batch_size):
        # Product of diagonal gaussians
        mean_x, scale_x = init_stats
        mean_y, scale_y = init_potentials
        if mean_x is None or scale_x is None:
            mean_out, scale_out = mean_y, scale_y
        else:
            mean_out, scale_out = mean_x, scale_x
        # var_xy = 1 / (1 / var_x + 1 / var_y)
        # mean_xy = ((mean_y / var_y) + (mean_x / var_x)) * var_xy
        distro = tfd.MultivariateNormalDiag(mean_out, scale_out)
        return (
            tf.transpose(distro.sample(self.n_particles), [1, 0, 2]),
            tf.ones((batch_size, self.n_particles), dtype=tf_floatx()) / self.n_particles
        )

    def _forward_pass(self, init_stats, encoding_potentials):
        # TODO: batch size can be removed
        batch_size = encoding_potentials[0].shape[0]
        particles, log_weights = self._init_filter(
            init_stats, (encoding_potentials[0][:, 0, :], encoding_potentials[1][:, 0, :]), batch_size
        )

        def forward_step(particles, log_weights, encoding_potential):
            particles = self.proposal_builder(particles, encoding_potential).sample()
            log_weights = self.weight_fn(log_weights, particles, encoding_potential)
            log_weights = log_weights - tf.math.reduce_logsumexp(log_weights, axis=-1, keepdims=True)
            # Compute effective sample size and entropy of weighting vector.
            # These are useful statistics for adaptive particle filtering.
            weights = tf.exp(log_weights)
            n_eff = 1.0 / tf.reduce_sum(tf.square(weights), axis=1)
            # resampling (systematic resampling) step
            particles, log_weights = tf.map_fn(
                lambda x: tf.cond(
                    x[0] < self.n_eff_threshold,
                    lambda: forward_resampling(x[1], x[2], self.alpha),
                    lambda: (x[1], x[2])
                ),
                # Transpose to iterate through batch dimension
                elems=(n_eff, particles, log_weights),
                dtype=(tf_floatx(), tf_floatx())
            )
            # Transpose so that dimensions denote [particles, batch, ...]
            return particles, log_weights

        particles, log_weights = tf.scan(
            lambda a, x: forward_step(a[0], a[1], x),
            # transpose encoding_potentials so that temporal axis is the first one
            elems=(tf.transpose(encoding_potentials[0], [1, 0, 2]), tf.transpose(encoding_potentials[1], [1, 0, 2])),
            initializer=(particles, log_weights)
        )
        return particles, log_weights

    # particles and weights have time as leading dimension: [time, batch, particles, ...]
    def _backward_pass(self, particles, log_weights):
        # Add a new axis to weights to account for the number of samples. Hence, the expected shape for weights
        # is [nb_sample, batch, particles]
        last_log_weights = tf.repeat(log_weights[-1][tf.newaxis, ...], self.nb_samples, axis=0)
        last_particles = particles[-1]
        last_sample = backward_sample(last_particles, last_log_weights)

        weights = tf.exp(log_weights)
        final_mean = tf.reduce_sum(particles[-1] * weights[-1, ..., tf.newaxis], axis=1)
        final_var = (
                tf.reduce_sum((particles[-1] ** 2) * weights[-1, ..., tf.newaxis], axis=1) - final_mean ** 2
        )
        final_prec = tf.map_fn(tf.linalg.diag, 1 / final_var)

        # def logdet(samples):
        #     vars = tf.reduce_mean(tf.square(samples), axis=0) - tf.square(tf.reduce_mean(samples, axis=0))
        #     # logdet estimate
        #     return tf.reduce_sum(tf.math.log(vars + 1e-10), -1) # TODO
        # last_logdet = logdet(last_sample)

        def backward_step(backward_info, forward_particles, log_weights):
            backward_particles = backward_info[0]
            # Add new axis to weights to account for nb_samples
            log_weights = log_weights[tf.newaxis, ...] + self.dynamics_fn(forward_particles, backward_particles)
            log_weights = log_weights - tf.reduce_logsumexp(log_weights, axis=-1, keepdims=True)
            samples = backward_sample(forward_particles, log_weights)
            return samples, log_weights

        # Drop last temporal particles and weights
        particles_1_Tm1 = particles[:-1, ...]
        weights_1_Tm1 = log_weights[:-1, ...]
        # Rearrange as [time, batch, particle, ...] for scan
        bparticles, blog_weights = tf.scan(
            lambda a, x: backward_step(a, *x),
            elems=(particles_1_Tm1, weights_1_Tm1),
            initializer=(last_sample, tf.zeros_like(last_log_weights, dtype=tf_floatx())),
            reverse=True
        )

        # Samples dimension taking into account the number of time steps
        # samples_dimension = (bparticles.shape[0] + 1) * bparticles.shape[-1]
        # entropies = 0.5 * (
        #         samples_dimension * (1 + tf.math.log(2.0 * tf.constant(np.pi, tf_floatx()))) +
        #         logdets[0]
        # )
        bparticles = tf.concat([bparticles, last_sample[tf.newaxis, ...]], axis=0)
        s = bparticles.shape
        covs = tfp.stats.covariance(tf.reshape(bparticles, [s[0] * s[-1], s[1], s[2]]), sample_axis=1, event_axis=0)
        covs = tf.transpose(covs, [2, 0, 1])
        # TODO
        covs = covs + 1e-8 * tf.eye(covs.shape[1], batch_shape=[covs.shape[0]], dtype=tf_floatx())
        # entropies = tfd.MultivariateNormalFullCovariance(covariance_matrix=covs).entropy()
        entropies = 0.5 * covs.shape[1] * (1 + tf.math.log(2 * tf.constant(np.pi, dtype=tf_floatx()))) + 0.5 * tf.linalg.logdet(covs)
        return (
            bparticles,
            entropies,
            tf.concat([blog_weights, last_log_weights[tf.newaxis, ...]], axis=0),
            final_mean,
            final_prec
        )


class VaeleParticleFilter(ParticleFilter):
    def __init__(self, sde_model: SdeModel, nb_samples, n_particles=200, n_eff_threshold=None):
        def proposal_builder(particles, encoding_potential):
            x = tf.reshape(particles, (-1, particles.shape[-1]))
            fx, _ = sde_model.drift_svgp.predict_f(x)
            # Product of diagonal gaussians
            mean_x = tf.reshape(x + fx, particles.shape)
            var_x = sde_model.diffusion.expected_diffusion()
            mean_y, scale_y = encoding_potential
            var_y = tf.square(scale_y)
            var_xy = 1 / (1 / var_x + 1 / var_y)[:, tf.newaxis, :]
            mean_xy = ((mean_y / var_y)[:, tf.newaxis, :] + (mean_x / var_x)) * var_xy
            return tfd.MultivariateNormalDiag(loc=mean_xy, scale_diag=tf.sqrt(var_xy))

        def dynamics_fn(particles, future_particles):
            # particles shape is [batch, n_particles, dim] and future_particles shape is [samples, batch, dim]
            # Output dimension should be [samples, batch, particles, ...]
            fx, _ = sde_model.drift_svgp.predict_f(tf.reshape(particles, (-1, particles.shape[-1])))
            fx = tf.reshape(fx, particles.shape)
            return tfd.MultivariateNormalDiag(
                loc=(particles + fx)[tf.newaxis, ...],
                scale_diag=tf.sqrt(sde_model.diffusion.expected_diffusion())
            ).log_prob(future_particles[:, :, tf.newaxis, :])

        def weight_fn(log_weights, particles, encoding_potential):
            fx, _ = sde_model.drift_svgp.predict_f(tf.reshape(particles, (-1, particles.shape[-1])))
            predicted_particles = particles + tf.reshape(fx, particles.shape)
            encoding_mean, encoding_scale = encoding_potential
            scale = tf.sqrt(tf.square(encoding_scale) + sde_model.diffusion.expected_diffusion())
            return (
                log_weights +
                tfd.MultivariateNormalDiag(encoding_mean[:, tf.newaxis, :], scale[:, tf.newaxis, :]).log_prob(
                    predicted_particles
                )
            )

        super(VaeleParticleFilter, self).__init__(
            proposal_builder, dynamics_fn, weight_fn, nb_samples,
            n_particles=n_particles, n_eff_threshold=n_eff_threshold
        )



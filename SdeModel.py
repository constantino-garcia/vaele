from utils.math import inverse_lower_triangular
from vaele_config import tf_floatx
from collections import namedtuple
import gpflow
from gpflow.utilities import set_trainable
from gpflow.config import default_positive_bijector
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from utils.inducing_points import kmeans_based_inducing_points, grid_based_inducing_points
from scipy.spatial.distance import pdist

tfd = tfp.distributions
triangular_solve = tf.linalg.triangular_solve

# amplitudes: a vector whose len equals the output dimension (SDE dimension)
# length_scales: a matrix whose shape[0] equals the SDE dimension and whose shape[1] equals input dimension
DriftParameters = namedtuple('DriftParameters', ('amplitudes', 'length_scales'))
# alphas and betas: vectors whose lens equal the output dimension (SDE dimension)
DiffParameters = namedtuple('DiffParameters', ('alphas', 'betas'))


class Diffusion(gpflow.Module):
    def __init__(self, diff_parameters: DiffParameters):
        super(Diffusion, self).__init__()

        def get_bijector():
            if gpflow.config.default_positive_bijector() == 'exp':
                return tfp.bijectors.Exp()
            elif gpflow.config.default_positive_bijector() == 'softplus':
                return tfp.bijectors.Softplus()
            else:
                raise ValueError("Unexpected value in default_positive_bijector()")

        assert len(diff_parameters.alphas) == len(diff_parameters.betas), "len(alphas) != len(betas)"
        self.dimension = len(diff_parameters.alphas)
        # TODO: remove this attempt to tie params
        # self.tied_params = tie_params
        # if tie_params:
        #     alphas = tf.reduce_mean(diff_parameters.alphas)
        #     betas = tf.reduce_mean(diff_parameters.betas)
        # else:
        alphas = diff_parameters.alphas
        betas = diff_parameters.betas
        self._alphas = gpflow.Parameter(
            tf.ones_like(alphas, dtype=tf_floatx()),
            # alphas,
            transform=get_bijector(), name='alphas'
        )
        self._betas = gpflow.Parameter(
            # TODO
            tf.ones_like(betas, dtype=tf_floatx()),
            # betas,
            transform=get_bijector(), name='betas'
        )
        self.prior_distribution = tfd.Gamma(alphas, betas)

    def alphas(self) -> tf.Tensor:
        return self._alphas

    def betas(self) -> tf.Tensor:
        return self._betas

    # TODO
    def _tie_params(self, input: tf.Tensor) -> tf.Tensor:
        if self.tied_params:
            return tf.repeat(input, self.dimension)
        else:
            return input

    def expected_precision(self) -> tf.Tensor:
        return self.alphas() / self.betas()

    def expected_diffusion(self) -> tf.Tensor:
        return 1. / self.expected_precision()

    def expected_log_precision(self) -> tf.Tensor:
        return tf.math.digamma(self.alphas()) - tf.math.log(self.betas())


class Drift(gpflow.models.SVGP):
    def __init__(self, kernel: gpflow.kernels.SeparateIndependent, inducing_points: tf.Tensor, num_latent: int,
                 prior_scale=1.0):
        self.nb_inducing_variables = int(inducing_points.shape[0])
        inducing_variables = gpflow.inducing_variables.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(
                tf.Variable(inducing_points, name='inducing_points.py')
            )
        )
        super().__init__(kernel, gpflow.likelihoods.Gaussian(),
                         inducing_variable=inducing_variables,
                         num_latent_gps=num_latent)

        self._q_mu_0 = self.q_mu
        self._q_sqrt_0 = self.q_sqrt

        self._vague_prior = np.any(np.isinf(prior_scale))
        if not self._vague_prior:
            self.prior_distribution = tfd.MultivariateNormalDiag(
                tf.zeros_like(tf.transpose(self.q_mu), dtype=tf_floatx()),
                tf.repeat(
                    tf.convert_to_tensor(prior_scale, dtype=tf_floatx())[..., tf.newaxis],
                    self.q_mu.shape[0], axis=-1
                )
            )
        #gpflow.utilities.set_trainable(self.likelihood.variance, False)

    def reset(self):
        self.q_mu.assign(self._q_mu_0)
        self.q_sqrt.assign(self._q_sqrt_0)


class SdeModel(tf.Module):
    # TODO: permit to specify drift_prior scale
    def __init__(self, inducing_points: tf.Tensor, drift_parameters: DriftParameters, diff_parameters: DiffParameters):
        # TODO: validate drift and diff_parameters
        self.diffusion = Diffusion(diff_parameters)
        self.dimension = self.diffusion.dimension
        kern_list = []
        for i, (amplitude, length_scale) in enumerate(zip(drift_parameters.amplitudes, drift_parameters.length_scales)):
            kern_list.append(
                gpflow.kernels.RBF(amplitude, length_scale)
                # gpflow.kernels.Matern12(amplitude, length_scale)
            )

        self.kernel = gpflow.kernels.SeparateIndependent(kern_list)
        self.drift_svgp = Drift(self.kernel, inducing_points, num_latent=self.dimension,
                                # TODO: using vague prior by default
                                prior_scale=tf.ones_like(tf.sqrt(diff_parameters.alphas / diff_parameters.betas))
        )
        self.variational_variables = list(
             tuple(self.diffusion.trainable_variables) +
             self.drift_svgp.q_mu.trainable_variables + self.drift_svgp.q_sqrt.trainable_variables
        )
        self.hyperpars = list(
            self.drift_svgp.kernel.trainable_variables + self.drift_svgp.inducing_variable.trainable_variables
        )

    def iv_values(self):
        # TODO: check change from return self.drift_svgp.inducing_variable.inducing_variable_shared.Z to
        return self.drift_svgp.inducing_variable.inducing_variable.Z

    def _get_Kmm_chol(self):
        iv_values = self.iv_values()
        Kmms = tf.stack(
            [k.K(iv_values) for k in self.kernel.kernels],
            axis=0
        )  # [P, M, M]
        chol_Kmm = tf.map_fn(tf.linalg.cholesky, Kmms)
        return chol_Kmm

    def redistribute_inducing_points(self, samples, lengthscale=None, strategy='cluster'):
        if strategy not in ['cluster', 'grid']:
            raise ValueError('strategy should be "grid" or "cluster"')

        if strategy == 'grid':
            def _range(x):
                return [np.min(x), np.max(x)]
            reshaped_samples = samples.numpy().reshape((-1, samples.shape[-1])).T
            ranges = [_range(x) for x in reshaped_samples]
            new_ips = grid_based_inducing_points(ranges, self.iv_values().shape[0])
        else:
            new_ips = kmeans_based_inducing_points(tf.reshape(samples, [-1, samples.shape[-1]]),
                                                   self.iv_values().shape[0], scale_range=0.)

        _ = self.drift_svgp.inducing_variable.trainable_variables[0].assign(new_ips)
        self.drift_svgp.reset()
        # Select a new lengthscale to avoid issues with the inverse matrix (ips will probably
        # be closer)
        if lengthscale is None:
            lengthscale = np.quantile(pdist(new_ips), 0.05)
        for kernel in self.drift_svgp.kernel.kernels:
            kernel.lengthscales.assign(lengthscale * tf.ones_like(kernel.lengthscales))

    def standard_params(self):
        return [
            self.diffusion.alphas(), self.diffusion.betas(),
            self.drift_svgp.q_mu, self.drift_svgp.q_sqrt
        ]

    def natural_params(self):
        return self.standard_to_natural_params(self.standard_params())

    def expectation_params(self):
        return self.standard_to_expectation(self.standard_params())

    def kullback_leibler(self, free_bits=tf.convert_to_tensor(0, dtype=tf_floatx())):
        return tf.reduce_sum(
            self.kullback_leibler_by_dimension(free_bits)
        )

    def kullback_leibler_by_dimension(self, free_bits=tf.convert_to_tensor(0, dtype=tf_floatx())):
        # TODO do not permit not whiten representations!
        if self.drift_svgp._vague_prior:
            # The kullback-leibler when using a vague prior, up to constant (theoretically, infinite)
            q_sqrt = self.drift_svgp.q_sqrt
            gaussian_kl = -0.5 * q_sqrt.shape[-1] - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(q_sqrt)), 1)
            # We don't apply the free bits technique when using the vague prior, since we already have
            # infinite free bits in the gaussian_kl!
            kl = (
                    gaussian_kl +
                    tfd.Gamma(self.diffusion.alphas(), self.diffusion.betas()).kl_divergence(
                        self.diffusion.prior_distribution
                    )
            )
        else:
            kl_mu = tf.transpose(
                tf.sqrt(self.diffusion.expected_precision())[tf.newaxis, ...] * self.drift_svgp.q_mu
            )
            q_sqrt = self.drift_svgp.q_sqrt
            kl = (
                tfd.MultivariateNormalTriL(kl_mu, q_sqrt).kl_divergence(
                    self.drift_svgp.prior_distribution
                ) +
                tfd.Gamma(self.diffusion.alphas(), self.diffusion.betas()).kl_divergence(
                    self.diffusion.prior_distribution
                )
            )
            kl = tf.math.maximum(free_bits, kl)
        return kl

    @staticmethod
    def natural_to_standard_params(theta):
        theta_0, theta_1, theta_2, theta_3 = theta

        var_sqrt_inv = tf.linalg.cholesky(-2 * theta_3)
        var_sqrt = inverse_lower_triangular(var_sqrt_inv)
        Q = tf.linalg.matmul(var_sqrt, var_sqrt, transpose_a=True)

        alphas = theta_0 + 0.5
        betas = -theta_1 - 0.5 * tf.einsum('ia,aij,ja->a', theta_2, Q, theta_2)
        mu = tf.einsum('aij,ja->ia', Q, theta_2)
        q_sqrt = tf.linalg.cholesky(Q)

        return alphas, betas, mu, q_sqrt

    @staticmethod
    def standard_to_natural_params(standard_params):
        alphas, betas, mu, q_sqrt = standard_params
        inverse_q_sqrt = inverse_lower_triangular(q_sqrt)
        precision = tf.linalg.matmul(
            inverse_q_sqrt, inverse_q_sqrt, transpose_a=True
        )

        theta_0 = alphas - 0.5
        theta_1 = -betas - 0.5 * tf.einsum('ia,aij,ja->a', mu, precision, mu)
        theta_2 = tf.einsum('aij,ja->ia', precision, mu)
        theta_3 = -0.5 * precision

        return theta_0, theta_1, theta_2, theta_3

    @staticmethod
    def standard_to_expectation(standard_params):
        alphas, betas, q_mu, q_sqrt = standard_params
        aux = tf.sqrt(alphas / betas)[tf.newaxis, ...] * q_mu
        return (
            tf.math.digamma(alphas) - tf.math.log(betas),
            alphas / betas,
            (alphas / betas)[tf.newaxis, ...] * q_mu,
            (
                tf.einsum('ia,ja->aij', aux, aux) +
                tf.linalg.matmul(q_sqrt, q_sqrt, transpose_b=True)
            )

        )

    @staticmethod
    def expectations_to_standard(expectations):
        eta_0, eta_1, eta_2, eta_3 = expectations
        search_results = tfp.math.secant_root(
            lambda alpha: tf.math.digamma(alpha) - tf.math.log(alpha) - eta_0 + tf.math.log(eta_1),
            -1 / (eta_0 - tf.math.log(eta_1)), - 0.5 / (eta_0 - tf.math.log(eta_1))
        )
        alpha = search_results.estimated_root
        beta = search_results.estimated_root / eta_1
        mu = eta_2 / eta_1[tf.newaxis, ...]
        aux = eta_2 / tf.sqrt(eta_1[tf.newaxis, ...])
        Q = eta_3 - tf.einsum('ji,ki->ijk', aux, aux)
        q_sqrt = tf.linalg.cholesky(Q)

        return alpha, beta, mu, q_sqrt

    def standard_to_xi(self, standard_params):
        return (
            self.diffusion._alphas.transform.inverse(standard_params[0]),
            self.diffusion._betas.transform.inverse(standard_params[1]),
            standard_params[2],
            self.drift_svgp.q_sqrt.transform.inverse(standard_params[3])
        )

    def xi_to_standard(self, xis_0, xis_1, xis_2, xis_3):
        return (
            self.diffusion._alphas.transform.forward(xis_0),
            self.diffusion._betas.transform.forward(xis_1),
            xis_2,
            self.drift_svgp.q_sqrt.transform.forward(xis_3)
        )

    def expectation_to_xi(self, expectations):
        return self.standard_to_xi(self.expectations_to_standard(expectations))

    def natural_to_xi(self, theta):
        return self.standard_to_xi(self.natural_to_standard_params(theta))

    def sample_trajectories(self, initial_points: tf.Tensor, sim_steps: int) -> tf.Tensor:
        """
        Simulate trajectories starting at initial points (one trajectory for each initial point) for 'sim_steps'
        simulation steps

        :param initial_points: a Matrix (nb_different_initial_points x dimension)
        :param sim_steps: number of simulation steps
        :return: SDE trajectories, one for each different initial point
        """
        # assert initial_points.ndim == 2, "initial_points is not a matrix"
        diffusions = tfp.distributions.MultivariateNormalDiag(
            tf.zeros(initial_points.shape, dtype=tf_floatx()),
            tf.sqrt(self.diffusion.expected_diffusion()),
        ).sample(sim_steps, dtype=tf_floatx())
        trajectories = tf.scan(
            lambda x_tm1, noise_term: x_tm1 + self.drift_svgp.predict_f(x_tm1)[0] + noise_term,
            elems=diffusions,
            initializer=initial_points,
            name='sde_sim'
        )
        # transpose so that output has shape (initial_points.shape[0], sim_steps + 1, input_dim)
        return tf.concat([
            initial_points[:, tf.newaxis, :],
            tf.transpose(trajectories, [1, 0, 2])
        ], axis=1)


if __name__ == '__main__':
    from vaele_config import default_float, np_floatx
    from SdeModel import *
    drift_parameters = DriftParameters(
        tf.convert_to_tensor([1, 1], dtype=default_float()),
        tf.convert_to_tensor(np.array([[1, 1], [2, 2]]), dtype=default_float())
    )
    diff_parameters = DiffParameters(
        tf.convert_to_tensor([1, 1], dtype=np_floatx()),
        tf.convert_to_tensor([2, 2], dtype=np_floatx())
    )

    d = Diffusion(diff_parameters)
    print(d.alphas())
    print(d.betas())
    print(d.expected_precision())
    # Z = tf.convert_to_tensor(
    #     np.random.randn(10 * 2).reshape((10, 2)),
    #     dtype=np_floatx()
    # )
    # sde_model = SdeModel(Z, drift_parameters, diff_parameters)
    # initial_points = tf.convert_to_tensor(np.random.normal(0, 1, (10, 2)), dtype=default_float())






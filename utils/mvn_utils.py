import numpy as np
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalFullCovariance, MultivariateNormalDiag
from  utils.tf_utils import to_floatx
from config import  tf_floatx


def multiply_gaussians(means_a, inv_cov_a, means_b, inv_cov_b):
    """
    :param means_a: N different means arranged as an array of N X D
    :param inv_cov_a: A single PRECISION matrix D x D
    :param means_b: M different means arranged as an array of M x D. N and M should be consistent with broadcast
    operations
    :param inv_cov_b: A single PRECISION matrix D x D
    :return:
    """
    tf.Assert(means_a.get_shape() == 2, [means_a], name='multiply_gaussians_means_a')
    tf.Assert(means_b.get_shape() == 2, [means_b], name='multiply_gaussians_means_b')

    prec = inv_cov_a + inv_cov_b
    sigma = tf.matrix_inverse(prec)
    sigma = (sigma + tf.transpose(sigma)) / 2.0

    mu = tf.matmul(
        tf.matmul(means_a, inv_cov_a) + tf.matmul(means_b, inv_cov_b),
        sigma
    )
    return mu, sigma, prec


# # TODO: merge with previous definition
# def multiply_gaussian_potentials(gaussians_with_common_cov, gaussian_2):
#     """
#     means_1 is a matrix representing several means (one mean per row)
#     mean_2 is a single mean
#     cov_1 and cov_2 represent two covariance matrices!
#     """
#     means_1, cov_1 = gaussians_with_common_cov
#     mean_2, cov_2 = gaussian_2
#     (cov_1_inv, cov_2_inv) = (tf.matrix_inverse(cov_1), tf.matrix_inverse(cov_2))
#     sigma = tf.matrix_inverse(cov_1_inv + cov_2_inv)
#     sigma = (sigma + tf.transpose(sigma)) / 2.0
#     # first term in inner product is n x s (s is the number of samples). To sum the second 'mean' we have to
#     # reshape to column vector. The second inner product results in n x s. To reorder mu as one sample per row
#     # we have to transpose it.
#     mu = tf.matmul(
#         tf.matmul(means_1, cov_1_inv) + tf.matmul(tf.expand_dims(mean_2, 0), cov_2_inv),
#         sigma
#     )
#     return mu, sigma


def mvn_diag_loglikelihood(x, mean, cov):
    if x.get_shape().ndims == 1:
        x_ = tf.reshape(x, (1, -1))
    else:
        x_ = x
    return MultivariateNormalDiag(mean, tf.sqrt(cov)).log_prob(x_)


def mvn_loglikelihood(x, mean, cov):
    if x.get_shape().ndims == 1:
        x_ = tf.reshape(x, (1, -1))
    else:
        x_ = x
    return MultivariateNormalFullCovariance(mean, cov).log_prob(x_)


def mvn_entropy(covariance, dimension=None):
    if dimension is None:
        dimension = tf.shape(covariance)[0]
    return to_floatx(dimension / 2. * (1 + np.log(2 * np.pi))) + 0.5 * tf.linalg.logdet(covariance)


def mvn_sample(nb_samples, mean, cov):
    return tf.contrib.distributions.MultivariateNormalFullCovariance(mean, cov).sample(nb_samples)


from config import tf_floatx
from SdeModel import SdeModel
from nnets.Mlp import Mlp
import numpy as np
import tensorflow as tf
from utils.mvn_utils import multiply_gaussians, mvn_sample, mvn_entropy
from utils.decorators import tf_property, define_scope
from utils.tf_utils import to_floatx


class MessagePassingAlgorithm(object):
    def __init__(self, sde: SdeModel, mean_net: Mlp, cov_net: Mlp, initial_means, initial_precs,
                 use_initial_state, nb_samples: int):
        assert nb_samples > 1, "nb_samples should be > 1"
        self.sde = sde
        self.input_dimension = sde.input_dimension
        self.mean_net = mean_net
        self.cov_net = cov_net
        self.initial_means = initial_means
        self.initial_precs = initial_precs
        self.use_initial_state = use_initial_state
        self.nb_samples = nb_samples

        # Collect all the kernel parameters from the SdeModel
        self.amplitudes = tf.constant(np.stack([k.amplitude for k in self.sde.kernels]))
        self.length_scales_matrices = tf.stack([k.length_scales_matrix for k in self.sde.kernels])
        self.inverse_length_scales_matrices = tf.stack([k.inverse_length_scales_matrix for k in self.sde.kernels])

        # self.filtering_betas # = self._compute_filtering_betas()
        # self.forward_pass
        # self.backward_pass
        # self.nb_samples
        # self.entropy

    @tf_property
    def filtering_betas(self):
        return tf.map_fn(
            lambda x: tf.squeeze(tf.matmul(tf.reshape(x[1], (1, -1)), x[0])),
            elems=[self.sde.inv_pseudo_inputs_matrices, self.sde.standard_params['means']],
            dtype=tf_floatx(),
            name="filtering_betas_map"
        )

    @define_scope("filtering_q")
    def compute_filtering_q(self, mean_tm1, cov_tm1):
        def q_single_kernel(params):
            amplitude, length_scales_matrix, inverse_length_scales_matrix = params
            S = cov_tm1 + length_scales_matrix
            S_inv = tf.matrix_inverse(S)
            determinant = tf.linalg.det(
                tf.matmul(cov_tm1, inverse_length_scales_matrix) + tf.eye(self.input_dimension, dtype=tf_floatx())
            )
            mult_factor = amplitude / tf.sqrt(determinant)

            def kernel_part(x):
                delta = tf.reshape(x - mean_tm1, (1, -1))
                return tf.squeeze(
                    tf.exp(-0.5 * tf.matmul(tf.matmul(delta, S_inv), tf.transpose(delta))),
                )

            return tf.squeeze(
                tf.map_fn(
                    lambda x: mult_factor * kernel_part(x),
                    elems=self.sde.inducing_points,
                    name="filtering_q_single_kernel"
                )
            )

        return tf.map_fn(
            fn=q_single_kernel,
            elems=[self.amplitudes, self.length_scales_matrices, self.inverse_length_scales_matrices],
            dtype=tf_floatx(),
            name='filtering_q'
        )

    @define_scope("Q_ab_given_ij")
    def calculate_Q_ab_given_ij(self, sigma_a, sigma_b,
                                amplitudes_i, amplitudes_j,
                                inverse_length_scales_i, inverse_length_scales_j,
                                R_inv_by_cov_tm1, R_det_factor):
        sigma_a_Lambda_i = tf.matmul(sigma_a, inverse_length_scales_i)
        sigma_b_Lambda_j = tf.matmul(sigma_b, inverse_length_scales_j)
        z_ab = sigma_a_Lambda_i + sigma_b_Lambda_j

        n_ab_2 = tf.squeeze(
            tf.log(amplitudes_i) + tf.log(amplitudes_j)
        ) - 0.5 * tf.squeeze(
            tf.matmul(sigma_a_Lambda_i, tf.transpose(sigma_a)) + tf.matmul(sigma_b_Lambda_j, tf.transpose(sigma_b)) -
            tf.matmul(z_ab, tf.matmul(R_inv_by_cov_tm1, tf.transpose(z_ab)))
        )
        return tf.exp(n_ab_2) / R_det_factor

    @define_scope("Q_given_ij")
    def calculate_Q_given_ij(self, sigmas, amplitudes_i, amplitudes_j,
                             inverse_length_scales_i, inverse_length_scales_j, R_inv_by_cov_tm1, R_det_factor):
        Q = tf.map_fn(
            lambda sigma_a: tf.map_fn(
                lambda sigma_b: self.calculate_Q_ab_given_ij(tf.expand_dims(sigma_a, 0), tf.expand_dims(sigma_b, 0),
                                                             amplitudes_i, amplitudes_j,
                                                             inverse_length_scales_i, inverse_length_scales_j,
                                                             R_inv_by_cov_tm1, R_det_factor),
                elems=sigmas
            ),
            elems=sigmas
        )
        return Q

    @define_scope("E_vf_vf_ij")
    def E_vf_vf_ij(self, i, j, sigmas, cov_tm1, betas, amplitudes, inverse_length_scales_matrices):
        R = (
            tf.matmul(cov_tm1, (inverse_length_scales_matrices[i] + inverse_length_scales_matrices[j])) +
            tf.eye(self.input_dimension, dtype=tf_floatx())
        )
        R_inv = tf.matrix_inverse(R)
        R_inv_by_cov_tm1 = tf.matmul(R_inv, cov_tm1)
        R_det_factor = tf.sqrt(tf.linalg.det(R))
        Q = self.calculate_Q_given_ij(sigmas, amplitudes[i], amplitudes[j],
                                      inverse_length_scales_matrices[i], inverse_length_scales_matrices[j],
                                      R_inv_by_cov_tm1,
                                      R_det_factor)
        return tf.squeeze(tf.matmul(betas[i:(i + 1), :], tf.matmul(Q, tf.transpose(betas[j:j + 1, :]))))

    @define_scope("E_vf_vf")
    def calculate_E_vf_vf(self, mean_tm1, cov_tm1, filtering_betas):
        sigmas = self.sde.inducing_points - mean_tm1
        if self.input_dimension > 1:
            lower_indices = np.array(list(zip(*np.tril_indices(self.input_dimension, k=-1))), dtype=np.int32)
            upper_indices = np.flip(lower_indices, axis=1)
            vf_vf_ij = tf.map_fn(
                lambda x: self.E_vf_vf_ij(x[0], x[1], sigmas, cov_tm1, filtering_betas,
                                          self.amplitudes, self.inverse_length_scales_matrices),
                elems=tf.constant(lower_indices),
                dtype=tf_floatx(),
                name='vf_vf_ij'
            )
        # Diagonal part
        vf_vf_ii = tf.map_fn(
            lambda x: self.E_vf_vf_ij(x, x, sigmas, cov_tm1, filtering_betas,
                                      self.amplitudes, self.inverse_length_scales_matrices),
            elems=tf.constant(np.arange(self.input_dimension)),
            dtype=tf_floatx(),
            name='vf_vf_ii'
        )
        # Shape of vf_vf_ii cannot be inferred. We set it
        vf_vf_ii.set_shape((self.input_dimension,))

        if self.input_dimension > 1:
            return (
                tf.scatter_nd(indices=lower_indices, updates=vf_vf_ij,
                              shape=[self.input_dimension, self.input_dimension]) +
                tf.scatter_nd(indices=upper_indices, updates=vf_vf_ij,
                              shape=[self.input_dimension, self.input_dimension]) +
                tf.diag(vf_vf_ii)
            )
        else:
            return tf.diag(vf_vf_ii)

    @define_scope("forward_prediction")
    def forward_prediction(self, mean_tm1, cov_tm1, inv_cov_tm1):
        qs = self.compute_filtering_q(mean_tm1, cov_tm1)
        filtering_betas_by_qs = self.filtering_betas * qs

        E_vf = tf.expand_dims(tf.reduce_sum(filtering_betas_by_qs, axis=1), 0)
        mean_t = mean_tm1 + E_vf

        psi_tensor3 = tf.map_fn(
            lambda inverse_length_scales_matrix:
            multiply_gaussians(self.sde.inducing_points, inverse_length_scales_matrix, mean_tm1, inv_cov_tm1)[0],
            elems=self.inverse_length_scales_matrices,
            name='filtering_psi'
        )

        E_x_vf = tf.transpose(tf.reduce_sum(tf.expand_dims(filtering_betas_by_qs, -1) * psi_tensor3, axis=1))
        E_vf_vf = self.calculate_E_vf_vf(mean_tm1, cov_tm1, self.filtering_betas)
        cov_x_vf = E_x_vf - tf.matmul(tf.transpose(mean_tm1), E_vf)
        cov_vf_x = tf.transpose(cov_x_vf)
        cov_vf_vf = E_vf_vf - tf.matmul(tf.transpose(E_vf), E_vf)
        # cov_vf_vf = tf.Print(cov_vf_vf, [cov_vf_vf], 'cov_f_f', summarize=4)
        # cov_x_vf = tf.Print(cov_x_vf, [cov_x_vf], 'cov_x_vf', summarize=4)
        cov_t = (
            tf.diag(self.sde.expected_diffusion) + cov_tm1 +
            cov_x_vf + cov_vf_x + cov_vf_vf
            # Regularize cov_t to avoid issues when computing the smoothed distribution
            # to_floatx(1e-5) * tf.eye(self.sde.input_dimension, dtype=tf_floatx())
        )

        # Cov(x_t, x_tm1| y_1:tm1)
        cov_t_tm1_gtm1 = cov_tm1 + cov_vf_x

        return (mean_t, cov_t, tf.matrix_inverse(cov_t)), cov_t_tm1_gtm1 #, E_x_vf, E_vf_vf, cov_x_vf, cov_vf_vf, E_vf
        # Add previous values to return if you want to check,

    @define_scope("forward_step")
    def _forward_step(self, message_tm1, encoding_potentials_t):
        """

        :param message_tm1: The filtered mean, covariance and precision of x_tm1|y_1:tm1
        :param encoding_potentials_t: The gaussian potential relating x_t with y_t
        :return: the filtered distribution x_t|y_1:t (mean, covariance and precision), the predicted
        distribution x_t|y_1:tm1, and the covariance cov(x_t, x_tm1| y_1:tm1)
        """
        # Ignore the predicted distribution and the conditional covariance from previous time step
        (mean_tm1, cov_tm1, prec_tm1), _, _ = message_tm1
        encoding_mean_t, encoding_prec_t = encoding_potentials_t

        (predicted_mean_t, predicted_cov_t, predicted_prec_t), cov_t_tm1_gtm1 = (
            self.forward_prediction(mean_tm1, cov_tm1, prec_tm1)
        )
        return (
            multiply_gaussians(predicted_mean_t, predicted_prec_t, tf.expand_dims(encoding_mean_t, 0), encoding_prec_t),
            (predicted_mean_t, predicted_cov_t, predicted_prec_t),
            cov_t_tm1_gtm1
        )

    @define_scope("single_observation_forward_pass")
    def _single_observation_forward_pass(self, encoder_means, encoder_covs, initial_mean, initial_prec):
        encoder_precs = tf.map_fn(tf.matrix_inverse, encoder_covs, name='encoding_precisions')
        initial_encoder_means, initial_encoder_covs, initial_encoder_precs = (
            tf.expand_dims(encoder_means[0], 0), encoder_covs[0], encoder_precs[0]
        )

        message_mean_t0, message_cov_t0, message_prec_t0 = tf.cond(
            self.use_initial_state,
            true_fn=lambda:  multiply_gaussians(initial_encoder_means, initial_encoder_precs,
                                                tf.expand_dims(initial_mean, 0), initial_prec),
            false_fn=lambda: (initial_encoder_means, initial_encoder_covs, initial_encoder_precs)
        )

        filtered_distributions, predicted_distributions, conditional_covs = tf.scan(
            self._forward_step,
            elems=[encoder_means[1:], encoder_precs[1:]],
            # Initialize the filtered distributions and the predicted distributions (see forward_step)
            initializer=(
                (message_mean_t0, message_cov_t0, message_prec_t0),
                (tf.zeros_like(message_mean_t0), tf.zeros_like(message_cov_t0), tf.zeros_like(message_prec_t0)),
                tf.zeros_like(message_cov_t0)
            ),
            name='forwards_messages_scan',
            # TODO
            parallel_iterations=1

        )

        return (
            # The filtered distributions (x_t|y_1:t)
            (
                tf.concat([tf.expand_dims(message_mean_t0, 0), filtered_distributions[0]], axis=0),  # means
                tf.concat([tf.expand_dims(message_cov_t0, 0), filtered_distributions[1]], axis=0),   # covs
                tf.concat([tf.expand_dims(message_prec_t0, 0), filtered_distributions[2]], axis=0)   # precs,
            ),
            #  The predicted distributions (x_t|y_1:tm1)
            predicted_distributions,
            # The covariances cov(x_t, x_tm1| y_1:tm1)
            conditional_covs
        )

    @tf_property
    def forward_pass(self):
        # Calculate average covs for monitoring purposes
        ecovs = tf.reduce_mean(
            tf.reduce_mean(self.cov_net.output, axis=0),   # batch mean
            axis=0
        )   # temporal mean
        tf.summary.scalar('encoding_covs_00', ecovs[0][0])
        tf.summary.scalar('encoding_covs_11', ecovs[1][1])

        return tf.map_fn(
            lambda x: self._single_observation_forward_pass(*x),
            elems=(self.mean_net.output, self.cov_net.output, self.initial_means, self.initial_precs),
            dtype=(
                # filtered distributions (mean, cov, prec)
                (tf_floatx(), tf_floatx(), tf_floatx()),
                # predicted distributions (mean, cov, prec),
                (tf_floatx(), tf_floatx(), tf_floatx()),
                # conditional covariances
                tf_floatx()
            ),
            # TODO
            parallel_iterations=1
        )

    ### Another try on the backward_step, based on RTS smoothing... We've found this to be numerically unstable
    @define_scope("backward_step")
    def _backward_step(self, samples_tp1, accumulated_entropy, filtered_mean_t, filtered_cov_t, filtered_prec_t,
                       predicted_mean_tp1, predicted_cov_tp1, predicted_prec_tp1, conditional_tp1_t_gt):
        """

        :param samples_tp1: A matrix of (nb_samples x dimension) samples at time t + 1. Note that each row should be
        a sample at time t + 1.
        :param filtered_mean_t:
        :param filtered_cov_t:
        :param filtered_prec_t:
        :param predicted_mean_tp1: mean of x_tp1 | y_1:t.
        :param predicted_cov_tp1:
        :param predicted_prec_tp1:
        :param conditional_tp1_t_gt:
        :return:
        """
        J_t = tf.matmul(predicted_prec_tp1, conditional_tp1_t_gt)
        conditional_mean_t = filtered_mean_t + tf.matmul(samples_tp1 - predicted_mean_tp1, J_t)

        tmp_matrix = tf.matmul(tf.transpose(J_t), conditional_tp1_t_gt)
        tmp_matrix = (tmp_matrix + tf.transpose(tmp_matrix)) / 2.0
        conditional_cov_t = filtered_cov_t - tmp_matrix
        conditional_cov_t = (conditional_cov_t + tf.transpose(conditional_cov_t)) / 2.0
        # For each mean, we obtain a sample. Therefore, we set nb_ samples = 1 to obtain a tensor of
        # (1, rows(conditional_means), dimension) = (1, nb_different_means, dimension). Hence, we squeeze dimension
        # 0 to be consistent with the dimension of samples_tp1
        return (
            tf.squeeze(mvn_sample(1, conditional_mean_t, conditional_cov_t), 0),
            accumulated_entropy + mvn_entropy(conditional_cov_t, self.input_dimension),
            conditional_mean_t,
            conditional_cov_t
        )


    @define_scope("single_observation_backward_pass")
    def _single_observation_backward_pass(self, filtered_distributions, predicted_distributions, conditional_covs):
        filtered_means, filtered_covs, filtered_precs = filtered_distributions
        predicted_means, predicted_covs, predicted_precs = predicted_distributions

        # means are represented using row vectors (1 x Dim). To avoid unwanted dimensions, we squeeze axis 0
        end_of_chain_samples = mvn_sample(self.nb_samples, tf.squeeze(filtered_means[-1], axis=0), filtered_covs[-1])
        last_entropy = mvn_entropy(filtered_covs[-1], self.input_dimension)

        # Do the backwards sampling: reverse all the stats from the distributions since we move backwards
        last_smoothed_means=tf.tile(filtered_means[-1], (self.nb_samples, 1))
        samples, acc_entropies, smoothed_means, smoothed_covs = tf.scan(
            lambda acc, args: self._backward_step(acc[0], acc[1], *args),
            elems=[filtered_means[-2::-1], filtered_covs[-2::-1], filtered_precs[-2::-1],
                   predicted_means[::-1], predicted_covs[::-1], predicted_precs[::-1],
                   conditional_covs[::-1]],
            initializer=(end_of_chain_samples, last_entropy, last_smoothed_means, filtered_covs[-1]),
            name='backward_pass',
            # TODO:
            parallel_iterations=1
        )

        # Pick the last accumulated entropy (sum of all)
        entropy = acc_entropies[-1]
        samples = tf.concat([tf.expand_dims(end_of_chain_samples, 0), samples], axis=0)
        smoothed_means = tf.concat([smoothed_means[::-1], tf.expand_dims(last_smoothed_means, 0)], axis=0)
        smoothed_covs = tf.concat([smoothed_covs[::-1], filtered_covs[-1:]], axis=0)
        # Reverse the samples so that they match temporal order and transpose them to arrange them as
        # (nb_samples, nb_time_steps, dimension)
        return tf.transpose(samples[::-1], [1, 0, 2]), entropy, smoothed_means, smoothed_covs

    @tf_property
    def backward_pass(self):
        batch_filtered_distributions, batch_predicted_distributions, batch_conditional_covs = self.forward_pass
        return tf.map_fn(
            lambda x: self._single_observation_backward_pass(
                (x[0], x[1], x[2]),
                (x[3], x[4], x[5]),
                x[6]
            ),
            elems=[
                batch_filtered_distributions[0], batch_filtered_distributions[1], batch_filtered_distributions[2],
                batch_predicted_distributions[0], batch_predicted_distributions[1], batch_predicted_distributions[2],
                batch_conditional_covs
            ],
            dtype=(tf_floatx(), tf_floatx(), tf_floatx(), tf_floatx()),
            # TODO:
            parallel_iterations=1
        )

    @tf_property
    def samples(self):
        return self.backward_pass[0]

    @tf_property
    def entropy(self):
        return self.backward_pass[1]

    @tf_property
    def smoothed_mean(self):
        return self.backward_pass[2]

    @tf_property
    def smoothed_cov(self):
        return self.backward_pass[3]

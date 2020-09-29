from collections import namedtuple
import vaele_config
from vaele_config import np_floatx, tf_floatx, vaele_jitter
import tensorflow as tf
from SdeModel import SdeModel
import tensorflow_probability as tfp
from utils.math import multiply_gaussians

# Named tuples to simplify argument passing
_SqeAuxTems = namedtuple(
    '_SqeAuxTerms',
    ('variances', 'Lambdas', 'inv_Lambdas')
)  # This is only used while not all _FilteringAuxTerms are available

_FilteringAuxTerms = namedtuple(
    '_FilteringAuxTerms',
    ('variances', 'Lambdas', 'inv_Lambdas', 'betas', 'gammas', 'inv_Kmms', 'chol_Kmms', 'Q_abij', 'Q_aij')
)


class MessagePassingAlgorithm(object):
    def __init__(self, sde: SdeModel, nb_samples: int):
        #assert nb_samples > 1, "nb_samples should be > 1"
        self.sde = sde
        # TODO: add option in Drift to whiten or not!
        self.whiten = sde.drift_svgp.whiten  # Is the SVGP using the whitening representation?
        self.nb_samples = nb_samples

    def predict_xt_given_tm1(self, mean_tm1, cov_tm1):
        """
        Predict step of the predict-update cycle while filtering
        """
        with tf.name_scope('predict/aux/'):
            filtering_terms = self._compute_aux_filter_terms(mean_tm1, cov_tm1)
            E_mf_given_tm1 = self._compute_E_mf_given_tm1(filtering_terms)
        with tf.name_scope('predit/stats'):
            mean_t_given_tm1 = mean_tm1 + E_mf_given_tm1
            cov_t_t_given_tm1, cov_t_tm1_given_tm1 = (
                self._compute_covs_given_tm1(mean_tm1, cov_tm1, E_mf_given_tm1, filtering_terms)
            )
        # print(cov_t_t_given_tm1)
        return (
            mean_t_given_tm1, cov_t_t_given_tm1, tf.linalg.inv(cov_t_t_given_tm1), cov_t_tm1_given_tm1
        )

    def _compute_aux_filter_terms(self, mean_tm1, cov_tm1):
        iv_values = self.sde.iv_values()
        Kmms = tf.stack(
            [k.K(iv_values) for k in self.sde.kernel.kernels],
            axis=0
        )  # [P, M, M]
        inv_Kmms = tf.map_fn(tf.linalg.inv, Kmms)
        chol_Kmms = tf.map_fn(tf.linalg.cholesky, Kmms)

        variances = tf.stack([rbf_kernel.variance for rbf_kernel in self.sde.kernel.kernels])
        variances = variances[..., tf.newaxis]  # [P 1]
        Lambdas = tf.stack([
            # tf.reshape neccessary for 1-D lengthscales
            tf.linalg.diag(tf.reshape(rbf_kernel.lengthscales ** 2, (-1,))) for rbf_kernel in self.sde.kernel.kernels
        ])
        inv_Lambdas = tf.stack([tf.linalg.diag(1 / tf.reshape(rbf_kernel.lengthscales ** 2, (-1,))) for rbf_kernel in
                                self.sde.kernel.kernels])
        sqe_terms = _SqeAuxTems(variances=variances, Lambdas=Lambdas, inv_Lambdas=inv_Lambdas)

        # Definitions from Equation 4.29 of the PhD thesis
        if self.whiten:
            betas = tf.linalg.triangular_solve(tf.linalg.adjoint(chol_Kmms),
                                               tf.transpose(self.sde.drift_svgp.q_mu)[..., tf.newaxis],
                                               lower=False)
            betas = tf.squeeze(betas, axis=-1)
        else:
            betas = tf.einsum('aij,ja->ai', inv_Kmms, self.sde.drift_svgp.q_mu)  # [P, M]
        gammas = self._compute_filtering_gammas(mean_tm1, cov_tm1, sqe_terms)
        Q_abij = self._compute_filtering_Q(mean_tm1, cov_tm1, sqe_terms)
        Q_aij = tf.gather_nd(Q_abij, [[i, i] for i in range(Q_abij.shape[0])])

        return _FilteringAuxTerms(
            variances=variances, Lambdas=Lambdas, inv_Lambdas=inv_Lambdas,
            betas=betas, gammas=gammas, inv_Kmms=inv_Kmms, chol_Kmms=chol_Kmms,
            Q_abij=Q_abij, Q_aij=Q_aij
        )

    def _compute_filtering_gammas(self, mean_tm1, cov_tm1, sqe_terms: _SqeAuxTems):
        """
        Deisenroth, 23 and 24
        Definition just after 4.29 of the PhD thesis
        """
        det_term = tf.map_fn(
            lambda x: 1 / tf.sqrt(tf.linalg.det(x)),
            (
                    tf.einsum('ij,ajk->aik', cov_tm1, sqe_terms.inv_Lambdas) +
                    tf.expand_dims(tf.eye(self.sde.dimension, dtype=tf_floatx()), 0)
            )
        )
        det_term = tf.expand_dims(det_term, 1)
        zeta = mean_tm1 - self.sde.iv_values()
        exp_term = tf.exp(-0.5 * tf.reduce_sum(
            tf.tensordot(zeta, tf.linalg.inv(cov_tm1 + sqe_terms.Lambdas), [[1], [1]]) *
            tf.expand_dims(zeta, 1),
            axis=2
        ))
        return sqe_terms.variances * det_term * tf.transpose(exp_term)

    def _compute_filtering_Q(self, mean_tm1, cov_tm1, sqe_terms: _SqeAuxTems):
        """Deisenroth 28"""
        R_ab = (
                tf.einsum('ij,abjk->abik', cov_tm1,
                          tf.expand_dims(sqe_terms.inv_Lambdas, 0) + tf.expand_dims(sqe_terms.inv_Lambdas, 1)
                          ) + tf.expand_dims(tf.expand_dims(tf.eye(self.sde.dimension, dtype=tf_floatx()), 0), 0)
        )
        det_R_ab = tf.map_fn(
            lambda x: tf.map_fn(tf.linalg.det, x),
            R_ab
        )
        inv_R_ab = tf.map_fn(
            lambda x: tf.map_fn(tf.linalg.inv, x),
            R_ab
        )
        zeta_i = self.sde.iv_values() - mean_tm1
        zeta_lambda = tf.einsum('axy,iy->aix', sqe_terms.inv_Lambdas, zeta_i)
        z_abij = (
                tf.expand_dims(tf.expand_dims(zeta_lambda, 1), 3) +
                tf.expand_dims(tf.expand_dims(zeta_lambda, 0), 2)
        )
        n_tmp = tf.transpose(
            tf.reduce_sum(
                tf.tensordot(zeta_i, sqe_terms.inv_Lambdas, [[1], [1]]) * tf.expand_dims(zeta_i, 1),
                axis=-1
            )
        )
        log_variances = tf.math.log(sqe_terms.variances)
        n2_abij = (
                tf.expand_dims(tf.expand_dims(log_variances + tf.transpose(log_variances), -1), -1) +
                -0.5 * (
                        tf.expand_dims(tf.expand_dims(n_tmp, 1), 3) + tf.expand_dims(tf.expand_dims(n_tmp, 0), 2) -
                        tf.reduce_sum(
                            tf.einsum('abxy,yabij->abijx', inv_R_ab,
                                      tf.tensordot(cov_tm1, z_abij, [[-1], [-1]]), name='Q_einsum') * z_abij,
                            axis=-1
                        )
                )
        )
        sqrt_det_R_ab = tf.sqrt(det_R_ab)[..., tf.newaxis, tf.newaxis]
        Q_abij = tf.exp(n2_abij) / sqrt_det_R_ab
        return Q_abij

    def _compute_E_mf_given_tm1(self, filtering_terms: _FilteringAuxTerms):
        return tf.reduce_sum(filtering_terms.betas * filtering_terms.gammas, axis=-1)

    def _compute_covs_given_tm1(self, mean_tm1, cov_tm1, E_mf_given_tm1, filtering_terms: _FilteringAuxTerms):
        """Deisenroth 31"""
        cov_mf_mf = self._compute_cov_mf_mf_givent_tm1(mean_tm1, cov_tm1, E_mf_given_tm1, filtering_terms)
        cov_mf_x = self._compute_cov_mf_x_given_tm1(mean_tm1, cov_tm1, filtering_terms)
        diag_terms = tf.linalg.diag(
            self.sde.diffusion.expected_diffusion()
            # self._compute_E_cov_fa_fa_given_tm1(mean_tm1, cov_tm1, filtering_terms)
        )
        cov_t_t_given_tm1 = cov_tm1 + cov_mf_mf + cov_mf_x + tf.transpose(cov_mf_x) + diag_terms
        cov_t_tm1_given_tm1 = cov_tm1 + cov_mf_x
        return cov_t_t_given_tm1, cov_t_tm1_given_tm1

    def _compute_cov_mf_mf_givent_tm1(self, mean_tm1, cov_tm1, E_mf_given_tm1, filtering_terms):
        cov_mf_mf = (
                self._compute_E_mf_mf_given_tm1(mean_tm1, cov_tm1, filtering_terms) -
                tf.einsum('i,j->ij', E_mf_given_tm1, E_mf_given_tm1, name='cov_mf_mf_einsum')
        )  # [P, P]
        return cov_mf_mf

    def _compute_E_cov_fa_fa_given_tm1(self, mean_tm1, cov_tm1, filtering_terms: _FilteringAuxTerms):
        if self.whiten:
            aux_term = tf.linalg.triangular_solve(
                tf.linalg.adjoint(filtering_terms.chol_Kmms),
                self.sde.drift_svgp.q_sqrt,
                lower=False
            )
        else:
            aux_term = tf.linalg.triangular_solve(
                tf.linalg.adjoint(filtering_terms.chol_Kmms),
                tf.linalg.triangular_solve(filtering_terms.chol_Kmms, self.sde.drift_svgp.q_sqrt, lower=True),
                lower=False
            )

        # einsum multiplies A by A^t
        matrix_term = filtering_terms.inv_Kmms - tf.einsum('aij,akj->aik', aux_term, aux_term)
        trace_term = tf.einsum('aij,aji->a', matrix_term, filtering_terms.Q_aij, name='E_cov_fa_fa_einsum')
        return tf.reshape(filtering_terms.variances, (-1,)) - trace_term

    def _compute_cov_mf_x_given_tm1(self, mean_tm1, cov_tm1, filtering_terms: _FilteringAuxTerms):
        expanded_cov_tm1 = tf.tile(cov_tm1[tf.newaxis, ...], [self.sde.dimension, 1, 1])
        matrix_term = tf.einsum('aij,ajk->aik', expanded_cov_tm1,
                                tf.map_fn(tf.linalg.inv, filtering_terms.Lambdas + expanded_cov_tm1),
                                name='compute_cov_mf_x_given_tm1_einsum'
                                )
        # Note that, from the name of the method, first dimension is related to mf and second to x
        return tf.reduce_sum(
            filtering_terms.betas[..., tf.newaxis] *
            filtering_terms.gammas[..., tf.newaxis] *
            tf.einsum('axy,iy->aix', matrix_term, self.sde.iv_values() - mean_tm1),
            axis=1
        )

    def _compute_E_mf_mf_given_tm1(self, mean_tm1, cov_tm1, filtering_terms: _FilteringAuxTerms):
        return tf.einsum(
            'ai,abi->ab',
            filtering_terms.betas,
            tf.einsum('abij,bj->abi', filtering_terms.Q_abij, filtering_terms.betas)
        )

    def _forward_step(self, message_tm1, encoding_potentials_t):
        """
        :param message_tm1: The filtered means (N x D), covariances and precisions (N x D x D) of
         x_tm1|y_1:tm1. N is due to the batchs; D is the dimension of the embedding space.
        :param encoding_potentials_t: The gaussian potentials relating x_t with y_t. Again, there is a
        batched dimension.
        :return: the filtered distribution x_t|y_1:t (means, covariances and precisions), the predicted
        distribution x_t|y_1:tm1, and the covariance matrices cov(x_t, x_tm1| y_1:tm1).
        """
        # Ignore the predicted distribution and the conditional covariance from previous time step
        (means_tm1, covs_tm1, precs_tm1), _, _ = message_tm1
        encoding_means_t, encoding_precs_t = encoding_potentials_t

        # The means_tm1 is N x D (N is due to batches). Transform each vector (D, ) into a
        # row vector (1, D) by adding a new axis
        expanded_means_tm1 = means_tm1[:, tf.newaxis, :]
        means_t_given_tm1, covs_t_t_given_tm1, precs_t_t_given_tm1, covs_t_tm1_given_tm1 = (
            tf.map_fn(
                lambda x: self.predict_xt_given_tm1(mean_tm1=x[0], cov_tm1=x[1]),
                elems=(expanded_means_tm1, covs_tm1),
                # predict_xt_given_tm1 returns
                # mean_t_given_tm1, cov_t_t_given_tm1, tf.linalg.inv(cov_t_t_given_tm1), cov_t_tm1_given_tm1
                # Hence:
                dtype=(tf_floatx(), tf_floatx(), tf_floatx(), tf_floatx()),
                name='forward_predict_map'
            )
        )
        # TODO: there is an squeeze here. Sometimes we expand, sometimes we squeeze. Unify approach
        # to avoid unnecessary operations
        means_t_given_tm1 = tf.squeeze(means_t_given_tm1, axis=1)
        return (
            multiply_gaussians(means_t_given_tm1, precs_t_t_given_tm1,
                               encoding_means_t, encoding_precs_t),
            (means_t_given_tm1, covs_t_t_given_tm1, precs_t_t_given_tm1),
            covs_t_tm1_given_tm1
        )

    def _forward_pass(self, encoding_means, encoding_covs, initial_mean, initial_prec):
        """
        :return
        * The mean, covs and precs of the filtered distributions (x_t|y_1:t) = [x1|y1:1, x2|y1:2, ...]
        * The mean, covs and precs of the predicted distributions (x_t|y_1:tm1) = [x2|y1, x3|y1:2, ...]
        * The covariances cov(x_t, x_tm1| y_1:tm1) = [cov_2,1|1:1, cov_3,2|2:2]
        The dimensions are arranged as [temporal_dimensions, batch, D (or DxD)]
        """
        encoding_precs = tf.map_fn(tf.linalg.inv, encoding_covs, name='encoding_precisions')

        # Build the first message combining the encoding stats at t=0 and the initial stats
        messages_mean_t0, messages_cov_t0, messages_prec_t0 = (
            multiply_gaussians(initial_mean, initial_prec, encoding_means[:, 0, :], encoding_precs[:, 0, ...])
        )
        # Transpose the encoding potentials so that the temporal axis is the left most (needed for the scan). We call
        temporal_encoding_means = tf.transpose(encoding_means, [1, 0, 2])
        temporal_encoding_precs = tf.transpose(encoding_precs, [1, 0, 2, 3])

        filtered_distributions, predicted_distributions, conditional_covs = tf.scan(
            self._forward_step,
            elems=[temporal_encoding_means[1:, ...], temporal_encoding_precs[1:, ...]],
            # Initialize the first argument of _forward_steps (the "accumulated args"). These consists of
            # a[0]: The message from the past m_tm1->t (mean, cov, prec)
            # a[1]: the belief state x_t|y1:tm1 (mean, cov, prec)
            # a[2]: cov_t_tm1_given_tm1:
            initializer=(
                (messages_mean_t0, messages_cov_t0, messages_prec_t0),
                (tf.zeros_like(messages_mean_t0), tf.zeros_like(messages_cov_t0), tf.zeros_like(messages_prec_t0)),
                tf.zeros_like(messages_cov_t0)
            ),
            name='forwards_messages_scan',
        )

        return (
            # The filtered distributions (x_t|y_1:t)
            (
                tf.concat([tf.expand_dims(messages_mean_t0, 0), filtered_distributions[0]], axis=0),  # means
                tf.concat([tf.expand_dims(messages_cov_t0, 0), filtered_distributions[1]], axis=0),  # covs
                tf.concat([tf.expand_dims(messages_prec_t0, 0), filtered_distributions[2]], axis=0)  # precs,
            ),
            #  The predicted distributions (x_t|y_1:tm1)
            predicted_distributions,
            # The covariances cov(x_t, x_tm1| y_1:tm1)
            conditional_covs
        )

    def _backward_pass(self, filtered_distros, predicted_distros, conditional_covs):
        """
        :param filtered_distros: Filtered distributions as output by the forward pass: (means, covs, precs).
        :param predicted_distros: Predicted distributions as output by the forward pass: (means, covs, precs).
        :param conditional_covs: conditional covariances as output by the forward pass.
        :return: Return samples as [nb_samples, batch, time, dimension]
        """
        filtered_means, filtered_covs, filtered_precs = filtered_distros
        predicted_means, predicted_covs, predicted_precs = predicted_distros

        last_sampling_mean = tf.repeat(filtered_means[-1:, ...], self.nb_samples, 0)
        last_sampling_cov = filtered_covs[-1, ...]

        last_time_distro = tfp.distributions.MultivariateNormalFullCovariance(
            loc=filtered_means[-1, ...],
            covariance_matrix=last_sampling_cov
        )
        last_entropy = last_time_distro.entropy()
        last_samples = last_time_distro.sample(self.nb_samples)  # resulting shape is [nb_sample, batch, dimension]

        def backwards_step(future_info, forward_distributions):
            """
            :param future_info: samples, entropy and sampling distros from the future tp1
            :param forward_distributions: distributions computed during the forward pass which are neccessary
            for drawing the samples. Arranged as
            (filtered_means_t, filtered_cov_t, predicted_mean_tp1, predicted_prec_tp1, conditional_tp1_t_gt)
            """
            samples_tp1, entropy_tp1, _, _ = future_info
            (filtered_means_t, filtered_cov_t,
             predicted_mean_tp1, predicted_prec_tp1, conditional_tp1_t_gt) = forward_distributions

            # Sampling
            # J_t is "ordered" as tp1_t
            J_t = tf.einsum('bij,bjk->bik', predicted_prec_tp1, conditional_tp1_t_gt)
            # s is the sample_number, b the batch:
            conditional_mean_t = (
                    filtered_means_t[tf.newaxis, ...] +
                    tf.einsum('sbi,bij->sbj', samples_tp1 - predicted_mean_tp1[tf.newaxis, ...], J_t)
            )
            conditional_cov_t = filtered_cov_t - tf.einsum('bji,bjk->bik', J_t, conditional_tp1_t_gt)
            # conditional_cov_t = tf.map_fn(lambda cov: cov + vaele_jitter() * tf.eye(*cov.shape, dtype=tf_floatx()), conditional_cov_t)
            distro_time_t = tfp.distributions.MultivariateNormalFullCovariance(
                loc=conditional_mean_t,
                covariance_matrix=conditional_cov_t
            )
            # resulting shape of samples is [nb_sample, batch, dimension]
            # likewise, the entropy will have shape (nb_sample, batch). But the nb_sample is redundant, since
            # the entropy only depends on the covariance, which is the sample for all the nb_samples. Therefore
            # (and for consistency with the last_time_entropy), we only take the first entropy
            new_samples = distro_time_t.sample()
            acc_entropy = distro_time_t.entropy()[0, ...] + entropy_tp1

            # # Smoothed distribution
            # smoothed_mean_t = (
            #         filtered_means_t[tf.newaxis, ...] +
            #         tf.einsum('sbi,bij->sbj', smoothed_mean_tp1 - predicted_mean_tp1[tf.newaxis, ...], J_t)
            # )
            # smoothed_cov_t = filtered_cov_t + tf.einsum(
            #     'bij,bjk->bik',
            #     tf.einsum('bji,bjk->bik', J_t, smoothed_cov_tp1 - tf.linalg.inv(predicted_prec_tp1)),
            #     J_t
            # )
            #
            # Two slice distro
            # slice_means = tf.concat(smoothed_mean_t, smoothed_mean_tp1)
            # slice_cov_tp1_t = tf.linalg.matmul(smoothed_cov_tp1, J_t)
            # slice_cov = tf.concat([
            # tf.concat([smoothed_cov_t, tf.transpose(slice_cov_tp1_t, [0, 2, 1])], axis=1),
            # tf.concat([slice_cov_tp1_t, smoothed_cov_tp1], axis=1)],
            # axis=0
            #)


            return (
                new_samples, acc_entropy,
                conditional_mean_t, conditional_cov_t
            )


        samples, acc_entropies, sampling_means, sampling_covs = tf.scan(
            backwards_step,
            elems=(
                filtered_means[:-1, ...],  # all but the last one
                filtered_covs[:-1, ...],     # all but the last one
                predicted_means, predicted_precs, conditional_covs
            ),
            initializer=(
                last_samples, last_entropy, last_sampling_mean, last_sampling_cov
            ),
            reverse=True
        )

        # Extract the first acc_entropies, which sums all the individual entropies (note that scan is reversed)
        acc_entropy = acc_entropies[0]
        samples = tf.transpose(tf.concat([samples, last_samples[tf.newaxis, ...]], axis=0), [1, 2, 0, 3])
        sampling_means = tf.transpose(tf.concat([sampling_means, last_sampling_mean[tf.newaxis, ...]], axis=0),
                                      [1, 2, 0, 3])
        sampling_covs = tf.transpose(tf.concat([sampling_covs, last_sampling_cov[tf.newaxis, ...]], axis=0),
                                      [1, 0, 2, 3])
        return samples, acc_entropy, (sampling_means, sampling_covs)

    def forward_backward(self, encoding_means, encoding_covs, initial_mean, initial_prec):
        filtered_distros, predicted_distros, conditional_covs = (
            self._forward_pass(encoding_means, encoding_covs, initial_mean, initial_prec)
        )
        final_state_mean = filtered_distros[0][-1, ...]
        final_state_prec = filtered_distros[2][-1, ...]
        samples, entropy, sampling_distro = self._backward_pass(filtered_distros, predicted_distros, conditional_covs)
        return samples, entropy, sampling_distro, final_state_mean, final_state_prec



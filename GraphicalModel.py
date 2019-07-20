from nnets.Mlp import Mlp, DiagCovMlp, DoubleMlp, DoubleVarianceMlp, StackedRnn, DiagCovStackedRnn
from MessagePassingAlgorithm import MessagePassingAlgorithm
import numpy as np
from SdeModel import SdeModel
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalDiag
from utils.decorators import tf_property, define_scope
from utils.tf_utils import floatx, tf_floatx, to_floatx, tf_to_floatx


class GraphicalModel(object):
    def __init__(self, inputs, training, sde_inputs, inferred_samples, initial_means, initial_precs,
                 use_initial_state, rnn_initial_states, experiment, optimizers):
        self.inputs = inputs
        self.target_y = inputs[..., -1:]
        self.training = training
        self.sde_inputs = sde_inputs
        self.inferred_samples = inferred_samples
        self.initial_means = initial_means
        self.initial_precs = initial_precs
        self.use_initial_state = use_initial_state
        self.experiment = experiment
        # The rnn states initialization must go after setting self.experiment, since the mean/cov_encoder
        # attribute access it
        self.rnn_initial_states = rnn_initial_states
        self.rnn_output_states = [self.mean_encoder.final_hidden_state, self.mean_encoder.final_state,
                                  self.cov_encoder.final_hidden_state, self.cov_encoder.final_state]

        self.sde_optimizer, self.nnets_optimizer, self.hp_optimizer = optimizers
        self.natgrad_scale = 1.  # set to nb_batches to scale the natural gradients
        # tf properties: Force its creation
        self.mean_encoder
        self.cov_encoder
        self.sde
        self.inference
        self.samples
        self.entropy
        self.mean_decoder
        self.cov_decoder
        self.sde_lower_bound
        self.nnet_lower_bound
        self.optimize_nnets
        self.optimize_hps
        self.optimize_sde_distribution

    @tf_property
    def diff_target(self):
        dy = (self.target_y[:, 1:, :] - self.target_y[:, :-1, :]) / self.experiment.sampling_time
        return tf.concat([dy, dy[:, -1:, :]], axis=1, name='dy_concat')

    @tf_property("encoders/mean")
    def mean_encoder(self):
        return StackedRnn(self.inputs, self.training,
                          input_shape=[self.experiment.len_tbptt, self.experiment.input_dimension],
                          configuration=[self.experiment.encoding_hidden_units,
                                         self.experiment.embedding_dim],
                          initial_hidden_state=self.rnn_initial_states[0],
                          initial_output_state=self.rnn_initial_states[1])
        # return Mlp(
        #     self.inputs, self.training,
        #     input_shape=[self.experiment.len_tbptt, self.experiment.input_dimension],
        #     configuration=[self.experiment.encoding_hidden_units, self.experiment.embedding_dim],
        #     dropout=self.experiment.encoders_dropout
        # )

    @tf_property("encoders/cov")
    def cov_encoder(self):
        return DiagCovStackedRnn(self.inputs, self.training,
                                 input_shape=[self.experiment.len_tbptt,
                                              self.experiment.input_dimension],
                                 configuration=[self.experiment.encoding_hidden_units,
                                                self.experiment.embedding_dim],
                                 initial_hidden_state=self.rnn_initial_states[2],
                                 initial_output_state=self.rnn_initial_states[3])
        # return DiagCovMlp(
        #     self.inputs, self.training,
        #     input_shape=[self.experiment.len_tbptt, self.experiment.input_dimension],
        #     configuration=[self.experiment.encoding_hidden_units, self.experiment.embedding_dim],
        #     dropout=self.experiment.encoders_dropout
        # )

    @tf_property
    def sde(self):
        return SdeModel(self.sde_inputs, self.experiment.drift_kernel_params,
                        self.experiment.alphas, self.experiment.betas,
                        np.random.random((self.experiment.nb_inducing_points,
                                          self.experiment.embedding_dim))
                        )

    @tf_property
    def inference(self):
        return MessagePassingAlgorithm(self.sde, self.mean_encoder, self.cov_encoder,
                                       self.initial_means, self.initial_precs,
                                       self.use_initial_state, self.experiment.nb_samples)

    @tf_property
    def samples(self):
        """
        :return: Simulated samples arranged as (batch_size, nb_samples, len_tbptt, embedding_dimension)
        """
        return self.inference.samples

    @tf_property
    def grouped_samples(self):
        """
        Group the samples identified by batch number and nb_samples into a single dimension (leading dimension)

        Since the costs are calculated by averaging through the batch and the samples of each batch, this is equivalent
        to treat average through all samples
        :return: Simulated samples arranged as (-1, len_tbptt, embedding_dimension)
        """
        return tf.reshape(self.inference.samples, (-1, self.experiment.len_tbptt,
                                                   self.experiment.embedding_dim),
                          name="group_samples")

    @tf_property
    def entropy(self):
        return self.inference.entropy

    @tf_property("decoders/mean")
    def mean_decoder(self):
        return DoubleMlp(
            self.samples, self.training,
            [self.experiment.len_tbptt, self.experiment.embedding_dim],
            [self.experiment.decoding_hidden_units, 2],
            dropout=self.experiment.decoders_dropout
        )

    @tf_property("decoders/cov")
    def cov_decoder(self):
        return DoubleVarianceMlp(
            self.samples, self.training,
            [self.experiment.len_tbptt, self.experiment.embedding_dim],
            [self.experiment.decoding_hidden_units, 2],
            dropout=self.experiment.decoders_dropout
        )

    def _trajectory_loglikelihood(self, trajectory, mvn):
        # TODO: study this stop gradient
        means, vars, variance_penalty = self.sde.get_induced_drift_stats(
            tf.stop_gradient(trajectory[:-1, :])
        )
        with tf.name_scope('diff'):
            dx = trajectory[1:, :] - trajectory[:-1, :]
        with tf.name_scope('phase_space_lk'):
            log_likelihood = tf.reduce_sum(mvn.log_prob(dx - tf.transpose(means)))
        # log_likelihood = tf.Print(log_likelihood, [tf.reduce_mean(tf.square(dx - tf.transpose(means)), axis=0)], "Vars: ")
        return log_likelihood, -0.5 * variance_penalty

    def _mc_embedding_loglikelihood(self, samples):
        mvn = MultivariateNormalDiag(
            tf.zeros(self.experiment.embedding_dim, dtype=tf_floatx()),
            self.sde.sqrt_expected_diffusion,
            name='phase_space_mvn'
        )
        lks_x, minus_var_penalties = tf.map_fn(
            lambda x: self._trajectory_loglikelihood(x, mvn),
            elems=samples,
            dtype=(tf_floatx(), tf_floatx()),
            name='mc_lk_x_map'
        )
        # The parameter that depends on the Gamma distribution is the same of all
        # trajectories and, therefore, we compute it outside of the loop
        # This could be further extracted...
        gamma_term = 0.5 * self.experiment.len_tbptt * tf.reduce_sum(
            self.sde.expected_log_precision - tf.log(self.sde.expected_precision)
        )
        lk_x = tf.reduce_mean(lks_x)
        minus_var_penalty = tf.reduce_mean(minus_var_penalties)

        tf.summary.scalar('variational_lk_x', lk_x)
        tf.summary.scalar('var_penalty', -minus_var_penalty)
        tf.summary.scalar('prec_factor', gamma_term)

        # TODO
        return lk_x + 0 * minus_var_penalty + gamma_term

    @tf_property
    def sde_lower_bound(self):
        grouped_isamples = tf.reshape(self.inferred_samples,
                                      (-1, self.experiment.len_tbptt, self.experiment.embedding_dim),
                                      name="group_inferred_samples")
        # grouped_isamples = tf.Print(grouped_isamples, [tf.zeros(10)], "AAAAAAAAAAAAAAAAAAAAAAAAaaAAAAAAAAAAAAAAAAAAAAAAAAA")
        sde_lower_bound = (
                self.natgrad_scale * self._mc_embedding_loglikelihood(grouped_isamples) - self.sde.kullback_leibler
        )
        # sde_lower_bound = tf.Print(sde_lower_bound, [tf.zeros(10)], "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        return sde_lower_bound

    @tf_property
    def nnet_lower_bound(self):
        lk_y = self._avg_marginal_loglikelihood_batch_ys()
        lk_x = self._mc_embedding_loglikelihood(self.grouped_samples)
        mean_entropy = tf.reduce_mean(self.entropy)

        tf.summary.scalar('lk_y', lk_y)
        tf.summary.scalar('lk_x', lk_x)
        tf.summary.scalar('Entropy', mean_entropy)
        tf.summary.scalar('log_beta/alpha',
                          tf_to_floatx(self.experiment.len_tbptt) * (
                                  0.5 * self.experiment.embedding_dim * tf_to_floatx(1 + np.log(2 * np.pi)) +
                                  0.5 * tf.linalg.logdet(tf.diag(self.sde.expected_diffusion))
                          )
                          )
        with tf.name_scope("nnet_lower_bound"):
            lb = self.natgrad_scale * (lk_y + lk_x + mean_entropy)
            tf.summary.scalar('nnet_lower_bound', lb)
        return lb

    def _avg_marginal_loglikelihood_batch_ys(self):
        random_vars = tf.concat([self.target_y, self.diff_target], axis=-1)
        lks_y_given_x = tf.map_fn(
            self._marginal_loglikelihood_y,
            elems=[random_vars, self.mean_decoder.output, self.cov_decoder.output],
            dtype=tf_floatx(),
            name='map_marginal_lk_ys'
        )
        return tf.reduce_mean(lks_y_given_x)

    def _marginal_loglikelihood_y(self, args):
        time_series, means, covs = args
        lks = tf.map_fn(
            lambda x: self._loglikelihood_y_given_x(time_series, *x),
            elems=[means, covs],
            dtype=tf_floatx(),
            name='mc_lk_y_given_x'
        )
        return tf.reduce_mean(lks)

    def _loglikelihood_y_given_x(self, random_vars, decoding_means, decoding_covs):
        # This map iterates through all the time steps of the observed series y and the decoding means/covs obtained
        # for a single embedding trajectory
        mvn = MultivariateNormalDiag(decoding_means, tf.sqrt(decoding_covs), name='conditional_observations_mvn')
        return tf.reduce_sum(mvn.log_prob(random_vars))

    @tf_property
    def optimize_nnets(self):
        with tf.name_scope("nnets_optimization"):
            return self.nnets_optimizer.minimize_with_clipping(
                -self.nnet_lower_bound, var_list=(
                        self.mean_encoder.params +
                        self.cov_encoder.params +
                        self.mean_decoder.params + self.cov_decoder.params
                )
            )

    @tf_property
    def optimize_hps(self):
        with tf.name_scope("hp_optimization"):
            return self.hp_optimizer.minimize_with_clipping(-self.sde_lower_bound, var_list=(
                    self.sde.kernel_params + [self.sde.inducing_points]
            ))

    # @tf_property
    # def optimize_nnets_and_hyperparams(self):
    #     with tf.name_scope("nnets_gradients"):
    #         nnets_grads = self.punctual_optimizer.compute_gradients(
    #             -self.nnet_lower_bound, var_list=(
    #                 self.mean_encoder.params +
    #                 self.cov_encoder.params +
    #                 self.mean_decoder.params + self.cov_decoder.params
    #             )
    #         )
    #     with tf.name_scope("hp_gradients"):
    #         hp_grads = self.punctual_optimizer.compute_gradients(-self.sde_lower_bound, var_list=(
    #             self.sde.kernel_params + [self.sde.inducing_points]
    #         ))
    #     grads_and_vars = nnets_grads + hp_grads
    #     with tf.name_scope("clip_gradients"):
    #         grads_and_vars = self.punctual_optimizer.clip_gradients(grads_and_vars)
    #     return self.punctual_optimizer.apply_gradients(grads_and_vars, name='nnets_and_hp_grads')

    @tf_property
    def optimize_sde_distribution(self):
        grads = self._sde_natural_gradients
        # Create the pairs (grads, variable)
        ngrad_updates = list(zip(*[grads, self.sde.distribution_params]))
        with tf.name_scope("clip_gradients"):
            clipped_gradients_and_vars = self.sde_optimizer.clip_gradients(ngrad_updates)
        return self.sde_optimizer.apply_gradients(clipped_gradients_and_vars)

    @staticmethod
    @define_scope("inverse_fisher_blocks")
    def get_inv_fisher_blocks(psi_0s, psi_1s, covs, precs, nb_inducing_points):
        @define_scope("single_dim_inverse_fisher_block")
        def get_fisher_matrix(params):
            a, m, cov, prec = params
            alpha = tf.exp(a)
            beta = tf.exp(a - m)

            f11_inv = tf.diag([
                1. / ((alpha ** 2) * tf.polygamma(to_floatx(1.), alpha) - alpha),
                1. / alpha
            ])
            # original implementation (to invert later!) f22 = (alpha / beta) * prec
            f22_inv = (beta / alpha) * cov
            # f22_inv = (beta / alpha) * tf.diag(tf.diag_part(cov))
            cov_tensor = tf.expand_dims(tf.expand_dims(cov, 2), 3)
            cov_tensor = cov_tensor * tf.transpose(cov_tensor)

            f33 = 0.25 * (tf.transpose(cov_tensor, [0, 2, 1, 3]) + tf.transpose(cov_tensor, [0, 3, 1, 2]))
            f33 = tf.reshape(f33, (nb_inducing_points ** 2, -1))
            # f22_inv = tf.Print(f22_inv, [f22_inv], "f22_inv")
            return f11_inv, f22_inv, tf.matrix_inverse(f33)

        return tf.map_fn(
            get_fisher_matrix,
            elems=[psi_0s, psi_1s, covs, precs],
            dtype=(tf_floatx(), tf_floatx(), tf_floatx()),
            name='map_inv_fisher_blocks')

    @tf_property
    def _sde_natural_gradients(self):
        inv_F11, inv_F22, inv_F33 = self.get_inv_fisher_blocks(
            self.sde.distribution_params[0],
            self.sde.distribution_params[1],
            self.sde.standard_params["covs"],
            self.sde.distribution_params[3],
            self.sde.nb_inducing_points
        )
        with tf.name_scope("sde_gradients"):
            grads = tf.gradients(-self.sde_lower_bound, self.sde.distribution_params)

        with tf.name_scope("precondition_gradients"):
            egrads_0 = tf.expand_dims(grads[0], 1)
            egrads_1 = tf.expand_dims(grads[1], 1)

            block_0_grads = tf.concat([egrads_0, egrads_1], axis=1)
            block_0_nat_grads = tf.reduce_sum(inv_F11 * tf.expand_dims(block_0_grads, 1), axis=2)

            mean_nat_grads = tf.reduce_sum(inv_F22 * tf.expand_dims(grads[2], 1), 2)
            # mean_nat_grads = grads[2]
            # mean_nat_grads = tf.Print(mean_nat_grads, [mean_nat_grads, grads[2]], "ng_vs_g:")

            flattened_prec_nat_grads = tf.reduce_sum(
                inv_F33 * tf.expand_dims(tf.reshape(grads[3], (self.experiment.embedding_dim, -1)), 1), 2)
            prec_nat_grads = tf.reshape(flattened_prec_nat_grads,
                                        (self.experiment.embedding_dim, self.sde.nb_inducing_points, -1))

            return [block_0_nat_grads[:, 0], block_0_nat_grads[:, 1], mean_nat_grads, prec_nat_grads]

            # mean_nat_grads = tf.Print(mean_nat_grads,[grads[0], block_0_nat_grads[:, 0]], "grads_0")
            # mean_nat_grads = tf.Print(mean_nat_grads,[grads[1], block_0_nat_grads[:, 1]], "grads_1")
            # return [grads[0], grads[1], mean_nat_grads, prec_nat_grads]


def update_graph(sess):
    return tf.summary.FileWriter('/tmp/graph', sess.graph)

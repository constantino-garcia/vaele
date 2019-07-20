from Kernel import GaussianKernel
import numpy as np
from config import floatx, tf_floatx, get_session
import tensorflow as tf
from utils.generic_utils import normal_gamma_kullback_leibler
from utils.tf_utils import tf_get_value, tf_set_value, variable_summaries
from utils.decorators import tf_property, define_scope


class SdeModel(object):
    def __init__(self, inputs, drift_kernels_params, alphas, betas, inducing_points):
        # Some basic assertions
        assert len(alphas) == len(betas), 'len(alphas) != len(betas)'
        if np.any(alphas <= 0.) or np.any(betas <= 0.):
            raise Exception("All 'alphas' and 'betas' should be > 0.0")
        assert len(drift_kernels_params) == len(alphas), (
            "len(drift_kernel_params) !=  input dimension (inferred from alphas)"
        )

        self.input = inputs
        self.input_dimension = len(alphas)
        # Diffusion
        self.alphas = alphas
        self.betas = betas
        # Drift and kernels
        self.drift_kernels_params = drift_kernels_params
        self.inducing_points = tf.get_variable(name='inducing_points',
                                               initializer=tf.constant(inducing_points.astype(floatx()))
                                               )
        self.nb_inducing_points = len(inducing_points)

        self.kernels, pseudo_inputs_matrices = (
            self._create_kernels(self.inducing_points, self.inducing_points, self.drift_kernels_params)
        )
        with tf.name_scope('pi_matrices'):
            self.pseudo_inputs_matrices = pseudo_inputs_matrices
        # Compute and store the inverses of the pseudo_inputs_matrices, since they are required to compute the expected
        # drift.
        with tf.name_scope('inverse_pi_matrices'):
            self.inv_pseudo_inputs_matrices = tf.map_fn(
                lambda matrix: tf.matrix_inverse(matrix),
                elems=self.pseudo_inputs_matrices,
                dtype=tf_floatx(),
                name='pi_matrix_inversion_map'
            )
        # Since the Gaussian Kernel is stationary, we can use the pseudo_input_kernel to compute the variance that
        # will result from computing K_i(x, x) for each dimension for all x.
        # self.marginal_variances is a vector of dimension self.input_dimension.
        with tf.name_scope('kernel_variances'):
            self.kernel_variances = tf.stack([k.variances[0] for k in self.kernels])

        self.distribution_params = [
            tf.get_variable(name='psi_0', initializer=tf.constant(self._initialize_psi0())),
            tf.get_variable(name='psi_1', initializer=tf.constant(self._initialize_psi1())),
            tf.get_variable(name='psi_2', initializer=tf.constant(self._initialize_psi2())),
            tf.get_variable(name='psi_3', initializer=tf.constant(self._initialize_psi3()))
        ]
        # Compute the standard params of a Normal-Gamma distribution from the optimizable params.
        # The initial_standard_params are stored to compute the KL divergence with respect to them afterwards.
        self.standard_params =self._get_standard_params()
        self.initial_standard_params = self._set_prior_params(self.alphas, self.betas)

        # Store all optimizable params for convenience
        self.kernel_params = []
        for kernel in self.kernels:
            self.kernel_params += kernel.params

        self.params = self.kernel_params + self.distribution_params
        # Set as output the expected drift
        self.output = self.get_expected_drift(self.input)

        # Force the creation of the tf_attributes
        self.expected_diffusion
        self.sqrt_expected_diffusion
        self.expected_precision
        self.expected_log_precision
        self.kullback_leibler

    def set_inducing_points(self, inducing_points):
        inducing_points = inducing_points.astype(floatx())
        tf_set_value(self.inducing_points, inducing_points)

    def _create_kernels(self, x, y, drift_kernels_params):
        """
        Creates a Gaussian Kernel for each dimension of the SDE using inputs x, y and parameters from
        drift_kernel_params. The dimension is inferred from drift_kernel_params
        :param x: TF matrix. First input to the kernel. Hence the output will have the shape (nb_rows_x, nb_rows_y).
        :param y: TF matrix. Second input to the kernel. Hence the output will have the shape (nb_rows_x, nb_rows_y).
        :param drift_kernels_params: A vector of dictionaries containing the parameters of each dimension of the SDE.
        :return: Vector of GaussianKernels and the stacked ouputs.
        """
        kernels = []
        for i, drift_kernel_params in enumerate(drift_kernels_params):
            with tf.name_scope('gaussian_kernel_' + str(i)):
                kernels.append(GaussianKernel(x, y, **drift_kernel_params))

        outputs = tf.stack([kernel.output for kernel in kernels], name='stack_kernels')
        return kernels, outputs

    def _initialize_psi0(self):
        return np.log(self.alphas).astype(floatx())

    def _initialize_psi1(self):
        return np.log(self.alphas / self.betas).astype(floatx())

    def _initialize_psi2(self):
        return np.zeros((self.input_dimension, self.nb_inducing_points), dtype=floatx())
        # return np.random.random((self.input_dimension, self.nb_inducing_points)).astype(floatx())

    def _initialize_psi3(self):
        initial_psi3 = []
        for kernel in self.kernels:
            prec_matrix = np.diag(1. / np.repeat(kernel.max_amplitude, self.nb_inducing_points))
            prec_matrix = (prec_matrix + prec_matrix.T) / 2.0
            initial_psi3.append(prec_matrix)
        return np.stack(initial_psi3).astype(floatx())

    def _set_prior_params(self, alphas, betas):
        # Store the initial_standard params as shared variables but the pseudo_inputs covariances,
        # which depend on the hyperparameters of the kernel.
        with tf.name_scope('prior_params'):
            standard_params = {
                'alphas': tf.Variable(alphas.astype(floatx()), name='initial_alphas', trainable=False),
                'betas': tf.Variable(betas.astype(floatx()), name='initial_betas', trainable=False),
                'means': tf.Variable(np.zeros((self.input_dimension, self.nb_inducing_points)).astype(floatx()),
                                     name='initial_means', trainable=False),
                'covs': self.pseudo_inputs_matrices
            }
        return standard_params

    def _get_standard_params(self):
        """
        Computes the standard Normal-Gamma parameters after the optimizable parameters have been set.
        :return: A dictionary of tf.Variable parameters representing the  standard parameters of a Normal-Gamma
        distribution an another dictionary with the result of its evaluation. The keys of the distribution are
        'alphas', 'betas', 'means', 'covs'.
        """
        standard_params = dict()

        with tf.name_scope('standard_params'):
            with tf.name_scope('alphas'):
                standard_params['alphas'] = tf.exp(self.distribution_params[0])
            with tf.name_scope('betas'):
                standard_params['betas'] = tf.exp(self.distribution_params[0] - self.distribution_params[1])
            with tf.name_scope('means'):
                standard_params['means'] = self.distribution_params[2]
            with tf.name_scope('covs'):
                standard_params['covs'] = tf.map_fn(
                    lambda precision_matrix: tf.matrix_inverse(precision_matrix),
                    elems=self.distribution_params[3],
                    name='invert_precision'
                )
        return standard_params

    def compute_inducing_matrices(self, inputs):
        outputs = tf.stack([kernel.compute(inputs, self.inducing_points) for kernel in self.kernels])
        return outputs

    def get_transformation_matrices(self, inducing_matrices):
        def get_transformation_matrix(inducing_matrix, inv_pseudo_inputs_matrix):
            return tf.matmul(inducing_matrix, inv_pseudo_inputs_matrix)

        transformation_matrices = tf.map_fn(
            lambda matrices: get_transformation_matrix(*matrices),
            elems=[inducing_matrices, self.inv_pseudo_inputs_matrices],
            dtype=tf_floatx(),
            name='transformation_matrix_map'
        )
        return transformation_matrices

    @tf_property
    def expected_diffusion(self):
        expected_diffusion = 1. / self.expected_precision
        tf.summary.scalar('expected_diffusion_0', expected_diffusion[0])
        tf.summary.scalar('expected_diffusion_1', expected_diffusion[1])
        return expected_diffusion

    @tf_property
    def sqrt_expected_diffusion(self):
        return tf.sqrt(self.expected_diffusion)

    @tf_property
    def expected_precision(self):
        return self.standard_params['alphas'] / self.standard_params['betas']

    @tf_property
    def expected_log_precision(self):
        return tf.digamma(self.standard_params['alphas']) - tf.log(self.standard_params['betas'])

    def get_expected_drift(self, sde_inputs, inducing_matrices=None, transformation_matrices=None):
        with tf.name_scope('drift_expectation'):
            if inducing_matrices is None:
                inducing_matrices = self.compute_inducing_matrices(sde_inputs)
            if transformation_matrices is None:
                transformation_matrices = self.get_transformation_matrices(inducing_matrices)

            return tf.reduce_sum(transformation_matrices * tf.expand_dims(self.standard_params['means'], axis=1),
                                 axis=2, name='E_drift')

    def get_drift_jacobian(self, x):
        """

        :param x: A batch of points in which evaluate the jacobian. The shape is [batch_size, embedding_dim]
        :return:  The jacobian ([embedding_dim, embedding_dim]) evaluated in each of the examples of the batch. The
        output size is [batch_size, embedding_dim, embedding_dim]
        """
        # kxs: [embedding_dim (a K(x, IPs) for each dimension, nb_inducing_points]
        kxs = self.compute_inducing_matrices(x)
        length_scales = tf.stack([k.length_scales for k in self.kernels])
        # delta: [batch_size, inducing_points, embedding_dim]
        delta = tf.expand_dims(self.inducing_points, axis=0) - tf.expand_dims(x, axis=1)

        def differentiate(params):
            length_scale, kx, inv_pi_matrix, mean = params
            # Expand the dimensions of length_scale and kx so that der_term has the same dimensions as delta
            der_term = delta / tf.reshape(tf.square(length_scale), (1, 1, self.input_dimension)) * tf.expand_dims(kx, -1)
            # Tensordot of [batch_size, nb_inducing_points, embedding_dimension] by a vector with shape: [nb_inducing_points],
            # therefore, the output is [nb_dimension, embedding_dimension]
            return tf.tensordot(der_term,
                                tf.squeeze(tf.matmul(inv_pi_matrix, tf.reshape(mean, (-1, 1)))),
                                [[1], [0]])

        # diff has shape [embedding_dimension (1 for each loop), batch_size, embedding_dimension]
        # The first 'embedding_dimension' loops over the functions, whereas that the last 'embedding_dimension' scans
        # each x_i
        diff = tf.map_fn(
            differentiate,
            elems=[length_scales, kxs, self.inv_pseudo_inputs_matrices, self.standard_params["means"]],
            dtype=floatx(),
            name="differentiate_sde_map"
        )
        # Reshape diff so that the output is: [batch_size, nb_of_functions, nb_of_independent_variables] ==
        # [batch_size, embedding_dimension, embedding_dimension]
        return tf.transpose(diff, perm=[1, 0, 2])

    # This is called \hat{P} in our paper.
    def get_precision_by_induced_variance(self, inputs, inducing_matrices=None, transformation_matrices=None):
        if inducing_matrices is None:
            inducing_matrices = self.compute_inducing_matrices(inputs)
        if transformation_matrices is None:
            transformation_matrices = self.get_transformation_matrices(inducing_matrices)

        def compute_precision_by_variance(kernel_variance, transformation_matrix, inducing_matrix):
            precision_by_variance = (
                (kernel_variance - tf.reduce_sum(transformation_matrix * inducing_matrix, axis=1))
            )
            return tf.reshape(precision_by_variance, [-1])

        precision_by_induced_variance = tf.map_fn(
            lambda variables: compute_precision_by_variance(*variables),
            elems=[self.kernel_variances, transformation_matrices, inducing_matrices],
            dtype=tf_floatx()
        )
        return precision_by_induced_variance

    def get_induced_drift_stats(self, inputs):
        """
        A convenient wrapper that calls get_expected_drift, precision_by_induced_variance and get_drift_variance_penalty
        reusing common operations.
        :return: returns the mean (get_expected_drift), get_precision_by_induced_variance
        (precision_by_induced_variance), and a drift variance penalty (get_drift_variance_penalty)
        """
        inducing_matrices = self.compute_inducing_matrices(inputs)
        transformation_matrices = self.get_transformation_matrices(inducing_matrices)

        means = self.get_expected_drift(inputs, inducing_matrices, transformation_matrices)
        variances = self.get_precision_by_induced_variance(inputs, inducing_matrices, transformation_matrices)
        variance_penalty = self.get_drift_variance_penalty(transformation_matrices, variances)

        return means, variances, variance_penalty

    def get_drift_variance_penalty(self, transformation_matrices, variances):
        """
        Calculates a penalty for large drift variance values used during the optimization of the lower bound
        :param transformation_matrices: Transformation matrices.
        :param variances: Variances computed using get_precision_by_induded_variance
        :return: A drift variance penalty.
        """

        def compute_variance_penalty(transformation_matrix, variational_cov):
            tmp = tf.reduce_sum(
                transformation_matrix * tf.matmul(transformation_matrix, variational_cov),
                axis=1
            )
            # tmp = tf.Print(tmp, [tf.shape(tmp)], 'var_penalty_shape')
            return tmp

        # For each dimension (for each drift function, compute its 'variance penalty'.
        variance_penalty = tf.map_fn(
            lambda variables: compute_variance_penalty(*variables),
            elems=[transformation_matrices, self.standard_params['covs']],
            dtype=tf_floatx(),
            name='var_penalty_map'
        )
        # variances = tf.Print(variances, [tf.shape(variances)], 'variances_shape')
        return tf.reduce_sum(variance_penalty) + tf.reduce_sum(variances)

    @tf_property
    def kullback_leibler(self):
        variational_kls = tf.map_fn(
            lambda variables: normal_gamma_kullback_leibler(*variables, dimension=self.nb_inducing_points),
            elems=[
                self.standard_params["alphas"], self.standard_params["betas"],
                self.standard_params["means"], self.standard_params["covs"],
                self.initial_standard_params['alphas'], self.initial_standard_params['betas'],
                self.initial_standard_params['means'], self.initial_standard_params['covs']
            ],
            dtype=tf_floatx(),
            name='kullback_leibler_map'
        )
        kl = tf.reduce_sum(variational_kls, name='sum_KL')
        tf.summary.scalar("kullback_leibler", kl)
        return kl


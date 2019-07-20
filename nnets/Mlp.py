import numpy as np
import tensorflow as tf
from config import tf_floatx, floatx
from utils.tf_utils import vector_to_tril, to_floatx, tf_to_floatx, tf_set_value
from tensorflow.contrib.distributions import fill_triangular

he_init = tf.contrib.layers.variance_scaling_initializer(factor=2.0, mode='FAN_IN', uniform=False, dtype=tf_floatx())
glorot_init = tf.contrib.layers.variance_scaling_initializer(factor=1.0, mode='FAN_AVG', uniform=True,
                                                             dtype=tf_floatx())
xaviern_init = tf.contrib.layers.xavier_initializer(uniform=False, dtype=tf_floatx())
xavieru_init = tf.contrib.layers.xavier_initializer(dtype=tf_floatx())

default_init = glorot_init


class StackedRnn(object):
    def __init__(self, nnet_input, training, input_shape, configuration, initial_hidden_state,
                 initial_output_state):
        self.input = nnet_input
        self.training = training
        self.initial_hidden_state = initial_hidden_state
        self.initial_output_state = initial_output_state
        with tf.variable_scope("hidden"):
            self.hidden_rnn_cell = tf.nn.rnn_cell.GRUCell(configuration[0])
            self.hidden_outputs, self.final_hidden_state = tf.nn.dynamic_rnn(self.hidden_rnn_cell, nnet_input,
                                                                             initial_state=initial_hidden_state,
                                                                             dtype=tf_floatx())

        with tf.variable_scope("output_rnn"):
            self.output_rnn_cell = tf.nn.rnn_cell.GRUCell(configuration[0])
            self.output_rnn, self.final_state = tf.nn.dynamic_rnn(self.output_rnn_cell, self.hidden_outputs,
                                                              initial_state=initial_output_state,
                                                              dtype=tf_floatx())

        # self.output_rnn = self.hidden_outputs
        # self.final_state = tf.zeros_like(self.final_hidden_state)

        with tf.variable_scope("output"):
            self.output_layer = tf.layers.Dense(units=configuration[1], activation=None,
                                                 kernel_initializer=default_init,
                                                 bias_initializer=tf.zeros_initializer(dtype=tf_floatx()),
                                                 dtype=tf_floatx(),
                                                 use_bias=False,
                                                 name='mlp_output_layer')
            self.output = self.output_layer(self.output_rnn)

        self.params = (
                self.hidden_rnn_cell.trainable_variables + self.output_rnn_cell.trainable_variables +
                self.output_layer.trainable_variables
        )


        # self.params = (
        #         self.hidden_rnn_cell.trainable_variables + self.output_rnn_cell.trainable_variables +
        #         self.output_layer.trainable_variables
        # )


class DiagCovStackedRnn(StackedRnn):
    def __init__(self, nnet_input, training, input_shape, configuration, initial_hidden_state,
                 initial_output_state):
            with tf.variable_scope("diag_cov_rnn"):
                super().__init__(nnet_input, training, input_shape, configuration, initial_hidden_state,
                                 initial_output_state)

                self.output = tf.exp(2 * self.output)
                self.output = tf.map_fn(tf.diag, tf.reshape(self.output, (-1, configuration[1])), name='map_create_diagonal_cov')
                self.output = tf.reshape(self.output, (-1, input_shape[0], configuration[1], configuration[1]))



class Mlp(object):
    def __init__(self, nnet_input, training, input_shape, configuration, hidden_activation=tf.nn.elu,
                 output_activation=None,
                 initializer=default_init, bias_initializer=tf.zeros_initializer(dtype=tf_floatx()),
                 l2_reg=to_floatx(0.0), dropout=to_floatx(0.0)):
        self.input_shape = input_shape
        self.configuration = configuration
        [n_hidden, n_out] = configuration

        if l2_reg > 0.0:
            hidden_layer_regularizer = tf.contrib.layers.l1_regularizer(l2_reg)
            output_layer_regularizer = tf.contrib.layers.l1_regularizer(l2_reg)
        else:
            hidden_layer_regularizer = None
            output_layer_regularizer = None
        self.hidden_layer = tf.layers.Dense(units=n_hidden, activation=hidden_activation,
                                            kernel_initializer=initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=hidden_layer_regularizer,
                                            dtype=tf_floatx(),
                                            name='hidden_layer')

        self.output_layer = tf.layers.Dense(units=n_out, activation=output_activation,
                                            kernel_initializer=initializer,
                                            bias_initializer=bias_initializer,
                                            kernel_regularizer=output_layer_regularizer,
                                            dtype=tf_floatx(),
                                            name='output_layer')

        if dropout > 0.0:
            self.hidden_dropout = tf.layers.Dropout(rate=dropout)
            # self.output_dropout = tf.layers.Dropout(rate=dropout)
        else:
            self.hidden_dropout = None
            # self.output_dropout = None

        self.input = nnet_input
        self.training = training
        self.output = self.compute(self.input, self.training)
        self.updates = []
        self.params = (
                self.hidden_layer.trainable_variables + self.output_layer.trainable_variables
        )
        tf.summary.histogram('hidden_layer_weights', self.hidden_layer.kernel)
        tf.summary.histogram('hidden_layer_bias', self.hidden_layer.bias)
        tf.summary.histogram('output_layer_weights', self.output_layer.kernel)
        tf.summary.histogram('output_layer_bias', self.output_layer.bias)
        tf.summary.histogram('outputs', self.output)

    def compute(self, inputs, training=None):
        if training is None:
            training = self.training
        hidden_layer_outputs = self.hidden_layer(inputs)
        if self.hidden_dropout:
            hidden_layer_outputs = self.hidden_dropout(hidden_layer_outputs, training=training)
        return self.output_layer(hidden_layer_outputs)
        # if self.output_dropout:
        #     outputs = self.output_dropout(outputs, training=training)
        # return outputs


class DoubleMlp(object):
    def __init__(self, nnet_input, training, input_shape, configuration, **kwargs):
        real_configuration = configuration.copy()
        real_configuration[-1] = real_configuration[-1] // 2
        with tf.name_scope('mlp_1'):
            self.mlp_1 = Mlp(nnet_input, training, input_shape, real_configuration, **kwargs)
        with tf.name_scope('mlp_2'):
            self.mlp_2 = Mlp(nnet_input, training, input_shape, real_configuration, **kwargs)

        self.input = nnet_input
        self.training = training
        self.params = self.mlp_1.params + self.mlp_2.params
        self.output = self.compute(self.input, self.training)

    def compute(self, inputs, training=None):
        return tf.concat([self.mlp_1.compute(inputs, training), self.mlp_2.compute(inputs, training)], axis=-1)


class VarianceMlp(Mlp):
    def __init__(self, nnet_input, training, input_shape, configuration, **kwargs):
        super().__init__(nnet_input, training, input_shape, configuration, **kwargs)

    def compute(self, inputs, training=None):
        return tf.exp(2 * super().compute(inputs, training))


class DoubleVarianceMlp(object):
    def __init__(self, nnet_input, training, input_shape, configuration, **kwargs):
        real_configuration = configuration.copy()
        real_configuration[-1] = real_configuration[-1] // 2
        with tf.name_scope('mlp_1'):
            self.mlp_1 = VarianceMlp(nnet_input, training, input_shape, real_configuration, **kwargs)
        with tf.name_scope('mlp_2'):
            self.mlp_2 = VarianceMlp(nnet_input, training, input_shape, real_configuration, **kwargs)

        self.input = nnet_input
        self.training = training
        self.params = self.mlp_1.params + self.mlp_2.params
        self.output = self.compute(self.input, self.training)

    def compute(self, inputs, training=None):
        return tf.concat([self.mlp_1.compute(inputs, training), self.mlp_2.compute(inputs, training)], axis=-1)


class CovMlp(Mlp):
    def __init__(self, inputs, training, input_shape, configuration, epsilon=0.0, **kwargs):
        self.output_dimension = configuration[-1]
        self.epsilon = to_floatx(epsilon)
        # The output shall be a vectorized lower triangular matrix (cholesky like)
        self.nb_chol_params = (self.output_dimension * (self.output_dimension + 1)) // 2
        real_configuration = configuration.copy()
        real_configuration[-1] = self.nb_chol_params

        super().__init__(inputs, training, input_shape, real_configuration, **kwargs)

    def compute(self, inputs, training=None):
        if training is None:
            training = self.training
        outputs = super().compute(inputs, training)

        if len(self.input_shape) >= 2:
            outputs = tf.reshape(outputs, (-1, self.nb_chol_params))
        outputs = self.network_output_to_cov_matrices(outputs, self.output_dimension, self.epsilon)
        if len(self.input_shape) >= 2:
            outputs = tf.reshape(outputs, (-1, self.input_shape[0], self.output_dimension, self.output_dimension))
        return outputs

    @staticmethod
    def network_output_to_cov_matrices(output, output_dimension, epsilon):
        with tf.name_scope('reshape_to_cov_matrix'):
            cholesky_tril = vector_to_tril(output, output_dimension)
            # cholesky_tril = tf.map_fn(fill_triangular, output)
            output = tf.reduce_sum(tf.expand_dims(cholesky_tril, axis=1) * tf.expand_dims(cholesky_tril, axis=2),
                                   axis=3)
            # output = tf.Print(output, [output[0]], 'output: ', summarize=9)
            if epsilon > 0.0:
                return output + tf.tile(tf.expand_dims(epsilon * tf.eye(output_dimension, dtype=tf_floatx()), 0),
                                        multiples=[tf.shape(output)[0], 1, 1])
            else:
                return output


class DiagCovMlp(VarianceMlp):
    def __init__(self, inputs, training, input_shape, configuration, epsilon=0.0, **kwargs):
        self.output_dimension = configuration[-1]
        self.epsilon = to_floatx(epsilon)
        super().__init__(inputs, training, input_shape, configuration, **kwargs)

    def compute(self, inputs, training=None):
        if training is None:
            training = self.training
        outputs = super().compute(inputs, training)

        if len(self.input_shape) >= 2:
            outputs = tf.reshape(outputs, (-1, self.output_dimension))
        outputs = tf.map_fn(tf.diag, outputs, name='map_create_diagonal_cov')
        if len(self.input_shape) >= 2:
            outputs = tf.reshape(outputs, (-1, self.input_shape[0], self.output_dimension, self.output_dimension))
        return outputs

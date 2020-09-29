from vaele_config import tf_floatx
import numpy as np
import tensorflow as tf
import gpflow
from magic_numbers import DROPOUT, REC_DROPOUT, BETA, ALPHA, BIDIRECTIONAL

_POSITIVE_TRANSFORMATION = 'softplus'

class DropTimeSteps(tf.keras.layers.Layer):
    def __init__(self, rate, dropped_scale=tf.convert_to_tensor(1, dtype=tf_floatx())):
        super(DropTimeSteps, self).__init__()
        self.rate = rate
        self.dropped_scale = dropped_scale

    def call(self, inputs, training=None):
        means, scales = inputs
        if training:
            mask = tf.nn.dropout(means[..., :1], rate=self.rate) == 0.0
            mask_0 = tf.reshape(tf.concat([[True], tf.repeat(False, means.shape[-2] -1)], axis=0), (1, -1, 1))
            mask = tf.math.logical_or(mask_0, mask)
            mask = tf.repeat(mask, means.shape[-1], axis=-1)
            return (
                # tf.where(mask, tf.zeros_like(means), means),
                means,
                tf.where(mask, self.dropped_scale * tf.ones_like(scales), scales)
            )
        return inputs
#
#
class Decoder(tf.keras.Model):
    """Currently, the output scale is a constant and does not depend on inputs"""
    def __init__(self, output_dim):
        super(Decoder, self).__init__()

        self.mean_output_layer = tf.keras.layers.TimeDistributed(
            Mlp([128], output_dim)
        )
        self.scale_diag_output_layer = tf.keras.layers.TimeDistributed(
            Mlp([128], output_dim, activation='softplus')
        )

    def call(self, inputs, training=None, mask=None):
        return (
            self.mean_output_layer(inputs),
            self.scale_diag_output_layer(inputs)
        )


# class LinearDecoder(tf.keras.Model):
#     """Currently, the output scale is a constant and does not depend on inputs"""
#     def __init__(self, output_dim):
#         super(LinearDecoder, self).__init__()
#
#         self.mean_output_layer = tf.keras.layers.TimeDistributed(
#             tf.keras.layers.Dense(output_dim, name='mean_output')
#         )
#         # TODO: init
#         # self.output_scale = tf.Variable(-2.0, name='scale_output', dtype=tf_floatx())
#         self.scale_diag_output_layer = tf.keras.layers.TimeDistributed(
#             tf.keras.layers.Dense(output_dim, _POSITIVE_TRANSFORMATION, name='scale_output')
#         )
#
#     def call(self, inputs, training=None, mask=None):
#         return (
#             self.mean_output_layer(inputs),
#             self.scale_diag_output_layer(inputs)
#         )

class LinearMlp(tf.keras.Model):
    def __init__(self, units_list, output_dim, activation=None, dropout=DROPOUT):
        super(LinearMlp, self).__init__()
        self.output_dim = output_dim
        self.dense_layers = []
        for i, units in enumerate(units_list):
            self.dense_layers.append(tf.keras.layers.Dense(units, None, name=f'hidden_layer_{i}'))
        self.output_layer = tf.keras.layers.Dense(output_dim, activation=activation)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        return self.output_layer(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


class Mlp(tf.keras.Model):
    def __init__(self, units_list, output_dim, activation=None, dropout=DROPOUT):
        super(Mlp, self).__init__()
        self.output_dim = output_dim
        self.dense_layers = []
        for i, units in enumerate(units_list):
            self.dense_layers.append(tf.keras.layers.Dense(units, 'relu', name=f'hidden_layer_{i}'))
        self.output_layer = tf.keras.layers.Dense(output_dim, activation=activation)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
        return self.output_layer(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


# class VaeleGRU(tf.keras.layers.GRU):
#     """GRU with trainable initial state"""
#     def __init__(self, *args, **kwargs):
#         super(VaeleGRU, self).__init__(*args, **kwargs)
#
#     def build(self, input_shape):
#         super(VaeleGRU, self).build(input_shape)
#         self.initial_state = self.add_weight(
#             name="vaele_initial_state",
#             shape=(self.units,),
#             initializer='glorot_uniform',
#             trainable=True
#         )
#
#     def _process_inputs(self, inputs, initial_state, constants):
#         proc_inputs, proc_initial_state, proc_constants = super(VaeleGRU, self)._process_inputs(
#             inputs, initial_state, constants
#         )
#         # Check if the proc_initial_state is all zeros
#         # Remember that the states are returned as a list (take into account that GRU only has 1 state,
#         # but LSTM has 2)
#         # The following code is suggested by the implementation in tf.keras.layers.recurrent.Rnn
#         proc_initial_state = [tf.cond(
#             tf.math.equal(tf.math.count_nonzero(proc_initial_state[0]), 0),
#             lambda: proc_initial_state[0],
#             lambda: proc_initial_state[0] + self.initial_state
#         )]
#
#         return proc_inputs, proc_initial_state, proc_constants


# class RnnInitialStateEncoder(tf.keras.Model):
#     def __init__(self, units, output_dim):
#         super().__init__()
#         self.base = tf.keras.layers.LSTM(
#             units=units, go_backwards=True, return_sequences=False, return_state=False, stateful=False
#         )
#         with tf.name_scope("output"):
#             self.mean_output_layer = Mlp([512], output_dim)
#             self.scale_output_layer = Mlp([512], output_dim, activation=_POSITIVE_TRANSFORMATION)
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.base(inputs, training=training, mask=mask)
#         x_mean, x_scale = tf.split(x, num_or_size_splits=2, axis=-1)
#         return self.mean_output_layer(x_mean), self.scale_output_layer(x_scale)

# class RnnInitialStateEncoder(tf.keras.Model):
#     def __init__(self, units, output_dim):
#         super().__init__()
#         self.conv_layers = []
#         for unit in units:
#             self.conv_layers.append(
#                 tf.keras.layers.Conv1D(unit, 3, activation='relu')
#             )
#             self.conv_layers.append(
#                 tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
#             )
#         self.flatten = tf.keras.layers.Flatten()
#         self.mlp = Mlp([256], output_dim, None)
#
#     def call(self, inputs, training=None):
#         x = inputs
#         for layer in self.conv_layers:
#             x = layer(x, training=training)
#         x = self.flatten(x)
#         return self.mlp(x)
#
#     @staticmethod
#     def automatic_units_selection(input_time_length, minimum_length=4):
#         # For each unit, the input length T is reduced as (T - 2) / 2
#         # Hence Length[n] = 2 ^ (-n) (T + 2) - 2
#         n_layers = -np.log2((minimum_length + 2) / (input_time_length + 2))
#         n_layers = int(np.floor(n_layers))
#         if n_layers < 1:
#             raise ValueError('n_layers < 1: check the input arguments')
#         elif n_layers == 1:
#             return [128]
#         elif n_layers == 2:
#             return [64, 128]
#         else:
#             return [32, 64] + [128] * (n_layers - 2)
#
# class InitialStateEncoder(tf.keras.Model):
#     def __init__(self, units, output_dim):
#         super().__init__()
#         self.conv_layers = []
#         for unit in units:
#             self.conv_layers.append(
#                 tf.keras.layers.Conv1D(unit, 3, activation='relu')
#             )
#             self.conv_layers.append(
#                 tf.keras.layers.MaxPool1D(pool_size=2, strides=2)
#             )
#         self.flatten = tf.keras.layers.Flatten()
#         self.mean_mlp = Mlp([256], output_dim, None)
#         self.scale_mlp = Mlp([256], output_dim, activation=_POSITIVE_TRANSFORMATION)
#
#     def call(self, inputs, training=None):
#         x = inputs
#         for layer in self.conv_layers:
#             x = layer(x, training=training)
#         x = self.flatten(x)
#         x_mean, x_scale = tf.split(x, num_or_size_splits=2, axis=-1)
#         return self.mean_mlp(x_mean), self.scale_mlp(x_mean)

    @staticmethod
    def automatic_units_selection(input_time_length, minimum_length=4):
        # For each unit, the input length T is reduced as (T - 2) / 2
        # Hence Length[n] = 2 ^ (-n) (T + 2) - 2
        n_layers = -np.log2((minimum_length + 2) / (input_time_length + 2))
        n_layers = int(np.floor(n_layers))
        if n_layers < 1:
            raise ValueError('n_layers < 1: check the input arguments')
        elif n_layers == 1:
            return [128]
        elif n_layers == 2:
            return [64, 128]
        else:
            return [32, 64] + [128] * (n_layers - 2)


# class RnnEncoder(tf.keras.Model):
#     def __init__(self, units_list, output_dim, bidirectional=BIDIRECTIONAL, tie_scale=False):
#         super().__init__()
#
#         self.tie_scale = tie_scale
#         self.kernel_size = 0
#         self.dilation_rate = 0
#         self.output_dim = output_dim
#
#         self.forward_init_state = tf.keras.layers.GRU(
#                 units=64, return_sequences=False, return_state=False, stateful=False,
#                 go_backwards=True
#         )
#         self.backward_init_state = tf.keras.layers.GRU(
#             units=64, return_sequences=False, return_state=False, stateful=False
#         )
#         self.adapters = []
#         for units in units_list:
#             self.adapters.append(
#                 (Mlp([2 * units], units, 'tanh'), Mlp([2 * units], units, 'tanh'))
#             )
#         self.stacked_grus = StackedGru(units_list, None, bidirectional)
#
#         with tf.name_scope("output"):
#             self.mean_output_layer = tf.keras.layers.TimeDistributed(
#                 Mlp([2 * units_list[-1]], output_dim, dropout=0.0)
#             )
#             # Note the dependency of the output_dim on tie_scale
#             self.scale_output_layer = tf.keras.layers.TimeDistributed(
#                 Mlp([2 * units_list[-1]], 1 if tie_scale else output_dim,
#                     activation=_POSITIVE_TRANSFORMATION, dropout=0.0)
#             )
#
#     def call(self, inputs, training=None, mask=None, initial_state=None):
#         forward_init_state = self.forward_init_state(inputs)
#         backward_init_state = self.backward_init_state(inputs)
#         initial_state = []
#         for fmlp, bmlp in self.adapters:
#             initial_state.append(
#                 (fmlp(forward_init_state), bmlp(backward_init_state))
#             )
#
#         x = inputs
#         x, state = self.stacked_grus(x, training=training, mask=mask, initial_state=initial_state)
#         x_mean, x_scale = tf.split(x, num_or_size_splits=2, axis=-1)
#         mean = self.mean_output_layer(x_mean)
#         scale = self.scale_output_layer(x_scale)
#         if self.tie_scale:
#             scale = tf.repeat(scale, self.output_dim, -1)
#         return mean, scale, state


class DynEmbedding(tf.keras.Model):
    def __init__(self, units_list, output_dim, bidirectional=BIDIRECTIONAL, tie_scale=False):
        super().__init__()
        # TODO
        conv_units_list = [32]  # [32, 64]
        self.output_dim = output_dim
        self.conv_layers = []
        for i, units in enumerate(conv_units_list):
            activation = None if i == (len(conv_units_list) - 1) else 'relu'
            self.conv_layers.append(
                (tf.keras.layers.Conv1D(units, 5, activation=activation, padding='same',
                                        dilation_rate=2 ** i),
                 tf.keras.layers.BatchNormalization(axis=-1))
            )
        # TODO: none is not used
        self.stacked_grus = StackedGru(units_list, None, bidirectional)
        with tf.name_scope("output"):
            self.output_layer = tf.keras.layers.TimeDistributed(
                Mlp([2 * units_list[-1]], output_dim, dropout=0.0)
            )

    def call(self, inputs, training=None, mask=None, initial_state=None):
        x = inputs
        for conv_layer, bn_layer in self.conv_layers:
            x = conv_layer(x)
            x = bn_layer(x, training=training)
        x, state = self.stacked_grus(x, training=training, mask=mask, initial_state=initial_state)
        return self.output_layer(x), state


class RnnEncoder(tf.keras.Model):
    def __init__(self, units_list, output_dim, bidirectional=BIDIRECTIONAL, use_scale_network=False,
                 tie_scale=False):
        super().__init__()
        # TODO
        conv_units_list = [32]  # [32, 64]
        self.tie_scale = tie_scale
        self.use_scale_network = use_scale_network
        self.kernel_size = 0
        self.dilation_rate = 0
        self.output_dim = output_dim

        # TODO: tie scale not used
        embedding_dim = output_dim
        self.embedding = DynEmbedding(units_list, embedding_dim, bidirectional, None)

        with tf.name_scope("output"):
            self.mean_output_layer = tf.keras.layers.TimeDistributed(
                Mlp([128], output_dim, dropout=0.0)
            )
            if self.use_scale_network:
                # Note the dependency of the output_dim on tie_scale
                self.scale_output_layer = tf.keras.layers.TimeDistributed(
                    Mlp([128], 1 if tie_scale else output_dim, dropout=0.0, activation='softplus')
                )
            else:
                transform = gpflow.utilities.positive()
                self.scale = gpflow.Parameter(0.1 * tf.ones(1 if tie_scale else output_dim, dtype=tf_floatx()),
                                              transform=transform, name='encoder_scale')

    def call(self, inputs, training=None, mask=None, initial_state=None):
        x, state = self.embedding(inputs, training, mask, initial_state)
        # We transform the low dimensional embedding to get a mean
        mean = self.mean_output_layer(x)
        if self.use_scale_network:
            scale = self.scale_output_layer(x)
            if self.tie_scale:
                scale = tf.repeat(scale, self.output_dim, -1)
        else:
            if self.tie_scale:
                scale = tf.repeat(self.scale, self.output_dim, -1)
            else:
                scale = self.scale
            scale = tf.zeros_like(tf.stop_gradient(mean)) + scale + 1e-5
        return mean, scale, state


class Encoder(tf.keras.Model):
    def __init__(self, units_list, kernel_size, output_dim, dilation_rate=1):
        super().__init__()

        assert kernel_size % 2 != 0, "Encoder: Use odd kernel_sizes!"
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate

        self.conv_layers = []
        for i, units in enumerate(units_list):
            # Last layer has a linear activation
            if i == len(units_list) - 1:
                padding = 'valid'
                activation = None
            else:
                activation = tf.keras.layers.LeakyReLU()
                padding = 'same'
            self.conv_layers.append([
               tf.keras.layers.Conv1D(units, kernel_size, padding=padding, dilation_rate=dilation_rate, activation=None),
               tf.keras.layers.BatchNormalization(momentum=0.8),
               activation
               ])

        with tf.name_scope("output"):
            # TODO
            # self.dropout = tf.keras.layers.TimeDistributed(tf.keras.layers.Dropout(DROPOUT))
            self.mean_output_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=output_dim, activation=None, name='mean_output',
                                      # kernel_regularizer = tf.keras.regularizers.l2(0.1)
                                      )
            )
            # self.bn_output = tf.keras.layers.BatchNormalization(momentum=0.8)
            self.scale_output_layer = tf.keras.layers.TimeDistributed(
                tf.keras.layers.Dense(units=output_dim, activation=_POSITIVE_TRANSFORMATION, name='scale_output')
            )

    def call(self, inputs, training=None, mask=None, initial_state=None):
        if initial_state is None:
            initial_state = [None, None]
        x = inputs
        for conv_layer, bn, activation in self.conv_layers:
            x = bn(conv_layer(x), training=training)
            if activation is not None:
                x = activation(x)

        # x = self.dropout(x, training=training)
        mean, scale = tf.split(x, num_or_size_splits=2, axis=-1)
        # mean = self.dropout(mean, training=training)
        # scale = self.dropout(scale, training=training)
        mean = self.mean_output_layer(mean)
        # mean = self.bn_output(mean, training=True)
        # mean = tf.keras.activations.tanh(mean)
        scale = 0 * self.scale_output_layer(scale) + 1e-4

        return mean, scale, [None, None]

    def compare_bn(self, inputs, training, initial_state=None):
        if initial_state is None:
            initial_state = [None, None]
        x = inputs
        for conv_layer, bn, activation in self.conv_layers:
            x = bn(conv_layer(x), training=training)
            if activation is not None:
                x = activation(x)
        return x


class StackedGru(tf.keras.Model):
    """
    A Encoder/Decoder Net prepared to yield Mean and Scale parameters as expected by the
    tfp.distributions.MultivariateNormalDiag class
    """
    # TODO: output dim
    def __init__(self, units_list, output_dim, bidirectional):
        super().__init__()
        self.bidirectional = bidirectional
        self.units_list = units_list
        self.gru_layers = []
        last_layer_idx = len(units_list) - 1
        for i, units in enumerate(units_list):
            name = "hidden_rnn_" + str(i) if i != last_layer_idx else "output_rnn"
            base = tf.keras.layers.GRU(
                units=units, dropout=DROPOUT, recurrent_dropout=REC_DROPOUT,
                return_sequences=True, return_state=True, stateful=False
            )
            with tf.name_scope(name):
                if bidirectional:
                    self.gru_layers.append(tf.keras.layers.Bidirectional(base, merge_mode='concat'))
                else:
                    self.gru_layers.append(base)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        output_states = []
        if initial_state is None:
            initial_state = [None] * len(self.gru_layers)
        # elif self.bidirectional:
        #     # Ignore backward state since it is not useful (it was built using samples from the past and
        #     # not samples from the future)
        #     initial_state = [
        #         (forward_state, tf.zeros_like(backward_state, dtype=tf_floatx())) for
        #         (forward_state, backward_state) in initial_state
        #     ]
        x = inputs
        for i, gru_layer in enumerate(self.gru_layers):
            if self.bidirectional:
                x, state_forward, state_backward = gru_layer(x, mask=mask, training=training, initial_state=initial_state[i])
                output_states.append((state_forward, state_backward))
            else:
                x, state = gru_layer(x, mask=mask, training=training, initial_state=initial_state[i])
                output_states.append(state)
        return x, output_states


if __name__ == '__main__':
    import tensorflow as tf
    import numpy as np
    conv1D = tf.keras.layers.Conv1D
    x = tf.convert_to_tensor(np.random.rand(1, 10, 2))
    conv1D(3, 9)(x).shape
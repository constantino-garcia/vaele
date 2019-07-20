"""
Wrap tensorflow optimizers to clip gradients and force the learning rate to be a tf.Variable (which enables
scheduling when training)
"""
import numpy as np
import tensorflow as tf
from config import floatx, tf_floatx


class ClippedOptimizer(object):
    def __init__(self, optimizer, learning_rate=0.001, decay_steps=50000, decay_rate=0.98,
                 clip_norm=0.0, clip_avg_norm=0.0,
                 clip_global_norm=0.0, clip_value=0.0, **kwargs):
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        if isinstance(learning_rate, np.float):
            learning_rate = tf.Variable(learning_rate, dtype=tf_floatx(), name='learning_rate')
        self.lr = learning_rate
        if decay_steps > 0 and decay_rate > 0:
            learning_rate = tf.train.exponential_decay(self.lr, self.global_step,
                                                       self.decay_steps, self.decay_rate,
                                                       staircase=True)
        self.optimizer = optimizer(learning_rate, **kwargs)

        self.clip_norm = clip_norm
        self.clip_avg_norm = clip_avg_norm
        self.clip_global_norm = clip_global_norm
        self.clip_value = clip_value

    def clip_gradients(self, gradients_and_values):
        if self.clip_norm > 0.:
            gradients_and_values = [
                (tf.clip_by_norm(grad, self.clip_norm, name='clip_grad_norm'), var) for
                grad, var in gradients_and_values
            ]
        if self.clip_avg_norm > 0.:
            gradients_and_values = [
                (tf.clip_by_average_norm(grad, self.clip_avg_norm, name='clip_avg_grad_norm'), var) for
                grad, var in gradients_and_values
            ]
        if self.clip_global_norm > 0.:
            gradients_and_values = [
                (tf.clip_by_global_norm(grad, self.clip_global_norm, name='clip_global_grad_norm'), var) for
                grad, var in gradients_and_values
            ]
        if self.clip_value > 0.:
            gradients_and_values = [
                (tf.clip_by_value(grad, -self.clip_value, self.clip_value, name='clip_grad'), var) for
                grad, var in gradients_and_values
            ]
        return gradients_and_values

    def minimize(self, *args, **kwargs):
        return self.optimizer.minimize(*args, **kwargs)

    def compute_gradients(self, *args, **kwargs):
        return self.optimizer.compute_gradients(*args, **kwargs)

    def apply_gradients(self, grads_and_vars, name=None):
        return self.optimizer.apply_gradients(grads_and_vars,
                                              global_step=self.global_step, name=name)

    def minimize_with_clipping(self, loss, var_list=None, name=None):
        grads_and_vars = self.compute_gradients(loss, var_list=var_list)

        vars_with_grad = [v for g, v in grads_and_vars if g is not None]
        if not vars_with_grad:
          raise ValueError(
              "No gradients provided for any variable, check your graph for ops"
              " that do not support gradients, between variables %s and loss %s." %
              ([str(v) for _, v in grads_and_vars], loss))

        grads_and_vars = self.clip_gradients(grads_and_vars)

        return self.apply_gradients(grads_and_vars, name=name)


class Sgd(ClippedOptimizer):
    def __init__(self, **kwargs):
        if 'optimizer' in kwargs.keys():
            raise ValueError("'optimizer' is not a valid pararameter")
        super(Sgd, self).__init__(optimizer=tf.train.GradientDescentOptimizer, **kwargs)


class Momentum(ClippedOptimizer):
    def __init__(self, **kwargs):
        if 'optimizer' in kwargs.keys():
            raise ValueError("'optimizer' is not a valid pararameter")
        super(Momentum, self).__init__(optimizer=tf.train.MomentumOptimizer, **kwargs)


class Adam(ClippedOptimizer):
    def __init__(self, **kwargs):
        if 'optimizer' in kwargs.keys():
            raise ValueError("'optimizer' is not a valid pararameter")
        super(Adam, self).__init__(optimizer=tf.train.AdamOptimizer, **kwargs)


class Adagrad(ClippedOptimizer):
    def __init__(self, **kwargs):
        if 'optimizer' in kwargs.keys():
            raise ValueError("'optimizer' is not a valid pararameter")
        super(Adagrad, self).__init__(optimizer=tf.train.AdagradOptimizer, **kwargs)



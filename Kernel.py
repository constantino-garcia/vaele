import numpy as np
import tensorflow as tf
from config import get_session, floatx, tf_floatx
from utils.generic_utils import squared_distances
from utils.tf_utils import variable_summaries


class GaussianKernel(object):
    def __init__(self, x, y, length_scales, amplitude=1.0, epsilon=1e-3):
        # Set parameters
        self.amplitude = np.asarray(amplitude, dtype=floatx())
        self.length_scales = tf.Variable(
            np.array(length_scales, dtype=floatx()),
            name='length_scales'
        )
        with tf.name_scope('length_scales_matrix'):
            self.length_scales_matrix = tf.diag(self.length_scales ** 2)
        with tf.name_scope('inverse_length_scales_matrix'):
            self.inverse_length_scales_matrix = tf.diag(1. / (self.length_scales ** 2))

        self.epsilon = np.asarray(epsilon, dtype=floatx())
        # Set, for consistency with other objects, the optimizable parameters in params
        self.params = [self.length_scales]

        # Keep track of the inputs
        self.x = x
        self.y = y

        with tf.name_scope('kernel_matrix'):
            self.output = self.compute(self.x, self.y)
        # Variance is the diagonal of the output in those cases where x == y
        # we also save it as a numerical value for initialization purposes
        self.max_amplitude = amplitude + epsilon
        if self.x == self.y:
            with tf.name_scope('kernel_variance'):
                self.variances = tf.diag_part(self.output)
        else:
            self.variances = None

    def compute(self, x, y=None):
        # Compute the output
        distances = squared_distances(x, y, self.length_scales)
        output = self.amplitude * tf.exp(-distances)
        # Add a small value to the diagonal for improving the stability if x == y
        if x == y and self.epsilon > 0:
            output = output + self.epsilon * tf.eye(int(output.get_shape()[0]), dtype=tf_floatx())
        return output

    def compute_and_get_grad_x(self, x, y=None):
        """
        Gradient of k(x,y) with respect to the first parameter
        :param x: A vector. Independent variable with respect which we derivate.
        :param y: A vector or matrix. Second argument to the kernel function.
        :return: the gradient of k(x,y) wrt x as a matrix with shape (y.shape[0], dimension)
        """
        with tf.name_scope('kernel_grad'):
            if x.get_shape().ndims == 1:
                x_ = tf.reshape(x, (1, -1))
            else:
                x_ = x
            output = self.compute(y, x_)
            grad = output * (y - x_) / (self.length_scales ** 2)
            return tf.transpose(output), grad

    def load_params(self, params):
        sess = get_session()
        for i, value in enumerate(params):
            assing_op = self.params[i].assign(value)
            sess.run(assing_op)

    def get_configuration(self):
        sess = get_session()
        return {'amplitude': self.amplitude, 'length_scales': sess.run(self.length_scales), 'epsilon': self.epsilon}


if __name__ == '__main__':
    from Kernel import *
    dim = 3
    x = tf.placeholder(floatx(), [None, None])
    x_flat = tf.placeholder(floatx(), [None])
    y = tf.placeholder(floatx(), [None, None])
    k = GaussianKernel(x, y, length_scales=np.ones(dim).astype(floatx()))
    xv = np.random.random((3, dim))
    yv = np.random.random((4, dim))

    sess = get_session()
    res = sess.run(squared_distances(k.x, k.y, k.params[0]), {x: xv, y: yv}),
    sess.run(k.grad_x(x_flat, y), {x_flat: np.random.random((dim, )), y: yv})



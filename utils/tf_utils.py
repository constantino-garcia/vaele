import pickle
from config import get_session, tf_floatx, floatx
import numpy as np
import tensorflow as tf

def to_floatx(x):
    return np.float64(x).astype(floatx())


def tf_to_floatx(x):
    if floatx() == 'float32':
        return tf.to_float(x)
    elif floatx() == 'float64':
        return tf.to_double(x)
    else:
        raise ValueError("Unexpected 'floatx()' value")


def save_shared_parameters(params, filename):
    f = open(filename, 'wb')
    pickle.dump([tf_get_value(p) for p in params], f)


def tf_get_value(variables):
    sess = get_session()
    if isinstance(variables, list):
        # Assume a list of variables
        values = [var.eval(session=sess) for var in variables]
    else:
        values = variables.eval(session=sess)
    return values


def tf_set_value(variables, values, sess=None):
    if sess is None:
        sess = get_session()
    if isinstance(variables, list):
        assert isinstance(values, list), "If 'variables' is a list, 'values' should also be a list"
        assert len(values) == len(variables), 'Invalid lengths'
        assign_ops = []
        for i, value in enumerate(values):
            assign_ops.append(variables[i].assign(value))
        sess.run(assign_ops)
    else: # assume a single variable
        # assert isinstance(variables, tf.Variable), (
        #      "'variables' should be either a 'tf.Variable' or a list of 'tf.Variables'"
        # )
        assign_op = variables.assign(values)
        sess.run(assign_op)
    return sess


# def tf_jacobian(y, x, n):
#     loop_vars = [
#         tf.constant(0, tf.int32),
#         tf.TensorArray(tf_floatx(), size=n),
#     ]
#     _, jacobian = tf.while_loop(
#         lambda i, _: i < n,
#         lambda i, result: (i + 1, result.write(i, tf.gradients(y[i], x)[0])),
#         loop_vars
#     )
#     return jacobian.stack()


def tf_jacobian(y, x, n):
    return tf.stack([tf.gradients(y[i], x)[0] for i in range(n)])


def vector_to_tril(vectors, dimension):
    """
    Transforms the input vectors with shape N x (dimension * (dimension + 1)) // 2 into N lower triangular
    matrices N x dimension x dimension. This function does not check that the input vectors have a valid shape.
    :param vectors: N x (dimension * (dimension + 1)) // 2 matrix representing N vectors.
    :param dimension: The dimension of the resulting matrices.
    :return: A tensor N x dimension x dimension representing N matrices of dimension x dimension
    """
    indices = np.array(list(zip(*np.tril_indices(dimension))), dtype=np.int32)
    return tf.map_fn(
        lambda vector: tf.scatter_nd(indices=indices, updates=vector, shape=[dimension, dimension]),
        vectors
    )


def variable_summaries(var):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization). Defined in
    https://www.tensorflow.org/get_started/summaries_and_tensorboard
    """
    with tf.name_scope('summaries'):
        # mean = tf.reduce_mean(var)
        # tf.summary.scalar('mean', mean)
        # with tf.name_scope('stddev'):
        #     stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        # tf.summary.scalar('stddev', stddev)
        # tf.summary.scalar('max', tf.reduce_max(var))
        # tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

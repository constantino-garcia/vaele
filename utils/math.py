from vaele_config import tf_floatx
import tensorflow as tf
import numpy as np


def inverse_lower_triangular(M):
    if M.shape.ndims != 3:
        raise ValueError("Number of dimensions for input is required to be 3.")
    I = tf.eye(M.shape[1], batch_shape=[M.shape[0]], dtype=tf_floatx())
    return tf.linalg.triangular_solve(M, I)


def multiply_gaussians(means_a, precisions_a, means_b, precisions_b):
    """
    :param means_a: N different means arranged as an array of N X D
    :param precision_a: N different precision matrices N x D x D
    :param means_b: M different means arranged as an array of M x D. N and M should be consistent with broadcast
    operations
    :param precision_b: M different precision matrices arranged as M x D x D
    :return: mus (N x D) and covs and precs (N x D x D)
    """
    precisions = precisions_a + precisions_b
    covariances = tf.map_fn(tf.linalg.inv, precisions)
    # covariances = (covariances + tf.map_fn(tf.transpose, covariances)) / 2.0

    mus = tf.einsum(
        'aij,aj->ai',
        covariances,
        tf.einsum('aij,aj->ai', precisions_a, means_a) + tf.einsum('aij,aj->ai', precisions_b, means_b)
    )
    return mus, covariances, precisions


def build_delay_space(ys, dim, time_lag):
    N = ys.shape[-2]
    jumpsvect = np.arange(0, dim * time_lag, time_lag)
    numelem = N - (dim - 1) * time_lag
    takens = np.zeros((ys.shape[0], numelem, dim))
    for y_it, y in enumerate(ys):
        for t in range(numelem):
            takens[y_it, t, :] = y.numpy()[t + jumpsvect, 0]
    return tf.convert_to_tensor(takens[..., -1:]), tf.convert_to_tensor(takens)


def build_centered_delay_embedding(ys, dim, time_lag):
    """
    :param ys:
    :param dim: An odd number indicating the embedding dimension. If dim is not an odd number,
    the next odd number is used as dimensions. This guarantees that the central point is well defined.
    :param time_lag:
    :return:
    """
    # Make sure dimension is odd
    if dim % 2 == 0:
        dim += 1
    half_dim = (dim - 1) // 2
    N = ys.shape[-2]
    jumpsvect = np.arange(-half_dim * time_lag, half_dim * time_lag + 1, time_lag)
    first_index = half_dim * time_lag
    last_index = N - half_dim * time_lag
    takens = np.zeros((ys.shape[0], last_index - first_index, dim))
    for y_it, y in enumerate(ys):
        y = y.numpy()
        for takens_index, t in enumerate(range(first_index, last_index)):
            takens[y_it, takens_index, :] = y[t + jumpsvect, 0]
    return tf.convert_to_tensor(takens), first_index, last_index


@tf.function
def tf_pairwise_distance(feature, squared: bool = False):
    """Computes the pairwise distance matrix with numerical stability.
    output[i, j] = || feature[i, :] - feature[j, :] ||_2
    Args:
      feature: 2-D Tensor of size [number of data, feature dimension].
      squared: Boolean, whether or not to square the pairwise distances.
    Returns:
      pairwise_distances: 2-D Tensor of size [number of data, number of data].
    """
    pairwise_distances_squared = tf.math.add(
        tf.math.reduce_sum(tf.math.square(feature), axis=[1], keepdims=True),
        tf.math.reduce_sum(
            tf.math.square(tf.transpose(feature)), axis=[0], keepdims=True
        ),
    ) - 2.0 * tf.matmul(feature, tf.transpose(feature))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = tf.math.maximum(pairwise_distances_squared, 0.0)
    # Get the mask where the zero distances are at.
    error_mask = tf.math.less_equal(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = tf.math.sqrt(
            pairwise_distances_squared
            + tf.cast(error_mask, dtype=tf.dtypes.float32) * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = tf.math.multiply(
        pairwise_distances,
        tf.cast(tf.math.logical_not(error_mask), dtype=tf_floatx()),
    )

    num_data = tf.shape(feature)[0]
    # Explicitly set diagonals to zero.
    mask_offdiagonals = tf.ones_like(pairwise_distances) - tf.linalg.diag(
        tf.ones([num_data], dtype=tf_floatx())
    )
    pairwise_distances = tf.math.multiply(pairwise_distances, mask_offdiagonals)
    return pairwise_distances
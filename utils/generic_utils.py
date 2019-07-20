import pickle
import numpy as np
from config import floatx, tf_floatx
from utils.tf_utils import tf_to_floatx
import scipy.spatial as spatial
import tensorflow as tf
from sklearn.cluster import KMeans


# def create_symm_matrix(vec, output_dimension):
#     rows = []
#     prev = 0
#     for i in range(output_dimension):
#         rows.append(T.concatenate([T.zeros((i, )), vec[prev:(prev + output_dimension - i)]]))
#         prev = prev + output_dimension - i
#     #rows.append(T.zeros((output_dimension, )))
#     mat = T.stack(rows)
#     mat = mat - T.nlinalg.alloc_diag(T.nlinalg.diag(mat)) + mat.T
#     return mat
#
# def flatten_symm_matrix(mat, output_dimension):
#     return mat[np.triu_indices(output_dimension)]


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def normal_gamma_kullback_leibler(alpha_0, beta_0, mu_0, cov_0, alpha_1, beta_1, mu_1, cov_1, dimension, epsilon=1e-6):
    mean_diff = tf.reshape((mu_0 - mu_1), (-1, 1))
    epsilon = tf_to_floatx(epsilon)
    # avoid numerical problems by adding a small quantity to the matrix diagonal
    with tf.name_scope("inverses"):
        inv_cov_1 = tf.matrix_inverse(cov_1)
    with tf.name_scope("matrix_agreeement"):
        matrix_agreement = tf.matmul(inv_cov_1, cov_0)
        matrix_agreement = tf.Print(matrix_agreement, [matrix_agreement], "matrix_agreement", summarize=9)
    cov_0_p = cov_0 + tf.eye(dimension, dtype=floatx()) * epsilon
    cov_1_p = cov_1 + tf.eye(dimension, dtype=floatx()) * epsilon
    cov_0_p = tf.Print(cov_0_p, [cov_0_p], "cov_0_p", summarize=9)
    cov_1_p = tf.Print(cov_1_p, [cov_1_p], "cov_1_p", summarize=9)
    return (
            0.5 * tf.reshape((alpha_0 / beta_0 * tf.matmul(tf.transpose(mean_diff), tf.matmul(inv_cov_1, mean_diff))),
                             []) +
            0.5 * tf.trace(matrix_agreement) +
            -0.5 * tf.linalg.logdet(cov_0_p) + 0.5 * tf.linalg.logdet(cov_1_p) +
            -dimension / 2. +
            alpha_1 * tf.log(beta_0 / beta_1) +
            (alpha_0 - alpha_1) * tf.digamma(alpha_0) +
            alpha_0 * (beta_1 / beta_0 - 1.) +
            -tf.lgamma(alpha_0) + tf.lgamma(alpha_1)
    )


def select_inducing_points_3d(x, nb_pseudo_inputs):
    def foo(x, n):
        m = np.min(x)
        M = np.max(x)
        f0 = 0.9
        f1 = 1.1
        if m < 0:
            m = m * f1
        else:
            m = m * f0
        if M < 0:
            M = M * f0
        else:
            M = M * f1
        return (M - m) / (n - 1)

    n = int(x.shape[1] ** (1 / 3)) + 1
    s = np.mean(x, axis=0)
    s1 = foo(x[:, 0], n)
    s2 = foo(x[:, 1], n)
    s3 = foo(x[:, 2], n)
    np.array([(s[0] + x * s1, s[1] + y * s2, s[2] + z * s3) for x in range(n) for y in range(n) for z in range(n)])


def kmeans_based_inducing_points(data, nb_pseudo_inputs, scale_range=0):
    if scale_range != 0:
        means = np.mean(data, axis=0)
        contracted = means + (data - means) * (1 - scale_range)
        expanded = means + (data - means) * (1 + scale_range)
        use_data = np.concatenate([data, contracted, expanded], axis=0)
    else:
        use_data = data
    km = KMeans(n_clusters=nb_pseudo_inputs, init='random').fit(
        use_data
    )
    return km.cluster_centers_


def expand_range(lower_bound, upper_bound, expansion_factor):
    if lower_bound > upper_bound:
        raise ValueError("Lower bound > upper bound")

    if lower_bound < 0 and upper_bound < 0:
        lower_bound *= expansion_factor
        upper_bound /= expansion_factor
    elif lower_bound < 0 and upper_bound > 0:
        lower_bound *= expansion_factor
        upper_bound *= expansion_factor
    elif lower_bound > 0 and upper_bound > 0:
        lower_bound /= expansion_factor
        upper_bound *= expansion_factor
    else:
        raise ValueError("Lower bound > upper bound")
    return lower_bound, upper_bound


def generate_uniform_grid(ranges, points_per_dimension):
    grid = np.meshgrid(*[np.linspace(xmin, xmax, points_per_dimension) for xmin, xmax in ranges])
    return np.concatenate([grid_item.reshape((-1, 1)) for grid_item in grid], axis=-1)


def grid_based_inducing_points(data, nb_pseudo_inputs, expansion_factor=1.25, fixed_range=None):
    dim = data.shape[1]
    grid_size = np.floor(nb_pseudo_inputs ** (1 / dim))
    if grid_size < 2:
        raise ValueError("Grid size < 2")
    if fixed_range is None:
        min_values = data.min(axis=0)
        max_values = data.max(axis=0)
    else:
        min_values = np.repeat(fixed_range[0], data.shape[1])
        max_values = np.repeat(fixed_range[1], data.shape[1])

    ranges = [expand_range(min_value, max_value, expansion_factor) for min_value, max_value in zip(min_values, max_values)]
    pseudo_inputs = generate_uniform_grid(ranges, grid_size)

    if nb_pseudo_inputs > pseudo_inputs.shape[0]:
        additional_points = kmeans_based_inducing_points(data, nb_pseudo_inputs - pseudo_inputs.shape[0])
        pseudo_inputs = np.concatenate([pseudo_inputs, additional_points], axis=0)

    return pseudo_inputs


def get_initial_pseudo_inputs_random_space(nb_pseudo_inputs, low, high):
    assert len(low) == len(high), "len(low) != len(high)"
    dim = len(low)
    for i in range(dim):
        assert low[i] < high[i], "low[{0}] >= high[{0}]".format(i)
    X = np.random.uniform(low, high, (5 * 10 ** np.min((dim, 6)), dim))
    km = KMeans(n_clusters=nb_pseudo_inputs, init='random').fit(X)
    return km.cluster_centers_


def save_object(object, filename):
    f = open(filename, 'wb')
    pickle.dump(object, f)


def load_object(filename):
    f = open(filename, 'rb')
    return pickle.load(f)


# def squared_frobenious_distances(x, y=None):
#     """
#     :param x: tensor3 with shape (nb_observations x matrix_dim_1 x matrix_dim_2)
#     :param y: tensor3 with shape (nb_observations x matrix_dim_1 x matrix_dim_2)
#     :return: matrix of distances between each of the matrices in x and y
#     """
#     if y is None:
#         y = x
#     if x.ndim != 3 or y.ndim != 3:
#         raise ValueError('Expected a tensor3 as input')
#     translation_vectors = (x.dimshuffle(0, 1, 2, 'x') - y.dimshuffle('x', 1, 2, 0))
#     return abs(translation_vectors ** 2).sum(1, 2)


def squared_distances(x, y=None, length_scales=None, name='squared_distances'):
    """
    :param x: A TensorFlow matrix with shape (nb_observations, dimension)
    :param y: A TensorFlow matrix with shape (nb_observations, dimension)
    :param length_scales: A vector with shape (dimension, ) or a single double indicating the lengh_scales of each
    dimension
    :return: The Euclidean squared distances that result after normalizing each dimension by its length_scale. That
    is dist = sum( (x_i - y_i) ** 2 / (2 * length_scale_i **2) )
    """
    with tf.name_scope(name):
        if x.get_shape().ndims == 1:
            x_ = tf.reshape(x, (-1, 1))
        else:
            x_ = x
        if y is None:
            y_ = x
        elif y.get_shape().ndims == 1:
            y_ = tf.reshape(y, (-1, 1))
        else:
            y_ = y
        if length_scales is not None:
            x_ = x_ / (np.sqrt(2.) * tf.expand_dims(length_scales, 0))
            y_ = y_ / (np.sqrt(2.) * tf.expand_dims(length_scales, 0))
        translation_vectors = (tf.expand_dims(x_, 2) - tf.expand_dims(tf.transpose(y_), 0))
        return tf.reduce_sum(translation_vectors ** 2, axis=1)


def distances_to_probabilities(distances, theiler=0, epsilon=1e-9):
    # find distances = 0 to set the probability to zero
    ones = tf.ones_like(distances)
    zeros = tf.zeros_like(distances)
    mask = tf.where(tf.equal(distances, zeros), ones, zeros)
    if theiler > 0:
        theiler_window = tf.matrix_band_part(ones, theiler, theiler)
        mask = tf.clip_by_value(mask + theiler_window, 0, 1)

    # add small value for stability
    top = tf.exp(-distances) + epsilon
    top = top - mask * top
    bottom = tf.reduce_sum(top, axis=1)
    return top / bottom


def nca_cost(target, prediction, length_scales=None, theiler=0):
    # check that target and prediction are matrices
    distances = squared_distances(prediction, length_scales=length_scales)
    probabilities = distances_to_probabilities(distances, theiler)
    # regression errors
    target_distances = tf.sqrt(squared_distances(target))
    nb_examples = tf_to_floatx(tf.shape(target_distances))[0]
    # TODO: remove
    # nb_examples = tf.Print(nb_examples, [nb_examples], 'nb_example')
    # probabilities = tf.Print(probabilities, [probabilities], 'pr', summarize=9)
    # target_distances = tf.Print(target_distances, [target_distances], 'target', summarize=9)
    return tf.reduce_sum(probabilities * target_distances) / nb_examples


def nca_dx_cost(x1, x2, margin=0.2, length_scales=None):
    dx1 = tf.stop_gradient(x1[1:, ] - x1[:-1, :])
    dx2 = tf.stop_gradient(x2[1:, ] - x2[:-1, :])

    def get_ith_penalty(xi, dxi):
        distances = squared_distances(xi, x2[:-1, :], length_scales=length_scales)
        probabilities = distances_to_probabilities(distances)
        # probabilities = tf.Print(probabilities, [probabilities[:5]], 'pr: ')
        distances = tf.sqrt(squared_distances(dxi, dx2))
        return tf.reduce_sum(probabilities * distances) / tf_to_floatx(tf.shape(distances))[0]

        # Another implementation using margin
        # distances = squared_distances(xi, x2[:-1, :], length_scales)
        # dx_in_margin = tf.boolean_mask(dx2, tf.less(distances[0, :], margin))
        # return tf.reduce_sum(squared_distances(dxi, dx_in_margin, length_scales))

    penalties = tf.map_fn(
        lambda inputs: get_ith_penalty(tf.reshape(inputs[0], (1, -1)), tf.reshape(inputs[1], (1, -1))),
        elems=[x1[:-1, :], dx1],
        dtype=tf_floatx(),
        name='ith_crossing_penalty'
    )

    return tf.reduce_sum(penalties)

    # distances = squared_distances(x1[:-1, :], x2[:-1, :], length_scales=length_scales)
    # probabilities = distances_to_probabilities(distances, theiler)
    # # regression errors
    # target_distances = tf.sqrt(squared_distances(x1[1:, :] - x1[:-1, :], x2[1:, :] - x2[:-1, :]))
    # return tf.reduce_sum(probabilities * target_distances) / tf_to_floatx(tf.shape(target_distances))[0]


# def frobenious_nca_cost(target, prediction):
#     distances = frobenious_distance(prediction)
#     probabilities = distances_to_probabilities(distances)
#     # regression errors
#     target_distances = T.sqrt(frobenious_distance(target))
#     return T.sum(probabilities * target_distances) / target_distances.shape[0]


def create_delay_embedding(x, dimension, tau=1):
    nb_embeddings = len(x) - (dimension - 1) * tau
    auxiliar_indices = np.array(range(0, dimension * tau, tau)).astype(int)
    embedding = np.zeros((nb_embeddings, dimension))
    for i in range(nb_embeddings):
        embedding[i, :] = x[i + auxiliar_indices]
    return embedding


def estimate_embedding_cov(initial_embedding, target_embedding, initial_radius=0.1,
                           radius_increase_factor=np.sqrt(2), min_nb_neighbours=5, theiler=100):
    point_tree = spatial.cKDTree(initial_embedding)
    embedding_covs = []
    nb_points = len(initial_embedding)
    for i in range(nb_points):
        radius = initial_radius
        nb_neighbours = 0
        # repeat increasing the radius while there are not enough neighbours
        while nb_neighbours < min_nb_neighbours:
            neighbours = point_tree.query_ball_point(initial_embedding[i], radius, p=np.inf)
            neighbours = [j for j in neighbours if np.abs(j - i) > theiler]
            nb_neighbours = len(neighbours)
            radius *= radius_increase_factor
        embedding_covs.append(
            np.cov(np.transpose(target_embedding[neighbours])).reshape(
                (target_embedding.shape[1], target_embedding.shape[1]))
        )
    return embedding_covs


def vaele_scale(x, sampling_time):
    """ Scale dx so that the drift equation has unit order of magnitude"""
    scaling_factor = abs(np.diff(x, axis=0)).std(axis=0)
    return (x.copy() / scaling_factor) * sampling_time

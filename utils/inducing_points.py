import pickle
import numpy as np
from vaele_config import np_floatx, tf_floatx
import scipy.spatial as spatial
import tensorflow as tf
from sklearn.cluster import KMeans


def _generate_random_space(min_values, max_values, sim_points=None):
    assert len(min_values) == len(max_values), "len(low) != len(high)"
    dim = len(min_values)
    for i in range(dim):
        assert min_values[i] < max_values[i], "low[{0}] >= high[{0}]".format(i)
    sim_shape = (10 ** np.min((dim, 6)), dim) if sim_points is None else (sim_points, dim)
    return np.random.uniform(min_values, max_values, sim_shape)


def _expand_range(lower_bound, upper_bound, expansion_factor):
    if lower_bound < 0 and upper_bound < 0:
        lower_bound *= expansion_factor
        upper_bound /= expansion_factor
    elif lower_bound < 0 < upper_bound:
        lower_bound *= expansion_factor
        upper_bound *= expansion_factor
    elif lower_bound > 0 and upper_bound > 0:
        lower_bound /= expansion_factor
        upper_bound *= expansion_factor
    else:
        raise ValueError(f"Lower bound > upper bound ({lower_bound} > {upper_bound})")
    return lower_bound, upper_bound


def kmeans_based_inducing_points(data, nb_pseudo_inputs, scale_range=0):
    if scale_range != 0:
        means = np.mean(data, axis=0)
        contracted = means + (data - means) * (1 - scale_range)
        expanded = means + (data - means) * (1 + scale_range)
        use_data = np.concatenate([data, contracted, expanded], axis=0)
    else:
        use_data = data
    km = KMeans(n_clusters=nb_pseudo_inputs, init='random').fit(use_data)
    return km.cluster_centers_


def _generate_uniform_grid(ranges, points_per_dimension):
    grid = np.meshgrid(*[np.linspace(xmin, xmax, int(points_per_dimension)) for xmin, xmax in ranges])
    return np.concatenate([grid_item.reshape((-1, 1)) for grid_item in grid], axis=-1)


def grid_based_inducing_points(data_ranges, nb_pseudo_inputs, expansion_factor=1):
    """data_ranges should be a list of 2D arrays. The len of the list is the number of dimensions. The items on
    the arrays are the [min, max] for that dimension"""
    dimension = len(data_ranges)
    grid_size = np.floor(nb_pseudo_inputs ** (1 / dimension))
    min_values, max_values = zip(*data_ranges)
    if grid_size < 2:
        raise ValueError("Grid size < 2")
    ranges = [_expand_range(min_value, max_value, expansion_factor) for min_value, max_value in zip(min_values, max_values)]
    pseudo_inputs = _generate_uniform_grid(ranges, grid_size)

    if nb_pseudo_inputs > pseudo_inputs.shape[0]:
        X = _generate_random_space(min_values, max_values)
        distances = spatial.distance.cdist(X, pseudo_inputs)
        indices = np.argsort(np.min(distances, axis=1))
        additional_indices = indices[-(nb_pseudo_inputs - pseudo_inputs.shape[0]):]
        additional_points = X[additional_indices, :]
        pseudo_inputs = np.concatenate([pseudo_inputs, additional_points], axis=0)

    return pseudo_inputs




if __name__ == '__main__':
    pseudo_inputs = grid_based_inducing_points(
        data_ranges=[(-1, 1) for _ in range(2)],
        nb_pseudo_inputs=10,
        expansion_factor=1
    )
    import matplotlib.pyplot as plt
    plt.scatter(*pseudo_inputs.transpose())

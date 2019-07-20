import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from config import get_session
from utils.tf_utils import tf_get_value
plt.ion()

def plot_embedding(em, drift, experiment, add_ips=True):
    sess = get_session()
    embedding = sess.run(em.encoding_mean_network.output,
                         {em.encoding_mean_network.input: experiment.lag_embedded_y}
    )
    fig = plt.figure()
    if embedding.shape[1] == 3:
        ax = fig.add_subplot(1, 2, 1, projection='3d')
    else:
        ax = fig.add_subplot(1, 1, 1)
    ax.plot(*[embedding[:, i] for i in range(embedding.shape[1])])
    if add_ips:
        inducing_points = tf_get_value(em.inducing_points)
        ax.scatter(*[inducing_points[:, i] for i in range(embedding.shape[1])], zorder=1000,
                   c='red', s=50)
        ax.scatter(*[inducing_points[:, i] for i in range(embedding.shape[1])], zorder=1000,
                   c='green', s=50)

    dim = embedding.shape[1]

    X = embedding[np.arange(0, len(embedding), 500), :]
    predictions = drift([X])[0]
    if embedding.shape[1] == 3:
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        ax2.quiver(*X.T, *(1e2 * predictions))  # , units='width')
    else:
        X = [np.linspace(np.min(embedding[:, dim_it]), np.max(embedding[:, dim_it]), 7) for
             dim_it in range(embedding.shape[1])]
        X = np.meshgrid(*X)
        predictions = (
            drift(
                [np.concatenate([X[i].flatten().reshape((-1, 1)) for i in range(dim)], axis=1)]
            )[0]
        )
        ax.quiver(*(X), *predictions, color='green', units='width')
    plt.show()

def plot_2d_drifts(samples, drift_predictions, sampling_time=1):
    alp = 0.3
    window = 50

    dsamples = np.diff(samples, axis=0)  # / em.sampling_time

    X = np.linspace(np.min(samples[:, 0]), np.max(samples[:, 0]), 31)
    Y = np.linspace(np.min(samples[:, 1]), np.max(samples[:, 1]), 31)
    X, Y = np.meshgrid(X, Y)
    predictions = []
    for i in range(len(X)):
        predictions.append(
            drift_predictions(np.concatenate([
                X[i].reshape((-1, 1)),
                Y[i].reshape((-1, 1))
            ], 1)) * sampling_time
        )
    predictions = np.stack(predictions)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X, Y, predictions[:, 0, :], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.scatter(samples[:-1, 0], samples[:-1, 1], dsamples[:, 0], zorder=10000)

    fig2 = plt.figure()
    ax = fig2.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X, Y, predictions[:, 1, :], cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.scatter(samples[:-1, 0], samples[:-1, 1], dsamples[:, 1], zorder=10000)

def plot_drifts_by_dimension(samples, drift_predictions, sampling_time=1, add_scatter=False):
    alp = 0.3
    window = 50

    dsamples = np.diff(samples, axis=0)
    predictions = drift_predictions(samples)

    fig = plt.figure()
    for i in range(samples.shape[1]):
        for j in range(samples.shape[1]):
            ax = fig.add_subplot(samples.shape[1], samples.shape[1], i * samples.shape[1] + j + 1)
            if add_scatter:
                ax.scatter(samples[:-1, i], dsamples[:, j])
            ax.plot(samples[:, i], predictions[j], c='red', zorder=100)
            ax.set_title('f' + str(j) + '(x_' + str(i) + ')')


def scatter_diff_2d(samples, drift_predictions, drift_index, sampling_time=1, ax=None):
    dsamples = np.diff(samples, axis=0)

    predictions = drift_predictions(samples)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.plot(samples[:, 0], samples[:, 1], predictions[drift_index, :])
    ax.scatter(samples[:-1, 0], samples[:-1, 1], dsamples[:, drift_index], zorder=10000, c='red')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Drift_' + str(drift_index))
    return ax


def plot_2d_drift(y, dy, drifts, fig=None):
    if fig is None:
        fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.plot(y[:, 0], dy[:, 0], color="blue")
    ax.plot(y[:, 0], drifts[0], color="red")
    ax = fig.add_subplot(222)
    ax.plot(y[:, 0], dy[:, 1], color="blue")
    ax.plot(y[:, 0], drifts[1], color="red")
    ax = fig.add_subplot(223)
    ax.plot(y[:, 1], dy[:, 0], color="blue")
    ax.plot(y[:, 1], drifts[0], color="red")
    ax = fig.add_subplot(224)
    ax.plot(y[:, 1], dy[:, 1], color="blue")
    ax.plot(y[:, 1], drifts[1], color="red")
    return fig
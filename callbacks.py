import numpy as np
import csv
from config import get_session
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from utils.tf_utils import tf_floatx, to_floatx, tf_set_value, tf_get_value
from utils.generic_utils import expand_range
import os
from matplotlib.patches import Ellipse
from nnets.optimizers import ClippedOptimizer
from collections import OrderedDict
import pickle
import tensorflow as tf
from VAE import VAE


class Callback(object):
    def __init__(self, monitor=None):
        self.monitor = monitor

    def set_monitor(self, monitor):
        self.monitor = monitor

    def _reset(self):
        pass

    def on_epoch_begin(self):
        pass

    def on_epoch_end(self):
        pass

    def on_batch_begin(self):
        pass

    def on_batch_end(self):
        pass

    def on_train_begin(self):
        pass

    def on_train_end(self):
        pass


class CallbackList(object):
    def __init__(self, callbacks=[]):
        self.callbacks = [c for c in callbacks]

    def len(self):
        return len(self.callbacks)

    def append(self, callback):
        self.callbacks.append(callback)

    def set_monitor(self, monitor):
        for callback in self.callbacks:
            callback.set_monitor(monitor)

    def reset(self):
        for callback in self.callbacks:
            callback.reset()

    def on_epoch_begin(self):
        for callback in self.callbacks:
            callback.on_epoch_begin()

    def on_epoch_end(self):
        for callback in self.callbacks:
            callback.on_epoch_end()

    def on_batch_begin(self):
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self):
        for callback in self.callbacks:
            callback.on_batch_end()

    def on_train_begin(self):
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self):
        for callback in self.callbacks:
            callback.on_train_end()


#
# # id: useful to distinguish the nnet lr vs the sde params lr
# class DecreaseLearningRate(Callback):
#     def __init__(self, factor=np.sqrt(0.5), patience=3, initial_patience=1,
#                  min_lr=1e-7, verbose=False):
#         super().__init__()
#         # Set with set_monitor
#         self.optimizer = None
#         if factor < 0 or factor > 1:
#             raise ValueError('The lr factor should be 0 < factor < 1')
#         self.factor = factor
#         self.initial_patience = initial_patience
#         self.initial_wait = initial_patience
#         self.patience = patience
#         self.wait = 0
#         self.min_lr = min_lr
#         self.verbose = verbose
#         self._reset()
#
#     def set_monitor(self, monitor):
#         super().set_monitor(monitor)
#         self.optimizers = monitor.optimizers
#
#     def _reset(self):
#         self.wait = 0
#         self.initial_wait = self.initial_patience
#
#     def on_train_begin(self):
#         self._reset()
#
#     def is_in_initial_wait(self):
#         return self.initial_wait > 0
#
#     def _message(self):
#         return 'Epoch {} | Reducing learning rate to {}'
#
#     def on_epoch_end(self):
#         if self.is_in_initial_wait():
#             self.initial_wait += -1
#             self.wait = 0
#         elif self.monitor.is_last_epoch_the_best():
#             self.wait = 0
#         else:
#             self.wait += 1
#             if self.verbose:
#                 print('DecreaseLearningRate wait is {}'.format(self.wait))
#             if self.wait >= self.patience:
#                 for optimizer in self.optimizers:
#                     old_lr = tf_get_value(optimizer.lr)
#                     new_lr = to_floatx(self.factor * old_lr)
#                     if new_lr < self.min_lr:
#                         new_lr = self.min_lr
#                     tf_set_value(optimizer.lr, new_lr)
#                     if self.verbose:
#                         print(self._message().format(
#                             self.monitor.epoch, new_lr
#                         ))
#                 self.wait = 0
#
#
# class DecreaseSdeLearningRate(DecreaseLearningRate):
#     def set_monitor(self, monitor):
#         if not isinstance(monitor, SvaeAlgorithmMonitor):
#             ValueError("'monitor' should be of class 'SvaeAlgorithmMonitor'")
#         super().set_monitor(monitor)
#         self.optimizers = [monitor.sde_optimizer]
#
#     def _message(self):
#         return 'Epoch {} | Reducing SDE learning rate to {}'
#
#
# class DecreaseNnetLearningRate(DecreaseLearningRate):
#     def set_monitor(self, monitor):
#         if not isinstance(monitor, SvaeAlgorithmMonitor):
#             ValueError("'monitor' should be of class 'SvaeAlgorithmMonitor'")
#         super().set_monitor(monitor)
#         self.optimizers = [monitor.nnet_optimizer]
#
#     def _message(self):
#         return 'Epoch {} | Reducing NNets learning rate to {}'

class LearningRateScheduler(Callback):
    def __init__(self, optimizer: ClippedOptimizer, scheduler):
        self.optimizer = optimizer
        self.scheduler = scheduler

    def on_epoch_end(self):
        new_lr = self.scheduler(self.monitor.epoch, tf_get_value(self.optimizer.lr))
        print("----> new lr: {}".format(new_lr))
        tf_set_value(self.optimizer.lr, new_lr)


class EarlyStopping(Callback):
    def __init__(self, patience=6, initial_patience=1, tolerance=1e-6, verbose=True):
        super().__init__()
        self.patience = patience
        self.initial_patience = initial_patience
        self.wait = 0
        self.initial_wait = initial_patience
        self.tolerance = tolerance
        self.verbose = verbose
        self._reset()

    def _reset(self):
        self.wait = 0
        self.initial_wait = self.initial_patience

    def is_in_initial_wait(self):
        return self.initial_wait > 0

    def on_train_begin(self):
        self._reset()

    def on_epoch_end(self):
        if self.is_in_initial_wait():
            self.initial_wait += -1
            self.wait = 0
        elif self.monitor.is_last_epoch_the_best():
            self.wait = 0
            if self.verbose:
                print(('[Early Stopping] Last epoch was the best ({})... resetting').format(
                    self.monitor.last_epoch_cost()))
        else:
            self.wait += 1
            if self.verbose:
                print('EarlyStop wait is {}'.format(self.wait))
            if self.wait >= self.patience:
                self.monitor.stop_training = True
                if self.verbose:
                    print(('Epoch {} | Early stopping').format(self.monitor.epoch))
                self.wait = 0


class SaveModel(Callback):
    def __init__(self, saver, filename, period=1, initial_patience=0, verbose=False):
        self.saver = saver
        self.filename = filename
        self.period = period
        self.epochs_since_last_save = 0
        self.best_saved_cost = -np.inf
        self.initial_patience = initial_patience
        self.initial_wait = initial_patience

        self.verbose = verbose

    def _reset(self):
        self.initial_wait = self.initial_patience

    def is_in_initial_wait(self):
        return self.initial_wait > 0

    def on_train_begin(self):
        self._reset()

    def _has_improved(self):
        return self.monitor.is_better(self.monitor.last_epoch_cost(), self.best_saved_cost)

    def on_epoch_end(self):
        if self.is_in_initial_wait():
            self.initial_wait += -1
            self.epochs_since_last_save = 0
        else:
            self.epochs_since_last_save += 1
            if self.epochs_since_last_save >= self.period and self._has_improved():
                self.best_saved_cost = self.monitor.last_epoch_cost()
                self.epochs_since_last_save = 0

                self.saver.save(self.monitor.session, self.filename)

                if self.verbose:
                    print('Epoch {} | Saving best model to {}'.format(
                        self.monitor.epoch, self.filename
                    ))


class CsvLogger(Callback):
    def __init__(self, filename, separator=',', append=False):
        super().__init__()
        self.sep = separator
        self.append = append
        self.filename = filename
        self.csv_file = None
        self.writer = None

    def on_train_begin(self):
        if self.append:
            self.csv_file = open(self.filename, 'a')
        else:
            self.csv_file = open(self.filename, 'w')
        self.writer = csv.writer(self.csv_file, delimiter=self.sep,
                                 quotechar="'", quoting=csv.QUOTE_MINIMAL)

    def on_batch_end(self):
        csv_row = [self.monitor.epoch, self.monitor.batch, self.monitor.last_batch_cost(),
                   self.monitor.best_cost]
        self.writer.writerow(csv_row)
        self.csv_file.flush()

    def on_epoch_end(self):
        csv_row = [self.monitor.epoch, 'end_of_epoch', self.monitor.last_epoch_cost(),
                   self.monitor.best_cost]
        self.writer.writerow(csv_row)
        self.csv_file.flush()

    def on_train_end(self):
        self.csv_file.close()
        self.writer = None


class ParamCsvLogger(Callback):
    def __init__(self, filename_id='/tmp/params_', separator=',', append=False):
        super().__init__()
        self.sep = separator
        self.append = append
        self.filename_id = filename_id
        self.csv_files = []
        self.writers = []

    def on_train_begin(self):
        if self.append:
            mode = 'a'
        else:
            mode = 'w'
        for type in self.monitor.model.embedding_model.sde.standard_params.keys():
            csv_file = open(self.filename_id + type + '.csv', mode)
            self.csv_files.append(csv_file)
            self.writers.append(
                csv.writer(csv_file, delimiter=self.sep,
                           quotechar="'", quoting=csv.QUOTE_MINIMAL)
            )

    def on_batch_end(self):
        for i, key in enumerate(self.monitor.model.embedding_model.sde.standard_params):
            csv_row = tf_get_value(self.monitor.model.embedding_model.sde.standard_params[key])[0]
            if key == 'covs':
                csv_row = csv_row[0]
            if isinstance(csv_row, float):
                csv_row = np.array([csv_row])
            # csv_row = ','.join([str(tmp) for tmp in csv_row])
            self.writers[i].writerow(csv_row)
            self.csv_files[i].flush()

    def on_train_end(self):
        for i in range(len(self.csv_files)):
            self.csv_files[i].close()
        self.writers = []


# class PlotPhaseSpace(Callback):
#     def __init__(self, filename_prefix, experiment, path=None, sep=',', n_points=1000, period=1):
#         super().__init__()
#         self.filename_prefix = filename_prefix
#         self.experiment = experiment
#         self.n_points = n_points
#
#         self.epochs_since_last_lims_update = 0
#         self.lims_update_period = 10
#
#         self.epochs_since_last_save = 0
#         self.period = period
#         self.save = path is not None
#         self.path = path
#         self.sep = sep
#         self.csv_files = []
#         self.writers = []
#
#     def on_train_begin(self):
#         plt.ioff()
#         if self.monitor is None:
#             raise ValueError('Set a Monitor before starting the training')
#         self.first_plot = True
#
#         self.monitor.embedded_mean = None
#         self.monitor.embedded_cov = None
#         self.monitor.samples = None
#         self.drift_predictions = None
#         self.X = None
#         self.monitor.batch_data = None
#         self.monitor.decoded_mean = None
#         self.monitor.decoded_cov = None
#         self.monitor.decoded_diff_mean = None
#         self.monitor.decoded_diff_cov = None
#
#         self.parameters_to_save = OrderedDict({
#             'embedded_mean': self.monitor.embedded_mean,
#             'embedded_cov': self.monitor.embedded_cov,
#             'samples': self.monitor.samples,
#             'drift': self.drift_predictions,
#             'X': self.X,
#             'batch_data': self.monitor.batch_data,
#             'decoded_mean': self.monitor.decoded_mean,
#             'decoded_cov': self.monitor.decoded_cov,
#             'decoded_diff_mean': self.monitor.decoded_diff_mean,
#             'decoded_diff_cov': self.monitor.decoded_diff_cov
#         })
#
#     @staticmethod
#     def eigsorted(cov):
#         vals, vecs = np.linalg.eigh(cov)
#         order = vals.argsort()[::-1]
#         return vals[order], vecs[:, order]
#
#     def save_params(self):
#         if self.save:
#             if self.epochs_since_last_save > self.period:
#                 for k, v in self.parameters_to_save.items():
#                     save_as = os.path.join(self.path, str(k) + "_" + str(self.monitor.epoch) + "_" +
#                                            str(self.monitor.batch) + '.pkl')
#                     pickle.dump(v, open(save_as, 'wb'))
#                 self.epochs_since_last_save = 0
#             else:
#                 self.epochs_since_last_save += 1
#
#     def on_epoch_end(self):
#         print('===================== SAVED ===================')
#         self.save_params()
#
#     def on_batch_end(self):
#         embedding, embedding_cov = self.monitor.embedded_mean, self.monitor.embedded_cov
#         # Pick just one sample for each batch
#         samples = self.monitor.samples
#         samples = samples[:, 0, ...]
#         ips = tf_get_value(self.monitor.model.embedding_model.sde.inducing_points)
#         fig_id = int(self.monitor.epoch * self.monitor.batches_per_epoch + self.monitor.batch)
#         with PdfPages(self.filename_prefix + '_phase_space_{:06d}'.format(fig_id) + '.pdf') as pdf:
#             fig = plt.figure()
#             dim = embedding.shape[-1]
#             if dim == 3:
#                 ax = fig.add_subplot(1, 2, 1, projection='3d')
#             else:
#                 ax = fig.add_subplot(1, 1, 1)
#             # Plot the embedding from the network, the pseudo_inputs, and an arrow to know the flow of time.
#             for s in samples:
#                 ax.plot(*[s[:, i] for i in range(dim)])
#             # ax.plot(*[samples[1][:, i] for i in range(dim)])
#             for emb in embedding:
#                 ax.plot(*[emb[:, i] for i in range(dim)])
#                 ax.scatter(*[emb[0, i] for i in range(dim)], marker="X", color="green")
#                 ax.scatter(*[emb[min(100, len(embedding) - 1), i] for i in range(dim)], marker="D", color="green")
#             ax.scatter(*[ips[:, i] for i in range(dim)], color='red')
#             if dim == 2:
#                 step = 10
#                 for emb, emb_covs in zip(embedding, embedding_cov):
#                     idx = np.arange(0, len(emb), step)
#                     aux = [self.eigsorted(ec) for ec in emb_covs]
#                     vals = np.array([a[0] for a in aux])
#                     vecs = [a[1] for a in aux]
#                     thetas = [np.degrees(np.arctan2(*v[:, 0][::-1])) for v in vecs]
#                     print('----------------')
#                     print(np.mean(emb_covs, axis=0))
#                     print(np.max(emb_covs))
#                     print(np.min(emb_covs))
#                     print('----------------')
#                     # Width and height are "full" widths, not radius
#                     aux = [2 * np.sqrt(v) for v in vals]
#                     widths = [a[0] for a in aux]
#                     heights = [a[1] for a in aux]
#                     ells = [Ellipse(xy=emb[i],
#                                     width=widths[i],
#                                     height=heights[i],
#                                     angle=thetas[i],
#                                     color='y',
#                                     alpha=0.4
#                                     # facecolor='none'
#                                     )
#                             for i in idx]
#                     for i, e in enumerate(ells):
#                         plt.scatter(emb[i][0], emb[i][1], marker='x', color='k')
#                         ax.add_artist(e)
#
#             # if dim is 2, we add the dynamic arrows directly on the plot
#             if dim == 2:
#                 if self.epochs_since_last_lims_update > self.lims_update_period or self.first_plot:
#                     def expand(m, M):
#                         fact = 1.5
#                         if m < 0 and M < 0:
#                             m *= fact
#                             M /= fact
#                         if m < 0 and M > 0:
#                             m *= fact
#                             M *= fact
#                         if m > 0 and M < 0:
#                             # impossible
#                             raise ValueError
#                         if m > 0 and M > 0:
#                             m /= fact
#                             M *= fact
#                         return m, M
#
#                     ranges = [expand(np.min(ips[:, dim_it]), np.max(ips[:, dim_it])) for dim_it in
#                               range(dim)]
#                     self.X = [
#                         np.linspace(min_X, max_X, int(self.n_points ** (1 / dim)) + 1) for min_X, max_X in ranges
#                     ]
#                     self.X = np.meshgrid(*self.X)
#                     self.first_plot = False
#                     self.epochs_since_last_lims_update = 0
#                 else:
#                     self.epochs_since_last_lims_update += 1
#                 # plt.plot(scatter, self.)
#                 sess = get_session()
#                 self.drift_predictions = (
#                     sess.run(
#                         self.monitor.model.embedding_model.sde.get_expected_drift(
#                             self.monitor.model.embedding_model.sde.input
#                         ), {
#                             self.monitor.model.embedding_model.sde.input:
#                                 [np.concatenate([self.X[i].flatten().reshape((-1, 1)) for i in range(dim)], axis=1)][0]
#                         }
#                     )
#                 )
#
#                 # predictions = predictions.reshape((dim, *(X[0].shape)))
#                 ax.quiver(*self.X, *self.drift_predictions, color='k', units='width', scale_units='xy', angles='xy')
#             # else:
#             #     ax = fig.add_subplot(1, 2, 2, projection='3d')
#             #     mini_embedding = embedding[np.arange(0, len(embedding), 300), :]
#             #     predictions = self._drift_predictions([mini_embedding])[0]
#             #     with open(os.path.expanduser("~/.vaele_factor"), "r") as f:
#             #         factor = float(f.readline())
#             #     print('===>', factor)
#             #     ax.quiver(*(mini_embedding.T),
#             #               *(factor * predictions),
#             #               normalize=False, color='k')# scale_units='xy', angles='xy')  # , units='width')
#             plt.title(str(self.monitor.epoch) + " " + str(self.monitor.batch))
#             pdf.savefig(fig)
#             plt.close()
#
#         with PdfPages(self.filename_prefix + '_output_{:06d}'.format(fig_id) + '.pdf') as pdf:
#             fig = plt.figure()
#             target = self.monitor.batch_data[..., -1]
#             nb_batches = self.monitor.batch_data.shape[0]
#             T = self.monitor.batch_data.shape[1]
#             # We use only the first sample
#             decoded_mean = self.monitor.decoded_mean[:, 0, :]
#             decoded_cov = self.monitor.decoded_cov[:, 0, :]
#             for i in range(nb_batches):
#                 time_axis = np.arange(i * T, (i + 1) * T)
#                 plt.plot(time_axis, decoded_mean[i], color='gray')
#                 plt.plot(time_axis, decoded_mean[i] - 3 * np.sqrt(decoded_cov[i]), linestyle='--', color='black')
#                 plt.plot(time_axis, decoded_mean[i] + 3 * np.sqrt(decoded_cov[i]), linestyle='--', color='black')
#                 plt.plot(time_axis, target[i])
#             pdf.savefig(fig)
#             plt.close()
#
#         with PdfPages(self.filename_prefix + '_diff_output_{:06d}'.format(fig_id) + '.pdf') as pdf:
#             # We use only the first sample
#             diff_mean = self.monitor.decoded_diff_mean[:, 0, :]
#             diff_cov = self.monitor.decoded_diff_cov[:, 0, :]
#             fig = plt.figure()
#             # TODO change to use real sampling time. Change it to be consistent with the diff in the embedding model
#             diff_series = np.diff(target, axis=-1) / 1e-2
#             # Make sure that diff series has the same length as target
#             diff_series = np.concatenate([diff_series, diff_series[..., -1:]], axis=-1)
#             for i in range(nb_batches):
#                 time_axis = np.arange(i * T, (i + 1) * T)
#                 plt.plot(time_axis, diff_mean[i], color='gray')
#                 plt.plot(time_axis, diff_mean[i] - 3 * np.sqrt(diff_cov[i]), linestyle='--', color='black')
#                 plt.plot(time_axis, diff_mean[i] + 3 * np.sqrt(diff_cov[i]), linestyle='--', color='black')
#                 plt.plot(time_axis, diff_series[i])
#             pdf.savefig(fig)
#             plt.close()


class Plot2DPhaseSpace(Callback):
    def __init__(self, model: VAE, experiment, filename_prefix, period=10, save_path=None, sep=',',
                 n_points=1000):
        super().__init__()
        self.model = model
        self.experiment = experiment

        self.filename_prefix = filename_prefix
        self.n_points = n_points

        self.steps_since_limits_update = 0
        self.limits_update_period = 10
        self.first_plot = True
        self.period = period

        self.steps_since_last_save = 0
        self.save = save_path is not None
        self.path = save_path
        self.sep = sep
        self.csv_files = []
        self.writers = []

        # Parameters computed during training
        self._forward_distributions = None
        self._samples = None
        self._X = None

    def on_train_begin(self):
        plt.ioff()
        if self.monitor is None:
            raise ValueError('Set a Monitor before starting the training')

    @staticmethod
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    def save_params(self, params: dict):
        if self.save and self.steps_since_last_save >= self.period:
            print("Saving plotting variables...")
            for k, v in params.items():
                save_as = os.path.join(self.path, str(k) + "_" + str(self.monitor.epoch) + "_" +
                                       str(self.monitor.batch) + '.pkl')
                pickle.dump(v, open(save_as, 'wb'))
            self.steps_since_last_save = 0
        else:
            self.steps_since_last_save += 1

    def on_epoch_end(self):
        pass
        # TODO
        # self.save_params(None)

    def on_batch_end(self):
        # Some constant values that we define here
        covs_steps = 10

        decoded_cov, decoded_mean, embeddings, filtered_covs, samples = self._run_tf_graph()
        # Pick just one sample for each batch
        samples = samples[:, np.random.choice(np.arange(self.experiment.nb_samples)), ...]
        inducing_points = tf_get_value(self.model.sde.inducing_points)
        fig_id = int(self.monitor.global_step)
        mean, diff_mean = decoded_mean[..., 0], decoded_mean[..., 1]
        cov, diff_cov = decoded_cov[..., 0], decoded_cov[..., 1]
        # Since there is a mean and cov for each sample, we use only the first one:
        mean, cov = mean[:, 0, :], cov[:, 0, :]
        diff_mean, diff_cov = diff_mean[:, 0, :], diff_cov[:, 0, :]
        ys = self.monitor.batch_data[..., -1]
        batch_size = self.monitor.batch_data.shape[0]
        T = self.monitor.batch_data.shape[1]

        drift_predictions = self._plot_phase_space(covs_steps, embeddings, fig_id, filtered_covs, inducing_points,
                                                   samples)
        self._plot_drifts(embeddings, fig_id)
        self._plot_y_and_estimates(T, batch_size, cov, fig_id, mean, ys)
        self._plot_dy_and_estimates(T, batch_size, diff_cov, diff_mean, fig_id, ys)

        self.save_params({
            "embeddings": embeddings,
            "filtered_covs": filtered_covs,
            "inducing_points": inducing_points,
            "samples": samples,
            "ys": ys,
            "Xs": self._X,
            "drifts": drift_predictions
        })

    def _plot_drifts(self, embeddings, fig_id):
        with PdfPages(self.filename_prefix + '_drifts_{:06d}'.format(fig_id) + '.pdf') as pdf:
            fig = plt.figure()
            ax = fig.add_subplot(2, 2, 1)
            for embedding in embeddings:
                dy = embedding[1:] - embedding[:-1]
                y = embedding[:-1]
                dp = self.monitor.session.run(self.model.sde.output, {self.model.sde.input: y})
                ax.plot(y[:, 0], dy[:, 0], color="blue")
                ax.plot(y[:, 0], dp[0], color="red")

            ax2 = fig.add_subplot(2, 2, 2)
            for embedding in embeddings:
                dy = embedding[1:] - embedding[:-1]
                y = embedding[:-1]
                dp = self.monitor.session.run(self.model.sde.output, {self.model.sde.input: y})
                ax2.plot(y[:, 0], dy[:, 1], color="blue")
                ax2.plot(y[:, 0], dp[1], color="red")

            ax3 = fig.add_subplot(2, 2, 3)
            for embedding in embeddings:
                dy = embedding[1:] - embedding[:-1]
                y = embedding[:-1]
                dp = self.monitor.session.run(self.model.sde.output, {self.model.sde.input: y})
                ax3.plot(y[:, 1], dy[:, 0], color="blue")
                ax3.plot(y[:, 1], dp[0], color="red")

            ax4 = fig.add_subplot(2, 2, 4)
            for embedding in embeddings:
                dy = embedding[1:] - embedding[:-1]
                y = embedding[:-1]
                dp = self.monitor.session.run(self.model.sde.output, {self.model.sde.input: y})
                ax4.plot(y[:, 1], dy[:, 1], color="blue")
                ax4.plot(y[:, 1], dp[1], color="red")

            pdf.savefig(fig)
            plt.close()

    def _plot_phase_space(self, covs_steps, embeddings, fig_id, filtered_covs, inducing_points, samples):
        with PdfPages(self.filename_prefix + '_phase_space_{:06d}'.format(fig_id) + '.pdf') as pdf:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            sample_colors = ['b', 'r', 'c', 'm', 'y', 'k']
            for i, trajectory in enumerate(samples):
                ax.plot(*[trajectory[:, i] for i in range(2)], sample_colors[i % len(sample_colors)])

            # Encoder mean + inducing_points
            for embedding in embeddings:
                ax.plot(*[embedding[:, i] for i in range(2)], color="green")
                # ax.scatter(*[embedding[0, i] for i in range(2)], marker="D", color="green")
                # ax.scatter(*[embedding[-1, i] for i in range(2)], marker="X", color="green")
                dy = embedding[1:] - embedding[:-1]
                idx = np.arange(0, len(embedding), 10)
                ax.quiver(*embedding[idx].T, *dy[idx].T, color='darkslategrey', units='dots', angles='xy')
            ax.scatter(*[inducing_points[:, i] for i in range(2)], color='red')

            self._plot_filtered_covs(filtered_covs, embeddings, ax, covs_steps)

            if self.steps_since_limits_update > self.limits_update_period or self.first_plot:
                self._X = self._update_limits(inducing_points, embeddings.reshape((-1, 2)))
                self.steps_since_limits_update = 0
                self.first_plot = False
            else:
                self.steps_since_limits_update += 1

            drift_predictions = self.run_drift_predictions(self._X)
            ax.quiver(*self._X, *drift_predictions, color='k', units='dots', angles='xy')
            plt.title(str(self.monitor.epoch) + " " + str(self.monitor.batch))
            pdf.savefig(fig)
            plt.close()
            return drift_predictions

    def _plot_dy_and_estimates(self, T, batch_size, diff_cov, diff_mean, fig_id, ys):
        with PdfPages(self.filename_prefix + '_diff_output_{:06d}'.format(fig_id) + '.pdf') as pdf:
            fig = plt.figure()
            # TODO change to use real sampling time. Change it to be consistent with the diff in the embedding model
            dy = np.diff(ys, axis=-1) / self.experiment.sampling_time
            # Make sure that diff series has the same length as target
            dy = np.concatenate([dy, dy[..., -1:]], axis=-1)
            for i in range(batch_size):
                time_axis = np.arange(i * T, (i + 1) * T)
                plt.plot(time_axis, diff_mean[i], color='gray')
                plt.plot(time_axis, diff_mean[i] - 3 * np.sqrt(diff_cov[i]), linestyle='--', color='black')
                plt.plot(time_axis, diff_mean[i] + 3 * np.sqrt(diff_cov[i]), linestyle='--', color='black')
                plt.plot(time_axis, dy[i])
            pdf.savefig(fig)
            plt.close()

    def _plot_y_and_estimates(self, T, batch_size, cov, fig_id, mean, ys):
        with PdfPages(self.filename_prefix + '_output_{:06d}'.format(fig_id) + '.pdf') as pdf:
            fig = plt.figure()
            for i in range(batch_size):
                time_axis = np.arange(i * T, (i + 1) * T)
                plt.plot(time_axis, mean[i], color='gray')
                plt.plot(time_axis, mean[i] - 3 * np.sqrt(cov[i]), linestyle='--', color='black')
                plt.plot(time_axis, mean[i] + 3 * np.sqrt(cov[i]), linestyle='--', color='black')
                plt.plot(time_axis, ys[i])
            pdf.savefig(fig)
            plt.close()

    def run_drift_predictions(self, X):
        drift_predictions = (
            self.monitor.session.run(
                self.model.sde.output, {
                    self.model.sde.input:
                        [np.concatenate([X[i].flatten().reshape((-1, 1)) for i in range(2)], axis=1)][0]
                }
            )
        )
        return drift_predictions

    @staticmethod
    def __expand(m, M, expansion_factor=1.05):
        default_range = (-1.5, 1.5)
        lower_bound, upper_bound = expand_range(m, M, expansion_factor)
        return min(lower_bound, default_range[0]), max(upper_bound, default_range[1])

    def _update_limits(self, inducing_points, data):
        full_data = np.concatenate([inducing_points, data], axis=0)
        ranges = [self.__expand(np.min(full_data[:, dim_it]), np.max(full_data[:, dim_it])) for dim_it
                  in range(2)]
        X = [np.linspace(min_X, max_X, int(self.n_points ** (1 / 2)) + 1) for min_X, max_X in ranges]
        X = np.meshgrid(*X)
        return X

    def _plot_filtered_covs(self, filtered_covs, embeddings, ax, covs_steps):
        for embedding, embedding_covs in zip(embeddings, filtered_covs):
            idx = np.arange(0, len(embedding), covs_steps)
            eigs_decomposition = [self.eigsorted(ec) for ec in embedding_covs]
            eigs = np.array([a[0] for a in eigs_decomposition])
            eigenvecs = [a[1] for a in eigs_decomposition]
            thetas = [np.degrees(np.arctan2(*v[:, 0][::-1])) for v in eigenvecs]
            # print(np.mean(embedding_covs, axis=0), "\n")
            # The width and height parameters of the Ellipse are diameters
            eigs_decomposition = [2 * np.sqrt(eig) for eig in eigs]
            widths = [eig[0] for eig in eigs_decomposition]
            heights = [eig[1] for eig in eigs_decomposition]
            for i in idx:
                plt.scatter(embedding[i][0], embedding[i][1], marker='x', color='k')
                ellipse = Ellipse(xy=embedding[i], width=widths[i], height=heights[i], angle=thetas[i],
                                  color='y', alpha=0.4)
                ax.add_artist(ellipse)

    def _run_tf_graph(self):
        [embeddings, decoded_mean, decoded_cov] = self.monitor.session.run([
            self.model.mean_encoder.output, self.model.mean_decoder.output, self.model.cov_decoder.output
        ], {
            self.model.inputs: self.monitor.batch_data,
            self.model.training: False,
            self.model.initial_means: self.monitor.training_state_means,
            self.model.initial_precs: self.monitor.training_state_precs,
            self.model.use_initial_state: self.monitor.use_initial_state,
            # RNN initial states
            self.model.rnn_initial_states[0]: self.monitor.rnn_initial_states[0],
            self.model.rnn_initial_states[1]: self.monitor.rnn_initial_states[1],
            self.model.rnn_initial_states[2]: self.monitor.rnn_initial_states[2],
            self.model.rnn_initial_states[3]: self.monitor.rnn_initial_states[3]
        })
        return (
            decoded_cov, decoded_mean, embeddings, self.monitor.distributions[1], self.monitor.inferred_samples
        )

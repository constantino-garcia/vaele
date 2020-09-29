import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist

from experiments.Experiment import Experiment
from magic_numbers import AMP, ALPHA, BETA
from SdeModel import DriftParameters, DiffParameters
from utils.inducing_points import grid_based_inducing_points
from utils.train import tbptt_chunks_generator
from vaele_config import np_floatx
from VAE import VAE


class GraphicalModelFactory:
    @staticmethod
    def build(encoder_type, encoder_hidden_units, encoder_kernel_size, encoder_dilation_rate,
              phase_space_dim, nb_pseudo_inputs, nb_samples, experiment: Experiment, data_ranges=None):

        #TODO: decide proper range
        # IPS (PIS)
        if data_ranges is None:
            data_ranges = [(-5, 5) for _ in range(phase_space_dim)]
        pseudo_inputs = grid_based_inducing_points(
            data_ranges=data_ranges,
            nb_pseudo_inputs=nb_pseudo_inputs,
            expansion_factor=1.0
        )
        # TODO
        # length_scale = np.quantile(pdist(pseudo_inputs), 0.1) / 2
        length_scale = 1.0
        print(length_scale)
        input_dim, amplitude_dy, alpha, beta = GraphicalModelFactory._get_estimates_from_data(experiment)

        drift_params = DriftParameters(
            amplitudes=tf.convert_to_tensor(np.repeat(amplitude_dy, phase_space_dim).astype(np_floatx())),
            length_scales=tf.convert_to_tensor(
                length_scale * np.ones((phase_space_dim, phase_space_dim)).astype(
                    np_floatx()
                )
            )
        )
        diff_params = DiffParameters(
            alphas=tf.convert_to_tensor(np.repeat(alpha, phase_space_dim).astype(np_floatx())),
            betas=tf.convert_to_tensor(np.repeat(beta, phase_space_dim).astype(np_floatx()))
        )

        if encoder_type == 'rnn':
            encoder_kernel_size = 0
            encoder_dilation_rate = 0

        elif encoder_type != 'cnn':
            raise ValueError('Invalid encoder type')

        gm = VAE(
            encoder_type=encoder_type,
            encoder_hidden_units=encoder_hidden_units,
            encoder_kernel_size=encoder_kernel_size,
            encoder_dilation_rate=encoder_dilation_rate,
            phase_space_dimension=phase_space_dim,
            nb_samples=nb_samples,
            input_dimension=input_dim,
            output_dimension=input_dim,
            len_tbptt=experiment.len_tbptt,
            drift_parameters=drift_params,
            diff_parameters=diff_params,
            pseudo_inputs=pseudo_inputs
        )

        # Make a first call to the graphical model to build it (the input_shapes must be available for building)
        print("Building Graphical Model...", end="")
        for y in experiment.train_dataset.take(1):
            for x_chunk, y_chunk in tbptt_chunks_generator(y, experiment.len_tbptt, experiment.time_lag,
                                                           gm.encoder.kernel_size, gm.encoder.dilation_rate):
                _ = gm.loss(x_chunk, y_chunk)
                break
        print("Done!")

        return gm

    @staticmethod
    def _get_estimates_from_data(experiment):
        # We use the train_dataset to set some conservative estimates of the drift's amplitudes,
        # and the Diffusion's alphas and betas.
        # regression_X = []
        # regression_y = []
        # for y in experiment.train_dataset:
        #     pass
        # # axis 1 is the temporal axis (axis 0 is batch and -1 the input_dimension)
        #     # dy will be temporarily correlated. Split in windows to get decorrelate estimates of the variance
        #     dy = np.diff(y, axis=1).reshape((-1, y.shape[-1]))
        #     regression_X.append(np.arange(dy.shape[0]))
        #     regression_y.append(dy)
        #
        # np.savetxt('/tmp/y.txt', y)
        #
        # regression_X = np.concatenate(regression_X)
        # regression_y = np.concatenate(regression_y)
        # import sklearn
        # regression_X = sklearn.preprocessing.scale(regression_X)
        #
        # import skmisc
        # import skmisc.loess
        # loess = skmisc.loess.loess(regression_X, regression_y.flatten(), span=0.3)
        # p = loess.predict(regression_X, stderror=True)
        # aaa = p.confidence()
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(regression_y.flatten())

        # plt.plot(aaa.fit)
        # # plt.plot(p.values + 3 * p.stderr)
        # # plt.plot(p.values - 3 * p.stderr)
        # plt.plot(aaa.lower, c='red')
        # plt.plot(aaa.upper, c='red')
        # plt.savefig('/tmp/waka')
        # # This is a very rough guess of alpha and beta

        amplitude_dy = AMP
        alpha = ALPHA
        beta = BETA

        input_dim = 1#y.shape[-1]

        return input_dim, amplitude_dy, alpha, beta
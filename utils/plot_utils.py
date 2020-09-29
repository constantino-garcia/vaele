import io
import numpy as np
import matplotlib
# matplotlib.use('Agg')
from matplotlib import cm, pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from sklearn.decomposition import PCA
from VAE import VAE

import time

# def plt_to_tf_image():
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     # Convert PNG buffer to TF image
#     image = tf.image.decode_png(buf.getvalue(), channels=4)
#     # Add the batch dimension
#     image = tf.expand_dims(image, 0)
#     return image


# def plot_decoder(y_, decoder_output_, nb_examples, dim, ci_scale=2):
#     decoder_mean_, decoder_diag_scale_ = decoder_output_
#     figures = []
#     for i in range(nb_examples):
#         figures.append(
#             plot_decoder_example(y_[i, :, :], (decoder_mean_[:, i, ...], decoder_diag_scale_[:, i, ...]), dim, ci_scale)
#         )
#     return figures


# def plot_decoder_example(y_, decoder_output_, dim, ci_scale=2):
#     """
#     :param y_: a single example with shape [time_steps, dim]
#     :param decoder_output_: the decoder output for a single example. For example, shape of mean is expected to be
#     [nb_samples, time, dim]
#     :param dim: which dimension to visualize
#     :return:
#     """
#     decoder_mean_, decoder_diag_scale_ = decoder_output_
#
#     # We will only plot the 'dim' dimension
#     y = y_[:, dim].numpy()
#     # Regarding the samples, We will only use the first one.
#     decoder_mean = decoder_mean_[0, :, dim].numpy()
#     decoder_diag_scale = decoder_diag_scale_[0, :, dim].numpy()
#
#     time = np.arange(len(y))
#     time_decoder = np.arange(len(decoder_mean))
#     plt.figure()
#     plt.plot(time, y)
#     plt.fill_between(time_decoder,
#                      decoder_mean - ci_scale * decoder_diag_scale,
#                      decoder_mean + ci_scale * decoder_diag_scale,
#                      alpha=0.5, color='orange'
#                      )
#     plt.plot(time_decoder, decoder_mean)
#     plt.legend(['Ground truth: y[n]', 'Decoder prediction'])
#     plt.xlabel('Number of sample: n')
#
#     return plt_to_tf_image()
#
#
# def plot_phase_space(samples_, encoder_output_, Z_, sde_model, nb_examples, representation_dim=2, plotting_range=None,
#                      drift_points_per_axis=10):
#     encoder_mean_, encoder_diag_scale_ = encoder_output_
#
#     if plotting_range is None:
#         plotting_range = (Z_.numpy().min(),  Z_.numpy().max())
#
#     assert representation_dim == 2 or representation_dim == 3, "Invalid representation dim"
#
#     if representation_dim == 2:
#         fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
#         axs = [ax1, ax2, ax3]
#     else:
#         fig = plt.figure()
#         ax1 = fig.add_subplot(2, 2, 1, projection='3d')
#         ax2 = fig.add_subplot(2, 2, 2)
#         ax3 = fig.add_subplot(2, 2, 3)
#         ax4 = fig.add_subplot(2, 2, 4)
#         axs = [ax1, ax2, ax3, ax4]
#
#     for i in range(nb_examples):
#         do_quiver = i == 0
#         do_ips = i == (nb_examples - 1)  # IP: Inducing Point
#         # We only represent the first sample
#         plot_phase_space_example(axs, samples_[0, i, ...].numpy(), encoder_mean_[i, ...].numpy(), Z_.numpy(),
#                                  sde_model, representation_dim, plotting_range, drift_points_per_axis,
#                                  do_quiver=do_quiver, do_ips=do_ips)
#
#     return plt_to_tf_image()
#
#
# def plot_decoder_space(samples_, encoder_output_, Z_, decoder, nb_examples, dim, plotting_range=None,
#                        drift_points_per_axis=20):
#     encoder_mean_, encoder_diag_scale_ = encoder_output_
#
#     if plotting_range is None:
#         plotting_range = (
#             samples_.numpy().min(),
#             samples_.numpy().max()
#         )
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 1, 1, projection='3d')
#
#     x = np.linspace(*plotting_range, drift_points_per_axis)
#     X, Y = np.meshgrid(x, x)
#     XY = np.stack([np.array([x, y])[np.newaxis, ...] for x, y in zip(X.flatten(), Y.flatten())])
#     fx, _ = decoder(XY)
#     Z = fx[..., 0, dim].numpy().reshape(X.shape)
#     ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#     ax1.view_init(elev=20., azim=20)
#
#     # plt.savefig('/tmp/test.png')
#     img1 = plt_to_tf_image()
#
#     fig = plt.figure()
#     ax1 = fig.add_subplot(1, 1, 1)
#     for i in range(nb_examples):
#         samples = samples_[0, i, ...].numpy()
#         enc = encoder_mean_[i, ...].numpy()
#         ax1.plot(*[enc[:, jj] for jj in range(2)])
#         ax1.plot(*([samples[:, jj] for jj in range(2)]))
#     if Z_ is not None:
#         ax1.plot(Z_.numpy()[:, 0], Z_.numpy()[:, 1], 'o')
#
#     ax1.contourf(X, Y, Z, cmap=cm.coolwarm)
#
#     return img1, plt_to_tf_image()
#
#
# def plot_phase_space_example(axs, samples, encoder_mean, Z, sde_model, representation_dim, plotting_range,
#                              drift_points_per_axis, do_quiver=False, do_ips=False):
#     if representation_dim == 2:
#         ax1, ax2, ax3 = axs
#     else:
#         ax1, ax2, ax3, ax4 = axs
#
#     ax1.plot(*[encoder_mean[:, i] for i in range(representation_dim)])
#     ax1.plot(*[samples[:, i] for i in range(representation_dim)])
#     if do_ips:
#         ax1.plot(*[Z[:, i] for i in range(representation_dim)], 'o')
#     # only plot the drift if samples's dim and representation's dim is 2
#     # Note we use samples_: the original samples for the check
#     if do_quiver and samples.shape[-1] == 2 and representation_dim == 2:
#         x = np.linspace(*plotting_range, drift_points_per_axis)
#         X, Y = np.meshgrid(x, x)
#         XY = np.stack([np.array([x, y]) for x, y in zip(X.flatten(), Y.flatten())])
#         fx, _ = sde_model.drift_svgp.predict_f(XY)
#         # This may fail if the length of the arrows is 0 (that is, at the beginning, when the drift is 0)
#         try:
#             ax1.quiver(XY[:, 0], XY[:, 1], fx[:, 0], fx[:, 1])
#         except:
#             print('Not showing drift: drift is 0')
#
#     ax2.plot(encoder_mean[:, 0])
#     ax2.plot(samples[:, 0])
#     ax3.plot(encoder_mean[:, 1])
#     ax3.plot(samples[:, 1])
#     if representation_dim == 3:
#         ax4.plot(samples[:, 2])
#         ax4.plot(encoder_mean[:, 2])


# def plot_drift_3d(samples_, sde_model, drift_points_per_axis=50):
#
#     plotting_range = (
#         samples_.numpy().min(),
#         samples_.numpy().max()
#     )
#
#     x = np.linspace(*plotting_range, drift_points_per_axis)
#     X, Y = np.meshgrid(x, x)
#     XY = np.stack([np.array([x, y])[np.newaxis, ...] for x, y in zip(X.flatten(), Y.flatten())])
#     fx, var_fx = sde_model.drift_svgp.predict_f(XY)
#     imgs = []
#     for dim in range(2):
#         fig = plt.figure()
#         ax1 = fig.add_subplot(1, 1, 1, projection='3d')
#         Z = fx[..., 0, dim].numpy().reshape(X.shape)
#         ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#         ax1.view_init(elev=20., azim=20)
#         imgs.append(plt_to_tf_image())
#
#     #ax1.contourf(X, Y, Z, cmap=cm.coolwarm)
#     return tf.concat(imgs, 0)
#
#
# def plot_drift(samples_, sde_model, ci_scale=2.0):
#     # Pick only the first batch for simplicity
#     samples = samples_[:, 0:1, :, :]
#     dx = samples[:, :, 1:, :] - samples[:, :, :-1, :]
#     x = tf.reshape(samples[:, :, :-1, :], [-1, samples.shape[-1]])
#     dx = tf.reshape(dx, (-1, dx.shape[-1]))
#     fx, var_fx = sde_model.drift_svgp.predict_f(x)
#     dimension = samples.shape[-1]
#     plt.figure()
#     plot_counter = 1
#     x = x.numpy()
#     dx = dx.numpy()
#     fx = fx.numpy()
#     sfx = np.sqrt(var_fx.numpy())
#
#     for i in range(dimension):
#         indices = np.argsort(x[:, i])
#         for j in range(dimension):
#             plt.subplot(dimension, dimension, plot_counter)
#             plt.plot(x[:, i][indices], dx[:, j][indices], 'o')
#             plt.fill_between(x[:, i][indices],
#                              fx[:, j][indices] - ci_scale * sfx[:, j][indices],
#                              fx[:, j][indices] + ci_scale * sfx[:, j][indices],
#                              alpha=0.5, color='orange'
#                              )
#             plt.plot(x[:, i][indices], fx[:, j][indices])
#             plt.title(str(i) + '-' + str(j))
#             plot_counter += 1
#     plt.tight_layout()
#     return plt_to_tf_image()


def plot_decoder(fig, axes, svae, y_input, y_target):
    batches = []
    _, _, (decoded_means, decoded_scales_diag), _ = svae._encode_and_decode(y_input, False, None)
    # Focus on first sample
    means = decoded_means[0]
    scales = decoded_scales_diag[0]
    drop = ((svae.encoder.kernel_size - 1) * svae.encoder.dilation_rate)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    for ax, yy, mean, scale in zip(axes, y_target, means, scales):
        ax.plot(yy[drop:, 0])
        ax.plot(mean[..., 0])
        ax.fill_between(np.arange(len(mean[..., 0])),
                        mean[..., 0] - 2 * scale[..., 0], mean[..., 0] + 2 * scale[..., 0],
                        color='orange', alpha=0.2
                        )


def plot_synthetic_samples(fig, axes, vae: VAE, y_input, y_target, simulation_steps):
    # print("------------------> Starting synthetic samples")
    # start = time.time() # TODO
    if isinstance(axes, np.ndarray):
        # Axes is a matrix (or vector) where the number of rows can be either 1 or 2 (do not or do phase_space)
        # and the number of columns is the number of examples to plot
        if axes.ndim == 2:
            assert axes.shape[0] == 1 or axes.shape[0] == 2, "Non valid number of rows"
            do_phase_space = axes.shape[0] == 2
            nb_examples = axes.shape[1]
        else:
            # If axes is a vector, interpret it as a matrix (1, -1)
            do_phase_space = False
            nb_examples = len(axes)
            axes = np.reshape(axes, (1, -1))
    else:
        do_phase_space = False
        nb_examples = 1
        # homogenize access to the axes
        axes = np.array([[axes]])
    nb_examples = min(nb_examples, y_input.shape[0])

    (samples, _, _), _, _, _ = vae._encode_and_decode(y_input, False, None)
    (mean, scale), synthetic_samples = vae.synthetize(y_input, y_target, simulation_steps)
    ips = vae.sde_model.drift_svgp.inducing_variable.variables[0]
    ax_lim = [
        min(np.min(samples[0, :nb_examples, ...]), np.min(synthetic_samples[:nb_examples, ...])),
        max(np.max(samples[0, :nb_examples, ...]), np.max(synthetic_samples[:nb_examples, ...]))
    ]
    for example_index in range(nb_examples):
        dec_ax = axes[0, example_index]
        phase_space_ax = axes[1, example_index] if do_phase_space else None
        _plot_synthetic_sample(
            dec_ax, phase_space_ax,
            y_target[example_index, :, 0],
            mean[example_index, :, 0],
            scale[example_index, :, 0],
            samples[0, example_index, ...],
            synthetic_samples[example_index, ...],
            vae, ax_lim
        )
        if samples.shape[-1] == 2 and phase_space_ax:
            phase_space_ax.scatter(ips[:, 0], ips[:, 1], c='C3')
    # print("------------------> End synthetic samples: {}".format(time.time() - start))


#
def _plot_synthetic_sample(dec_ax, phase_space_ax, y_target, mean, scale, samples, synthetic_samples, vae: VAE,
                           ax_lim):
    dec_ax.plot(y_target)
    time = len(y_target) - 1 + np.arange(len(mean))
    dec_ax.plot(time, mean, color='orange')
    dec_ax.fill_between(time, mean - 2 * scale, mean + 2 * scale, alpha=0.2, color='orange')
    if phase_space_ax:
        phase_space_ax.set_ylim(ax_lim)
        phase_space_ax.set_xlim(ax_lim)
        # We only plot the sample of each batch, and the first two dimensions
        phase_space_ax.plot(*[samples[..., i] for i in range(2)])
        phase_space_ax.plot(*[synthetic_samples[..., i] for i in range(2)], color='orange')
        # plot the drift's vector field only in the case of two dimensional phase space
        if samples.shape[-1] == 2:
            x = np.linspace(tf.reduce_min(synthetic_samples), tf.reduce_max(synthetic_samples), 10)
            X, Y = np.meshgrid(x, x)
            XY = np.stack([np.array([x, y]) for x, y in zip(X.flatten(), Y.flatten())])
            fx, _ = vae.sde_model.drift_svgp.predict_f(XY)
            # This may fail if the length of the arrows is 0 (that is, at the beginning, when the drift is 0)
            try:
                phase_space_ax.quiver(XY[:, 0], XY[:, 1], fx[:, 0], fx[:, 1])
            except:
                pass

# def plot_predictions(x_chunk, y_chunk, gm: VAE, simulation_steps, representation_dim=2, plotting_range=None,
#                      drift_points_per_axis=10):
#     Z_ = gm.sde_model.iv_values()
#     if plotting_range is None:
#         plotting_range = (Z_.numpy().min(),  Z_.numpy().max())
#
#     assert representation_dim == 2 or representation_dim == 3, "Invalid representation dim"
#
#     if representation_dim == 2:
#         fig, (ax1, ax2) = plt.subplots(2, 1)
#     else:
#         fig = plt.figure()
#         ax1 = fig.add_subplot(2, 1, 1, projection='3d')
#         ax2 = fig.add_subplot(2, 1, 2)
#
#     samples, encoded_dist, decoded_dist, loss, states = gm(x_chunk, y_chunk, training=False,
#                                                            initial_state=None,
#                                                            effective_nb_timesteps=2.0 # TODO
#                                                            )
#     # Use last samples as starting point. We use just the first sample from each batch
#     samples = samples[0, :, :, :]
#     chunk_size = samples.shape[1]
#     initial_points = samples[:, -1, :]
#     predicted_samples = gm.sde_model.sample_trajectories(initial_points, simulation_steps)
#
#     decoder_outputs = []
#     nb_chunks = predicted_samples.shape[1] // chunk_size
#     assert nb_chunks >= 1, "Insufficient number of simulation_steps"
#     for chunk in range(nb_chunks):
#         start = chunk * chunk_size
#         end = (chunk + 1) * chunk_size
#         decoder_outputs.append(gm.decoder(predicted_samples[:, start:end, :]))
#
#     mean, scale = zip(*decoder_outputs)
#     mean = tf.concat(mean, axis=1)
#     scale = tf.concat(scale, axis=1)
#
#     # For the moment, plot only the first example:
#     for example in range(1):
#         ax1.plot(*[samples[example, :, i] for i in range(representation_dim)])
#         ax1.plot(*[predicted_samples[example, :, i] for i in range(representation_dim)])
#
#         ax2.plot(y_chunk[example, :, 0])
#         time = chunk_size + np.arange(mean.shape[1]) - 1
#         ax2.plot(time, mean[example, :, 0])
#         ax2.fill_between(time,
#                          mean[example, :, 0] - 2 * scale[example, :, 0],
#                          mean[example, :, 0] + 2 * scale[example, :, 0],
#                          alpha=0.2, color='orange')
#
#     if samples.shape[-1] == 2 and representation_dim == 2:
#         x = np.linspace(*plotting_range, drift_points_per_axis)
#         X, Y = np.meshgrid(x, x)
#         XY = np.stack([np.array([x, y]) for x, y in zip(X.flatten(), Y.flatten())])
#         fx, _ = gm.sde_model.drift_svgp.predict_f(XY)
#         # This may fail if the length of the arrows is 0 (that is, at the beginning, when the drift is 0)
#         try:
#             ax1.quiver(XY[:, 0], XY[:, 1], fx[:, 0], fx[:, 1])
#         except:
#             print('Not showing drift: drift is 0')
#
#     return plt_to_tf_image()
#
#
def plot_encoded_samples(fig, axes, svae, y):
    ((samples, _, (particles, log_weights)), (encoded_means, encoded_scales), _, _, ) = svae._encode_and_decode(y, False, None)
    # Focus on first batch, first sample...
    samples = samples[0, 0, ...]
    ms = encoded_means[0, ...]
    ss = encoded_scales[0, ...]
    particles = particles[:, 0, ...]
    log_weights = log_weights[:, 0, :]
    w = tf.exp(log_weights)
    if isinstance(axes, np.ndarray):
        axes = axes.flatten()
    for ax, sample, m, s, p in zip(axes, tf.transpose(samples), tf.transpose(ms), tf.transpose(ss),
                                      tf.transpose(particles, [2, 0, 1])):
        ax.plot(m, label='mean', color='blue')
        ax.fill_between(np.arange(m.shape[0]), m - s, m + s, color='blue', alpha=0.2)
        ax.plot(sample, label='sample', color='orange')
        ax.scatter(np.tile(np.arange(p.shape[0]), (p.shape[1], 1)).T.ravel(),
                   p, s=w * 1000 / np.sqrt(w.shape[1]),
                   alpha=0.15, label='Particles')
    ax.legend()


def plot_drift_predictions(fig, axes, svae, y):
    ((samples, _, _), _, _, _, ) = svae._encode_and_decode(y, False, None)
    x = tf.reshape(samples, (-1, samples.shape[-1]))
    fx, var_fx = svae.sde_model.drift_svgp.predict_f(x)
    predicted_mean = tf.reshape(x + fx, samples.shape)
    fxs = tf.reshape(fx, samples.shape)
    predicted_scale = tf.reshape(tf.sqrt(var_fx), samples.shape)
    # Pick only first sample
    samples = samples[0]
    predicted_mean = predicted_mean[0]
    fxs = fxs[0]
    predicted_scale = predicted_scale[0]

    if not isinstance(axes, np.ndarray) or axes.ndim != 2:
        axes = axes.reshape((1, -1))
    for ax_row, yy, mean, fx, scale in zip(axes, samples, predicted_mean, fxs, predicted_scale):
        for i, ax in enumerate(ax_row):
            ax.plot(yy[..., i])
            # predictions are delayed by 1 sample with respect to yy
            indexes = np.arange(1, len(mean[..., i]) + 1)
            ax.plot(indexes, mean[..., i])
            ax.fill_between(
                indexes,
                mean[..., i] - 2 * scale[..., i], mean[..., i] + 2 * scale[..., i],
                color='orange', alpha=0.2
            )
            ax.plot(indexes, fx[..., i])
    ax.legend(['samples', 'predictions', 'fx'])

import os

from vaele_config import tf_floatx
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

from SdeModel import SdeModel
from VAE import VAE
from utils.math import build_delay_space


def draw_fast_samples(vae, initial_state, x_chunk):
    rnn_state, mean0, scale0 = vae._handle_x0_state(initial_state, x_chunk)
    mean, scales, states = vae.encoder(x_chunk, training=True, initial_state=rnn_state)
    distro = tfd.MultivariateNormalDiag(mean, scales)
    samples = distro.sample(vae.mpa.nb_samples)
    covs = tfp.stats.covariance(tf.reshape(samples, (*samples.shape[:2], -1)), sample_axis=0)
    # TODO
    covs = covs + 1e-8 * tf.eye(covs.shape[1], batch_shape=[covs.shape[0]], dtype=tf_floatx())
    entropies = 0.5 * covs.shape[1] * (
            1 + tf.math.log(2 * tf.constant(np.pi, dtype=tf_floatx()))) + 0.5 * tf.linalg.logdet(covs)
    return samples, entropies, (mean, scales), (mean0, scale0), states


def add_bursts(y, distros=None, min_noise=0.02, noise_std=1):
    events = np.zeros_like(y)
    for i in range(events.shape[0]):
        start = np.random.choice(np.arange(y.shape[-2]))
        length = int(np.random.default_rng().exponential(20))
        if length > 0:
            events[i, start:(start + length), :] = 1
    noise = min_noise + np.random.normal(0, noise_std, size=events.shape) * events
    return y + tf.convert_to_tensor(noise)


def tbptt_chunks_generator(data_, len_tbptt, time_lag, kernel_size, dilation_rate,
                           noise_std=0, do_bursts=True):
    # TODO
    # rossler 3, 15
    # lorenz, 5, time_lag
    target, data = build_delay_space(data_, 5, time_lag)
    prediction_lag = 0
    len_tbptt = min(data.shape[1], len_tbptt)
    do_bursts = False
    # Length that can be used to generate both the lagged version and the target signal from
    # the original data
    nb_drop = (kernel_size - 1) * dilation_rate
    nb_chunks = int(np.floor((data.shape[1] - max(prediction_lag + nb_drop, nb_drop)) / len_tbptt))
    if nb_chunks < 1:
        raise RuntimeError('Cannot generate chunks in tbptt')
    for i in range(nb_chunks):
        # Add extra samples to handle that the convolution removes the nb_drop lattest ones
        inputs = data[:, (i * len_tbptt):((i + 1) * len_tbptt + nb_drop), :]
        # FIXME
        # output = tf.concat([
        #     data[:, (i * len_tbptt + nb_drop):((i + 1) * len_tbptt + nb_drop), :],
        #     data[:, (i * len_tbptt + prediction_lag + nb_drop):((i + 1) * len_tbptt + prediction_lag + nb_drop), :],
        # ], axis=-1)
        # output = output[:, nb_drop:(-nb_drop), :]
        output = target[:, (i * len_tbptt):((i + 1) * len_tbptt + nb_drop), :]
        # TODO: noise
        if do_bursts:
            inputs = add_bursts(inputs)
        elif noise_std > 0:
            inputs = inputs + tf.random.normal(inputs.shape, stddev=noise_std, dtype=tf_floatx())
        yield inputs, output


@tf.function
def optimize_nnets_and_hpars(y_input, y_target, optimizer, gm: VAE, initial_state=None,
                             effective_nb_timesteps=None,
                             kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx())):
    vars = (
            gm.encoder.trainable_variables +
            list(gm.decoder.trainable_variables) +
            list(gm.sde_model.hyperpars)
    )
    with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
        tape.watch(vars)
        (samples, entropies, sampling_dist), encoded_dist, decoded_dist, final_state = gm._encode_and_decode(
            y_input, training=True, initial_state=initial_state
        )
        bloss = gm._breaked_loss(
            y_input, y_target, samples, entropies, encoded_dist, decoded_dist, initial_state,
            effective_nb_timesteps, kl_weight
        )
        loss = tf.reduce_sum(bloss)
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))
    return loss, final_state


# @tf.function
# def optimize_sde_hpars(y_input, y_target, hpars_optimizer, gm, initial_state=None,
#                        effective_nb_timesteps=None,
#                        kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx()), clip_value=100.):
#     hpars = list(gm.sde_model.hyperpars)
#     (samples, entropies, _), encoded_dist, decoded_dist, final_state, _ = gm._encode_and_decode(
#         y_input, training=True, initial_state=initial_state
#     )
#     with tf.GradientTape() as tape:
#         tape.watch(hpars)
#         breaked_loss = gm._breaked_loss(
#             y_target, samples, entropies, encoded_dist, decoded_dist, initial_state, effective_nb_timesteps, kl_weight
#         )
#         loss = tf.reduce_sum(breaked_loss)
#     hgrads = tape.gradient(loss, hpars)
#     hpars_optimizer.apply_gradients(zip(hgrads, hpars))
#     return loss, breaked_loss, final_state



@tf.function
def optimize_sde_standard_grad(y_input, y_target, gm:VAE, optimizer, initial_state=None,
                              effective_nb_timesteps=None,
                              kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx()),
                              clip_value=100.):
   vvars = gm.sde_model.variational_variables
   (samples, entropies, _), encoded_dist, decoded_dist, final_state = gm._encode_and_decode(
       y_input, training=False, initial_state=initial_state
   )
   with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape:
       tape.watch(vvars)
       breaked_loss = gm._breaked_loss(
           y_input, y_target, samples, entropies, encoded_dist, decoded_dist,
           initial_state, effective_nb_timesteps, kl_weight
       )
       loss = tf.reduce_sum(breaked_loss)
       # loss = gm.loss(y_input, y_target, training=True)
   vgrads = tape.gradient(loss, vvars)
   optimizer.apply_gradients(zip(vgrads, vvars))
   return loss, breaked_loss, final_state
#


@tf.function
def optimize_sde_with_nat_grad(y_input, y_target, gm,
                               gamma=1.0, initial_state=None, effective_nb_timesteps=None,
                               kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx()),
                               clip_value=None):
    sde_nat_grads, breaked_loss, loss, final_state = gm.nat_grads(
        y_input, y_target, training=False, initial_state=initial_state,
        effective_nb_timesteps=effective_nb_timesteps,
        kl_weight=kl_weight
    )
    if clip_value:
        sde_nat_grads = [tf.clip_by_value(nat_grad, -clip_value, clip_value) for nat_grad in sde_nat_grads]

    thetas = SdeModel.standard_to_natural_params(
        [gm.sde_model.diffusion._alphas, gm.sde_model.diffusion._betas,
         gm.sde_model.drift_svgp.q_mu, gm.sde_model.drift_svgp.q_sqrt]
    )
    new_xis = SdeModel.natural_to_standard_params(
        (thetas[0] - gamma * sde_nat_grads[0],
         thetas[1] - gamma * sde_nat_grads[1],
         thetas[2] - gamma * sde_nat_grads[2],
         thetas[3] - gamma * sde_nat_grads[3])
    )
    gm.sde_model.diffusion._alphas.assign(new_xis[0])
    gm.sde_model.diffusion._betas.assign(new_xis[1])
    gm.sde_model.drift_svgp.q_mu.assign(new_xis[2])
    gm.sde_model.drift_svgp.q_sqrt.assign(new_xis[3])

    return loss, breaked_loss, final_state


def get_breaked_loss(y_input, y_target, gm,
                     gamma=1.0, initial_state=None, effective_nb_timesteps=None,
                     kl_weight=tf.convert_to_tensor(1.0, dtype=tf_floatx()),
                     clip_value=None):
    sde_nat_grads, breaked_loss, loss, final_state = gm.nat_grads(
        y_input, y_target, training=False, initial_state=initial_state,
        effective_nb_timesteps=effective_nb_timesteps,
        kl_weight=kl_weight
    )
    return loss, breaked_loss, final_state


def beep():
    duration = 1  # seconds
    freq = 440  # Hz
    os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))


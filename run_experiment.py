from callbacks import Plot2DPhaseSpace, CallbackList
from GraphicalModel import GraphicalModel
import numpy as np
from nnets.optimizers import Momentum
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import tensorflow as tf
import pickle
from time import time
from utils.plot_utils import plot_2d_drift
from utils.tf_utils import tf_floatx, floatx, get_session, tf_get_value, tf_set_value
from utils.train.Monitor import Monitor
from utils.train.train_utils import prepare_training_dirs, create_epoch_gen
from utils.generic_utils import kmeans_based_inducing_points, grid_based_inducing_points, expand_range

# Load the experiment
from experiments.lotka_volterra.experiment import experiment

# ======================= Config Section ======================= #
do_run = True
SAVE_WHOLE_PATIENCE = 5                               # TODO
epochs_since_last_save = 0
initialized_ips = False
do_save = False
PICKLED_PP = 'data/tmp/pickled_pp/'                     # TODO
tensorboard_directory = 'data/tmp/tensorboard_' + experiment.ts_id
pretrain_path = 'data/pretraining/' + experiment.ts_id
save_path = 'data/tmp/best_' + experiment.ts_id
restore = False
mean_pretrain_epochs = 200
cov_pretrain_epochs = 1000
drift_pretrain_epochs = 200
hp_patience = 5
grid_tuning_patience = 0  # Move to experiment!!
ips_patience = 15
hp_iterations = 1
vars_target = np.repeat(5e-2, 2)  # target for pretarining the covariance encoder
do_pretraining = False  # Precedence over load_pretrained_model
load_pretrained_model = False
do_plot = False
select_inducing_points = lambda x, y: grid_based_inducing_points(x,y) #kmeans_based_inducing_points(x, y,)  # choose between kmeans_based... and grid_based...
# ============================================================== #
if do_run:
    prepare_training_dirs(tensorboard_directory, save_path)
pickle.dump(experiment, open('data/tmp/experiment_' + experiment.ts_id, 'wb'))

inputs = tf.placeholder(tf_floatx(), [None, experiment.len_tbptt, experiment.input_dimension], name='inputs')
training = tf.placeholder(tf.bool, [], name='training_phase')
sde_inputs = tf.placeholder(tf_floatx(), [None, experiment.embedding_dim], name='sde_inputs')
input_samples = tf.placeholder(tf_floatx(),
                               [experiment.batch_size, experiment.nb_samples, experiment.len_tbptt,
                                experiment.embedding_dim], name="input_samples")

# los valores numericos
initial_means = tf.placeholder(tf_floatx(), [experiment.batch_size, experiment.embedding_dim],
                               name="initial_means")
initial_precs = tf.placeholder(tf_floatx(), [experiment.batch_size, experiment.embedding_dim, experiment.embedding_dim],
                               name="input_precs")
use_initial_state = tf.placeholder(tf.bool, [], 'use_initial_state')

# h: hidden, o: output, m: mean, c:cov, is:initial_state
h_is_m = tf.placeholder(tf_floatx(), [experiment.batch_size, experiment.encoding_hidden_units],
                        name='hidden_initial_state_mean')
o_is_m = tf.placeholder(tf_floatx(), [experiment.batch_size, experiment.encoding_hidden_units],
                        name='output_initial_state_mean')
h_is_c = tf.placeholder(tf_floatx(), [experiment.batch_size, experiment.encoding_hidden_units],
                        name='hidden_initial_state_cov')
o_is_c = tf.placeholder(tf_floatx(), [experiment.batch_size, experiment.encoding_hidden_units],
                        name='output_initial_state_cov')
rnn_initial_states = [h_is_m, o_is_m, h_is_c, o_is_c]
rnn_initial_states_values = [
    np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
    np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
                    np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
    np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx())
]


with tf.variable_scope("optimizers"):
    sde_optimizer = Momentum(learning_rate=1e-4, clip_value=1, use_nesterov=True, momentum=0.9, name='sde_optimizer')
    nnets_optimizer = Momentum(learning_rate=1e-4, clip_value=1, use_nesterov=True, momentum=0.9,
                               name='nnets_optimizer')
    # nnets_optimizer = Adam(learning_rate=1e-4, clip_value=10, name='nnets_optimizer')
    hp_optimizer = Momentum(learning_rate=1e-4, clip_value=10, use_nesterov=True, momentum=0.9,
                            name='hp_optimizer')
    # hp_optimizer = Adam(learning_rate=1e-3, clip_value=10, name='hp_optimizer')

model = GraphicalModel(inputs, training, sde_inputs, input_samples, initial_means, initial_precs, use_initial_state,
                       rnn_initial_states, experiment, [sde_optimizer, nnets_optimizer, hp_optimizer])

# save_best = SaveModel(saver, 'data/results/' + experiment.ts_id + '_training', verbose=experiment.verbose, period=5)
plot_embedding = Plot2DPhaseSpace(model, experiment, 'data/tmp/' + experiment.ts_id, period=10,
                                  save_path="data/tmp/pickled")
# callbacks = CallbackList([plot_embedding, save_best])
# callbacks = CallbackList([plot_embedding])
callbacks = CallbackList([])


def create_pretrain_saver():
    length_scales_names=[kernel.length_scales.name for kernel in model.sde.kernels]
    save_vars = []
    for name in length_scales_names:
        save_vars += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, name)
    return tf.train.Saver(
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "encoders") +
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "decoders") +
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "sde/psi_2:0") +
        tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "sde/inducing_points:0") +
        save_vars
    )


def __expand(m, M, expansion_factor=1.05):
    default_range = (-1.5, 1.5)
    lower_bound, upper_bound = expand_range(m, M, expansion_factor)
    return min(lower_bound, default_range[0]), max(upper_bound, default_range[1])


def _update_limits(inducing_points, data, n_points=1000):
    full_data = np.concatenate([inducing_points, data], axis=0)
    ranges = [__expand(np.min(full_data[:, dim_it]), np.max(full_data[:, dim_it]))
              for dim_it in range(experiment.embedding_dim)]
    X = [np.linspace(min_X, max_X, int(n_points ** (1 / experiment.embedding_dim)) + 1) for min_X, max_X in ranges]
    X = np.meshgrid(*X)
    return X


def run_drift_predictions(X):
    data = np.concatenate([X[i].flatten().reshape((-1, 1)) for i in range(experiment.embedding_dim)], axis=1)
    drift_predictions = sess.run(
        model.sde.output, {
            model.sde.input: data
        }
    )
    return data, drift_predictions

def compute_complete_embedding(experiment):
    print("\t > Computing complete embedding ...")
    rnn_initial_states_values = [
        np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
        np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
        np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
        np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx())
    ]

    training_state_means = np.zeros(
        (experiment.batch_size, experiment.embedding_dim)).astype(floatx())
    training_state_precs = np.inf * np.ones(
        (experiment.batch_size, experiment.embedding_dim, experiment.embedding_dim)
    ).astype(floatx())

    encoded_means = []
    encoded_covs = []
    decoded_means = []
    decoded_covs = []
    smoothed_distributions = []
    samples = []

    for _, epoch in enumerate(
            create_epoch_gen(experiment.lag_embedded_y, nb_epochs=1, batch_size=experiment.batch_size,
                             len_tbptt=experiment.len_tbptt)):
        for batch_it, batch in enumerate(epoch):
            ##TODO
            if experiment.ts_id == "random_walk":
                batch_aux = []
                for b in batch:
                    batch_aux.append(b - b[0])
                batch = np.stack(batch_aux)
            else:
                print(experiment.ts_id)
            print("----> {}".format(batch_it))
            encoded_mean, encoded_cov, decoded_mean, decoded_cov, backward_pass, output_states, samp = (
                sess.run([model.mean_encoder.output, model.cov_encoder.output,
                          model.mean_decoder.output, model.cov_decoder.output,
                          model.inference.backward_pass,
                          model.rnn_output_states,
                          model.samples
                ], {
                    model.mean_encoder.input: batch,
                    training: False,
                    use_initial_state: (batch_it != 0),
                    ## RNN things,
                    initial_means: training_state_means,
                    initial_precs: training_state_precs,
                    model.rnn_initial_states[0]: rnn_initial_states_values[0],
                    model.rnn_initial_states[1]: rnn_initial_states_values[1],
                    model.rnn_initial_states[2]: rnn_initial_states_values[2],
                    model.rnn_initial_states[3]: rnn_initial_states_values[3]
                })
            )
            encoded_means.append(encoded_mean)
            encoded_covs.append(encoded_cov)
            decoded_means.append(decoded_mean)
            decoded_covs.append(decoded_cov)
            smoothed_distributions.append((backward_pass[2], backward_pass[3]))
            samples.append(samp)

            training_state_means = np.mean(backward_pass[2][:, -1, :, :], axis=1)
            training_state_precs = np.linalg.inv(backward_pass[3][:, -1, :, :])
            rnn_initial_states_values = output_states

    # use only first samples (index 0)
    smoothed_means = [sdist[0][:, :, 0, :] for sdist in smoothed_distributions]
    smoothed_covs = [sdist[1] for sdist in smoothed_distributions]
    print("\t > Finished!")
    return [
        np.concatenate(encoded_means, 1), np.concatenate(encoded_covs, 1),
        np.concatenate(decoded_means, 2)[:, 0, :, :], np.concatenate(decoded_covs, 2)[:, 0, :, :],
        (np.concatenate(smoothed_means, 1), np.concatenate(smoothed_covs, 1)),
        np.concatenate(samples, 2)[:, 0, :, :]
    ]

if do_run:
    data = []
    for _, epoch in enumerate(
                create_epoch_gen(experiment.lag_embedded_y, nb_epochs=1,
                                 batch_size=experiment.batch_size,
                                 len_tbptt=experiment.len_tbptt)):
            for batch_it, batch in enumerate(epoch):
                data.append(batch)

    all_data = np.concatenate(data, axis=1)
    pickle.dump(all_data, open(os.path.join(PICKLED_PP, experiment.ts_id + '_all_data.pkl'), 'wb'))


    sess = get_session()
    sess.run(tf.global_variables_initializer())
    if load_pretrained_model and os.path.exists(pretrain_path + '.meta'):
        print("===== Loading pretrained model =====")
        pretrain_saver = create_pretrain_saver()
        pretrain_saver.restore(sess, pretrain_path)
    else:
        embedding, _, _, _, _, _ = compute_complete_embedding(experiment)
        embedding = embedding.reshape((-1, experiment.embedding_dim))
        inducing_points = select_inducing_points(embedding, model.sde.nb_inducing_points)
        tf_set_value(model.sde.inducing_points, inducing_points.astype(floatx()))

        with PdfPages("/tmp/initial_conditions.pdf") as pdf:
            plt.figure()
            plt.plot(embedding[:, 0], embedding[:, 1])
            plt.scatter(inducing_points[:, 0], inducing_points[:, 1], color='orange')
            pdf.savefig()

            y = embedding[:-1]
            dy = np.diff(embedding, axis=0)
            drifts = sess.run(model.sde.output, {model.sde.input: y})
            plot_2d_drift(y, dy, drifts)
            pdf.savefig()

    tfsaver = tf.train.Saver()
    tensorboard_periodicity = 1

    if tensorboard_directory is not None:
        tensorboard_summary = tf.summary.merge_all()
        tensorboard_writer = tf.summary.FileWriter(tensorboard_directory, sess.graph)

    monitor = Monitor()
    monitor.session = sess
    callbacks.set_monitor(monitor)
    callbacks.on_train_begin()

    # previous_coarse_grained_embedding = compute_embedding(experiment.coarse_grained_data)

    previous_inducing_points = tf_get_value(model.sde.inducing_points)
    global_iteration = 0
    try:
        print("Starting to train")
        for epoch_it, epoch in enumerate(
                create_epoch_gen(experiment.lag_embedded_y, experiment.nb_epochs, experiment.batch_size,
                                 experiment.len_tbptt)):
            lower_bounds = np.zeros(2)
            training_state_means = np.zeros(
                (experiment.batch_size, experiment.embedding_dim)).astype(floatx())
            training_state_precs = np.inf * np.ones(
                (experiment.batch_size, experiment.embedding_dim, experiment.embedding_dim)
            ).astype(floatx())

            rnn_initial_states_values = [
                np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
                np.zeros([experiment.batch_size, experiment.embedding_dim]).astype(floatx()),
                                np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
                np.zeros([experiment.batch_size, experiment.embedding_dim]).astype(floatx())
            ]
            rnn_initial_states_values = [
                np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
                np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
                np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx()),
                np.zeros([experiment.batch_size, experiment.encoding_hidden_units]).astype(floatx())
            ]
            for batch_it, y_batch in enumerate(epoch):
                is_not_first_iteration = (batch_it != 0)
                # In the first iteration the initial state has infinite variance, or equivanlently, we don't use
                # the infor from the training_state. Next iterartions of the batch do use the training_state
                callbacks.on_batch_begin()
                start = time()
                inferred_samples, backward_pass, next_rnn_initial_states_values = sess.run(
                    [model.samples, model.inference.backward_pass, model.rnn_output_states],
                    {
                        model.inputs: y_batch, training: False,
                        use_initial_state: is_not_first_iteration,
                        initial_means: training_state_means,
                        initial_precs: training_state_precs,
                        rnn_initial_states[0]: rnn_initial_states_values[0],
                        rnn_initial_states[1]: rnn_initial_states_values[1],
                        rnn_initial_states[2]: rnn_initial_states_values[2],
                        rnn_initial_states[3]: rnn_initial_states_values[3]
                    }
                )
                print("Done with inferring samples")
                smoothed_distributions = (backward_pass[2], backward_pass[3])

                if global_iteration >= ips_patience:
                    if not initialized_ips:
                        print("****** SETTING IPS VALUES: {}, {} *******".format(epoch_it, batch_it))
                        previous_coarse_grained_embedding, _, _, _, _, _ = compute_complete_embedding(experiment)
                        previous_coarse_grained_embedding = previous_coarse_grained_embedding.reshape((-1, experiment.embedding_dim))
                        inducing_points = select_inducing_points(previous_coarse_grained_embedding,
                                                                 model.sde.nb_inducing_points)
                        tf_set_value(model.sde.inducing_points, inducing_points.astype(floatx()))
                        initialized_ips = True
                        do_save = True

                        # from mpl_toolkits.mplot3d import Axes3D
                        # fig = plt.figure()
                        # ax = fig.add_subplot(111, projection='3d')
                        # ax.scatter(*tf_get_value(model.sde.inducing_points).T, color="red")
                        # ax.plot(*previous_coarse_grained_embedding.T)


                    sde_lower_bound, _ = sess.run([model.sde_lower_bound, model.optimize_sde_distribution], {
                        model.inferred_samples: inferred_samples,
                        training: True,
                        use_initial_state: is_not_first_iteration,
                        initial_means: training_state_means,
                        initial_precs: training_state_precs,
                        rnn_initial_states[0]: rnn_initial_states_values[0],
                        rnn_initial_states[1]: rnn_initial_states_values[1],
                        rnn_initial_states[2]: rnn_initial_states_values[2],
                        rnn_initial_states[3]: rnn_initial_states_values[3]
                    })
                    print("Done with sde_lower_bound")
                else:
                    sde_lower_bound=-np.inf

                if global_iteration % tensorboard_periodicity == 0 and global_iteration > 0:
                    [nnet_lower_bound, _, summary] = sess.run([
                        model.nnet_lower_bound,
                        model.optimize_nnets,
                        tensorboard_summary
                    ], {
                        model.inputs: y_batch,
                        model.inferred_samples: inferred_samples,
                        training: True,
                        use_initial_state: is_not_first_iteration,
                        initial_means: training_state_means,
                        initial_precs: training_state_precs,
                        rnn_initial_states[0]: rnn_initial_states_values[0],
                        rnn_initial_states[1]: rnn_initial_states_values[1],
                        rnn_initial_states[2]: rnn_initial_states_values[2],
                        rnn_initial_states[3]: rnn_initial_states_values[3]
                    })
                    tensorboard_writer.add_summary(summary, global_iteration)
                else:
                    [nnet_lower_bound, _] = sess.run([
                        model.nnet_lower_bound,
                        model.optimize_nnets
                    ], {
                        model.inputs: y_batch,
                        model.inferred_samples: inferred_samples,
                        training: True,
                        use_initial_state: is_not_first_iteration,
                        initial_means: training_state_means,
                        initial_precs: training_state_precs,
                        rnn_initial_states[0]: rnn_initial_states_values[0],
                        rnn_initial_states[1]: rnn_initial_states_values[1],
                        rnn_initial_states[2]: rnn_initial_states_values[2],
                        rnn_initial_states[3]: rnn_initial_states_values[3]
                    })
                print("Done with nnet optimization")

                for _ in range(hp_iterations):
                    sess.run(model.optimize_hps, {model.inferred_samples: inferred_samples})
                    print("Done with hp optimization")

                global_iteration += 1
                lower_bounds += np.array([sde_lower_bound, nnet_lower_bound])

                epochs_since_last_save += 1
                if epochs_since_last_save >= SAVE_WHOLE_PATIENCE or do_save:
                    print("******* SAVING ALL DATA *******")
                    do_save = False
                    complete_info = compute_complete_embedding(experiment)
                    previous_coarse_grained_embedding = complete_info[0].reshape((-1, experiment.embedding_dim))
                    ips = tf_get_value(model.sde.inducing_points)
                    complete_info = complete_info + [ips]
                    if experiment.embedding_dim == 2 or experiment.embedding_dim == 3:
                        data_for_limits = np.concatenate([
                            complete_info[0].reshape((-1, experiment.embedding_dim)),
                            complete_info[-1].reshape((-1, experiment.embedding_dim))
                        ], axis=0)
                        XXX = _update_limits(ips, data_for_limits)
                        XXXp, drifts = run_drift_predictions(XXX)
                        complete_info = complete_info + [XXX, drifts, XXXp]

                    save_this_name = experiment.ts_id + "_" + str(epoch_it) + "_" + str(batch_it)
                    pickle.dump(complete_info, open(
                        os.path.join(PICKLED_PP, save_this_name + ".pkl"),
                        "wb"
                    ))
                    saved_tooo = tfsaver.save(sess, os.path.join(save_path, save_this_name + ".ckpt"))
                    print("Model saved to %s" % saved_tooo)
                    epochs_since_last_save = 0

                if callbacks.len() > 0:
                    monitor.update(epoch_it, batch_it, y_batch, sde_lower_bound, nnet_lower_bound, inferred_samples,
                                   smoothed_distributions, training_state_means, training_state_precs,
                                   is_not_first_iteration, rnn_initial_states_values)

                # Get the new training state from the last temporal instant and by averaging out the samples in the
                # case of the mean (the covs are all the same for all the samples)
                training_state_means = np.mean(backward_pass[2][:, -1, :, :], axis=1)
                training_state_covs = backward_pass[3][:, -1, :, :]
                training_state_precs = np.linalg.inv(backward_pass[3][:, -1, :, :])
                rnn_initial_states_values = next_rnn_initial_states_values

                print("\tEpoch {}, Batch {}: SDE = {}, NNET = {}".format(epoch_it, batch_it, sde_lower_bound,
                                                                         nnet_lower_bound))

                print("Calling callbacks")
                callbacks.on_batch_end()
                if monitor.stop_training:
                    print("Stop training")
                    break
                print("End of callbacks")
            elapsed_time = time() - start
            lower_bounds /= (batch_it + 1)
            print("Epoch {}: SDE = {}, NNET = {} (Elapsed {} seconds)".format(epoch_it, sde_lower_bound,
                                                                              nnet_lower_bound, elapsed_time))
            # print(tf_get_value(model.sde.kernels[0].length_scales))
            # print(tf_get_value(model.sde.kernels[1].length_scales))
            # print(tf_get_value(model.sde.standard_params['alphas']))
            # print(tf_get_value(model.sde.standard_params['betas']))
            # print(tf_get_value(model.sde.standard_params['betas'] / tf_get_value(model.sde.standard_params['alphas'])))
            # print(tf_get_value(model.sde.standard_params['means']))
            # print(tf_get_value(model.sde.standard_params['covs']))
            # print('---------------------------------------')
            callbacks.on_epoch_end()
            if monitor.stop_training:
                print("Stop training")
                break

    except Exception as e:
        print(e)
        print("Unexpected error")
        raise e

    finally:
        # callbacks.on_train_end()
        if tensorboard_directory is not None:
            tensorboard_writer.close()

from experiments.Experiment import Experiment
_generate_data = False

if _generate_data:
    import os
    # run sim_oscillator.R
    from utils.data import generate_tf_records_from_csv
    for mode in ['train', 'test']:
        csv_folder = os.path.join('data/oscillator/csv', mode)
        tfrecords_folder = os.path.join('data/oscillator', mode)
        generate_tf_records_from_csv(csv_folder, tfrecords_folder)

experiment = Experiment(
    tfrecords_folder='data/oscillator',
    example_timesteps=70,
    len_tbptt=70,
    nb_epochs=1000,
    batch_size=8,
    buffer_size=128,
    tensorboard_dir='runs/tensorboard/oscillator',
    checkpoint_dir='runs/ckpt/oscillator/',
    scaling_function=None
)

svae_settings = {
    'encoder_hidden_units': [64],
    'encoder_type': 'rnn',
    'encoder_kernel_size': 31,
    'encoder_dilation_rate': 1,
    'phase_space_dim': 2,
    'nb_pseudo_inputs': 2 ** 4,
    'nb_samples': 10
}


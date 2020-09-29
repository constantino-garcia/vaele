import os
import numpy as np
from utils.tf_utils import floatx


# Adapted from https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
def create_batch_gen(raw_data, batch_size, len_tbptt):
    """
    From the embedded time series raw_data, we create a Batch for training shaped as
    [batch_size, len_tbptt, embedding_dimension]
    If our lag_embedded_y is a bidimensional series (1, 2), (3, 4), (5, 6), ..., (19, 20)
    len_tbptt = 3 and batch_size = 2, we rearrange raw_data as data:
    (1, 2),   (3, 4),   (5, 6),  (7, 8), (9, 10)
    (11, 12), (13, 14), (15,16), ...,     (19, 20)
    etc.
    For a single train step, we return data[:, i * len_tbptt : (i + 1) * len_tbptt]
    A whole epoch exhausts the columns of data

    :param raw_data: A m-dimensional time series of length T arranged as a matrix (T, m).
    :param batch_size: Number of batches in the training step.
    :param len_tbptt: Number of time steps before truncating the gradient (tbptt: Truncated Backprop through time).
    :return: An iterator that returns a single batch for the training procedure.
    """
    data_length, dimension = raw_data.shape
    # partition raw data into batches and stack them vertically in a data matrix
    batch_partition_length = data_length // batch_size
    if batch_partition_length < len_tbptt:
        raise ValueError('batch_partition_len < len_tbptt')
    data = np.zeros([batch_size, batch_partition_length, dimension], dtype=floatx())
    for i in range(batch_size):
        data[i] = raw_data[batch_partition_length * i:batch_partition_length * (i + 1), :]
    # further divide batch partitions into num_steps for truncated backpropagation
    epoch_size = batch_partition_length // len_tbptt
    for i in range(epoch_size):
        # There is no need to check bounds since epoch_size was calculated with //
        yield data[:, i * len_tbptt:(i + 1) * len_tbptt, :]


def create_epoch_gen(raw_data, nb_epochs, batch_size, len_tbptt):
    for i in range(nb_epochs):
        yield create_batch_gen(raw_data, batch_size, len_tbptt)


# TODO: we should use to generators as in https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html
def get_epoch_size(raw_data, batch_size, len_tbptt):
    data_length, dimension = raw_data.shape
    batch_partition_length = data_length // batch_size
    if batch_partition_length < len_tbptt:
        raise ValueError('batch_partition_len < len_tbptt')
    epoch_size = batch_partition_length // len_tbptt
    return epoch_size


def prepare_training_dirs(tensorboard_directory, save_path, do_run):
    if not do_run:
        if os.path.exists(tensorboard_directory):
            import shutil
            shutil.rmtree(tensorboard_directory)
        if os.path.exists(save_path):
            import shutil
            shutil.rmtree(save_path)
        os.makedirs(save_path)
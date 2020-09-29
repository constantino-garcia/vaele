from vaele_config import tf_floatx
import numpy as np
import tensorflow as tf
import glob
import os
import shutil
import warnings

from utils.mutual_information import estimate_lag_value
from utils.train import tbptt_chunks_generator


class Experiment(object):
    def __init__(self, *, tfrecords_folder, example_timesteps, len_tbptt, nb_epochs, batch_size, buffer_size,
                 time_lag=None, nb_target_outputs=1, tensorboard_dir=None, checkpoint_dir=None, scaling_function=None):
        """
        :param tfrecords_folder: the folder containing all the TFRecords with .tfrecords extension
        :param example_timesteps: The length of each of the time series.
        :param len_tbptt: Length of the Truncated Backpropagation Through Time. That is, the length of each of the
        chunks in which each of the time series examples should be split.
        :param nb_epochs: number of epochs for training the svae.
        :param batch_size: the batch size, to create a batched tf.data.Dataset.
        :param buffer_size: the buffer_size for buffering, to create a shuffled tf.data.Dataset. Shuffling
        is applied before batching
        :param time_lag: Lag to be used for generating lagged versions of the target signal, to be used as
        inputs.
        :param nb_target_ouputs: Controls the number of different time series that the SVAE will try to predict.
        The SVAE will try to predict y(t), y(t + time_lag), ..., y(t + (nb_target_outputs - 1) * time_lag. Using several
        targets may enhance to build a phase space with more dynamical information.
        :param scaling_function: a scaling function to be applied to the dataset with the aim of normalizing
        it.
        """
        self.experiment_name = os.path.basename(tfrecords_folder)
        self.tfrecords_folder = tfrecords_folder
        self.example_timesteps = example_timesteps
        self.len_tbptt = len_tbptt
        self.time_lag = time_lag
        self.nb_target_ouputs = nb_target_outputs
        self.nb_epochs = nb_epochs
        self.batch_size = batch_size
        self.tensorboard_dir = tensorboard_dir
        self.checkpoint_dir = checkpoint_dir

        train_dir = os.path.join(tfrecords_folder, 'train')
        test_dir = os.path.join(tfrecords_folder, 'test')

        self.len_train_data = 0
        if os.path.exists(train_dir):
            self.train_dataset = self._parse_dataset(train_dir, scaling_function, batch_size, buffer_size)
            assert self.train_dataset is not None, "Empty train folder ({})".format(train_dir)
            self.len_train_data = 0
            for _ in self.train_dataset:
                self.len_train_data += 1
        else:
            raise RuntimeError('{} folder does not exist'.format(train_dir))

        self.effective_len = self.len_train_data * self.batch_size * example_timesteps
        self.effective_len = tf.convert_to_tensor(self.effective_len, dtype=tf_floatx())

        self.len_test_data = 0
        if os.path.exists(test_dir):
            # last param is 0 to avoid shuffling
            self.test_dataset = self._parse_dataset(test_dir, scaling_function, batch_size, 0)
            for _ in self.test_dataset:
                self.len_test_data += 1

        self.has_test = True if self.test_dataset is not None else False

        # Now that data is available use it to estimate the time_lag (if needed)
        if self.time_lag is None:
            y = next(iter(self.train_dataset.take(1)))
            lags = [estimate_lag_value(y_batch.numpy().flatten()) for y_batch in y]
            if np.all(np.isnan(lags)):
                raise RuntimeError("Could not find a proper lag value")
            else:
                self.time_lag = int(np.round(np.nanmedian(lags)))
                print(f"Estimated time lag is {self.time_lag}")

        # Make sure that both the tensorboard directory and the checkpoint folder are empty
        if self.tensorboard_dir is not None and os.path.exists(self.tensorboard_dir):
            shutil.rmtree(self.tensorboard_dir)

        if self.checkpoint_dir is not None:
            ckpt_files = glob.glob(os.path.join(self.checkpoint_dir, "*"))
            for ckpt_file in ckpt_files:
                os.remove(ckpt_file)

    def _parse_dataset(self, folder, scaling_function, batch_size, shuff_buffer_size=0):
        """creates a batched Dataset object. If shuff_buffer_size > 0, shuffling is used"""
        files = glob.glob(os.path.join(folder, "*.tfrecords"))
        nb_files = len(files)
        if nb_files >= batch_size:
            tfdata = tf.data.TFRecordDataset(files)
            # parse the protobuff to a dictionary with the 'signal' feature
            parsed_dataset = tfdata.map(
                lambda proto: tf.io.parse_single_example(proto, {
                    'signal': tf.io.FixedLenFeature([self.example_timesteps], tf.float32,
                                                    default_value=np.zeros(self.example_timesteps))
                })
            )
            # Extract the 'signal' feature from the dictionary to get a tensor representing the signal as
            # a vector of length (len_tbptt, ). We add an extra dimension for consistency with the expected
            # input of RNNs, in which the latest dimension is the dimension of the feature space.
            parsed_dataset = parsed_dataset.map(
                lambda x: x['signal'][..., tf.newaxis],
                num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
            if tf_floatx() != tf.float32:
                parsed_dataset = parsed_dataset.map(
                    lambda x: tf.cast(x, tf_floatx()),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
            if scaling_function is not None:
                parsed_dataset = parsed_dataset.map(
                    lambda x: scaling_function(x),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE
                )
            parsed_dataset = parsed_dataset.cache() #TODO
            if shuff_buffer_size > 0:
                parsed_dataset = parsed_dataset.shuffle(shuff_buffer_size)
            parsed_dataset = parsed_dataset.batch(batch_size, drop_remainder=True)
            parsed_dataset = parsed_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE) # TODO
        else:
            warnings.warn(f"Insufficient number of files in {folder} to create batches of size {batch_size} (returning None)")
            parsed_dataset = None
        return parsed_dataset

    def collect_examples(self, nb_examples=None, train=True):
        collected_examples = 0
        examples = []
        dataset = self.train_dataset if train else self.test_dataset
        if dataset:
            for y in dataset:
                examples.append(y)
                collected_examples += y.shape[0]
                if nb_examples and collected_examples > nb_examples:
                    break
            examples = tf.concat(examples, axis=0)
            return examples[:nb_examples] if nb_examples else examples
        else:
            warnings.warn('Dataset not available')
            return None

    def collect_tbptt_examples(self, nb_examples=None, train=True):
        collected_examples = 0
        examples = []
        dataset = self.train_dataset if train else self.test_dataset
        if dataset:
            for yt in dataset:
                # TODO
                for y, _ in tbptt_chunks_generator(yt, self.len_tbptt, self.time_lag, 0, 0):
                    examples.append(y)
                collected_examples += yt.shape[0]
                if nb_examples and collected_examples > nb_examples:
                    break
            examples = tf.concat(examples, axis=0)
            return examples[:nb_examples] if nb_examples else examples
        else:
            warnings.warn('Dataset not available')
            return None
import csv
import glob
import numpy as np
import os
import pandas as pd
import tensorflow as tf


def generate_csv_chunks(signal_len, original_file, destination_folder):
    """
    WARNING! This removes the current contents of destination_folder
    :param signal_len:
    :param destination_folder:
    :param original_file:
    :return:
    """
    filelist = glob.glob(os.path.join(destination_folder, "*"))
    # Ignore directories
    filelist = [file for file in filelist if os.path.isfile(file)]
    for f in filelist:
        os.remove(f)
    record_number = 0
    data = []
    current_len = 0
    os.getcwd()
    with open(original_file, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            # row is just a list of one element
            data.append(float(row[0]))
            current_len += 1
            if current_len == signal_len:
                filename = os.path.join(destination_folder, 'y_{}.csv'.format(record_number))
                # print('Saving in ' + filename)
                np.savetxt(filename, np.array(data).reshape((1, -1)), delimiter=',')
                data = []
                record_number += 1
                current_len = 0


def csv_to_tfrecord(filename, tfrecords_folder):
    # This assumes that the file ends with .csv
    destination = os.path.join(tfrecords_folder, os.path.basename(filename).split(".")[0] + ".tfrecords")
    csv_file = pd.read_csv(filename, header=None).values
    with tf.io.TFRecordWriter(destination) as writer:
        example = tf.train.Example()
        for row in csv_file:
            # Similar to tf.train.Feature(float_list=tf.train.FloatList(value=row.reshape(-1)))
            example.features.feature["signal"].float_list.value.extend(row)
        writer.write(example.SerializeToString())


def generate_tf_records_from_csv(csv_folder, tfrecords_folder):
    filenames = glob.glob(os.path.join(csv_folder, '*.csv'), )
    print(filenames)
    nb_filenames = len(filenames)
    for i, csv_filename in enumerate(filenames):
        print(f"{i}/{nb_filenames}: processing {csv_filename}")
        csv_to_tfrecord(csv_filename, tfrecords_folder)


def remove_csvs(csv_folder):
    filenames = glob.glob(os.path.join(csv_folder, '*.csv'))
    for filename in filenames:
        os.remove(filename)

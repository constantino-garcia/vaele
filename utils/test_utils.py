import csv
import numpy as np
from utils.tf_utils import to_floatx


def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def read_file(filename, batch_size=1, delimiter=' ', random=True):
    """A generator function to reads a file."""
    batch_data = []
    filelen = file_len(filename)
    nb_batches = filelen // batch_size
    lines_counter = 0
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=delimiter)
        if random:
            batches = np.random.permutation(np.arange(nb_batches))
            while True:
                for batch in batches:
                    # print('seeking at', batch)
                    csvfile.seek(0)
                    position = 0
                    while position != batch * batch_size:
                        _ = next(reader)
                        position += 1
                    for row in reader:
                        batch_data.append(np.array([to_floatx(item) for item in row]))
                        lines_counter += 1
                        if lines_counter % batch_size == 0:
                            yield np.stack(batch_data)
                            batch_data = []
                            lines_counter = 0
                            break
                batches = np.random.permutation(np.arange(nb_batches))
                batch_data = []
                lines_counter = 0
        else:
            while True:
                for row in reader:
                    batch_data.append(np.array([to_floatx(item) for item in row]))
                    lines_counter += 1
                    if lines_counter % batch_size == 0:
                        yield np.stack(batch_data)
                        batch_data = []
                        lines_counter = 0
                csvfile.seek(0)
                batch_data = []
                lines_counter = 0
                # reset batch_data to avoid creating discontinuities
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import gpflow
from gpflow.config import default_float
from gpflow.config import set_default_positive_bijector
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as backend

set_default_positive_bijector('softplus')

_VAELE_DEFAULT_FLOAT_ = 'float64'
_VAELE_DEFAULT_JITTER_ = 1e-6  # 1e-6 is gpflow default jitter


def _translate_str_float_to_np(float_str):
    _STR_FLOAT_TO_NP_ = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}
    if float_str not in _STR_FLOAT_TO_NP_:
        raise ValueError('Invalid float string')
    return _STR_FLOAT_TO_NP_[float_str]


backend.set_floatx(_VAELE_DEFAULT_FLOAT_)
gpflow.config.set_default_float(_translate_str_float_to_np(_VAELE_DEFAULT_FLOAT_))
if _VAELE_DEFAULT_JITTER_:
    gpflow.config.set_default_jitter(_VAELE_DEFAULT_JITTER_)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Currently, memory growth needs to be the same across GPUs
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def floatx():
    return backend.floatx()


def np_floatx():
    return default_float()


def tf_floatx():
    return tf.as_dtype(np_floatx())

def vaele_jitter():
    return gpflow.default_jitter()


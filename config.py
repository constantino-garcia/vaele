import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import tensorflow as tf

_SESSION = None


def floatx():
    return 'float64'


def tf_floatx():
    return tf.as_dtype(floatx())


def get_session():
    global _SESSION
    default_session = tf.get_default_session()
    if default_session is not None:
        session = default_session
    else:
        if _SESSION is None:
            _SESSION = tf.Session()
        session = _SESSION
    return session




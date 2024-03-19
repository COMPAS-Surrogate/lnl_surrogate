import logging
import os
import sys


def silence_tensorflow():
    """Silence every unnecessary warning from tensorflow."""
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ["KMP_AFFINITY"] = "noverbose"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    # We wrap this inside a try-except block
    # because we do not want to be the one package
    # that crashes when TensorFlow is not installed
    # when we are the only package that requires it
    # in a given Jupyter Notebook, such as when the
    # package import is simply copy-pasted.
    try:
        import tensorflow as tf

        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(3)
    except ModuleNotFoundError:
        pass


silence_tensorflow()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "|%(asctime)s|%(name)s|%(levelname)s| %(message)s"
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

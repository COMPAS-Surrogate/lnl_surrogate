import logging
import os
import sys
import traceback
import warnings

from lnl_computer.logger import logger as lnl_computer_logger
from lnl_computer.logger import setup_logger

"""Silence every unnecessary warning from tensorflow."""
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
try:
    import tensorflow as tf

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(2)
except ModuleNotFoundError:
    pass

warnings.filterwarnings("ignore", category=RuntimeWarning)


logger = setup_logger("lnl_surrogate")


class Suppressor:
    def __init__(self, verbosity: int):
        self.verbosity = verbosity

    def __enter__(self):
        if self.verbosity == 0:
            self.stdout = sys.stdout
            sys.stdout = self

    def __exit__(self, exception_type, value, traceback):
        if self.verbosity == 0:
            sys.stdout = self.stdout
            if exception_type is not None:
                raise Exception(
                    f"Got exception: {exception_type} {value} {traceback}"
                )

    def write(self, x):
        pass

    def flush(self):
        pass


def set_log_verbosity(verbosity: int, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    logger.setLevel("ERROR")
    lnl_computer_logger.setLevel("ERROR")

    if verbosity > 0:
        logger.setLevel("INFO")

    if verbosity > 1:
        lnl_computer_logger.setLevel("INFO")

    if verbosity > 2:
        import tensorflow as tf
        from trieste.logging import set_tensorboard_writer

        logger.setLevel("DEBUG")
        lnl_computer_logger.setLevel("DEBUG")
        summary_writer = tf.summary.create_file_writer(outdir)
        set_tensorboard_writer(summary_writer)
        logger.debug(
            f"visualise optimization progress with `tensorboard --logdir={outdir}`"
        )

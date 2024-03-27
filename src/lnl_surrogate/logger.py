import logging
import os

from lnl_computer.logger import logger as lnl_computer_logger
from lnl_computer.logger import setup_logger

"""Silence every unnecessary warning from tensorflow."""
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
try:
    import tensorflow as tf

    tf.get_logger().setLevel("ERROR")
    tf.autograph.set_verbosity(3)
except ModuleNotFoundError:
    pass

logger = setup_logger(__name__, "LNL-SURROGATE")


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

import logging
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    "|%(asctime)s|%(name)s|%(levelname)s| %(message)s"
)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Module code
from ._version import __version__, __version_info__

__author__ = "Claudio Noguera"

# get the location of the

import logging

logger = logging.getLogger(__name__)

handler = logging.StreamHandler()
formatter = logging.Formatter(
    '%(asctime)s|%(levelname)s|%(name)s|%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

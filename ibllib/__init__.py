__version__ = "2.7.0"
import warnings

from ibllib.misc import logger_config

warnings.filterwarnings("always", category=DeprecationWarning, module="ibllib")

# if this becomes a full-blown library we should let the logging configuration to the discretion of the dev
# who uses the library. However since it can also be provided as an app, the end-users should be provided
# with an useful default logging in standard output without messing with the complex python logging system
# -*- coding:utf-8 -*-

import logging

USE_LOGGING = True
#%(asctime)s,%(msecs)d
if USE_LOGGING:
    logger_config(name="ibllib")
else:
    # deactivate all log calls for use as a library
    logging.getLogger("ibllib").addHandler(logging.NullHandler())

try:
    import one
except ModuleNotFoundError:
    logging.getLogger("ibllib").error("Missing dependency, please run `pip install ONE-api`")

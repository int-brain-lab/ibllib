# -*- coding: utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date:   2018-07-24 18:02:52
# @Last Modified by:   Niccolò Bonacchi
# @Last Modified time: 2018-07-26 17:54:37
from ibllib.dsp import savitzky_golay, smooth
from ibllib.io import alf, globus, one, params, raw_data_loaders
from ibllib.misc import flatten, misc, timing
import ibllib.time as time


# if this becomes a full-blown library we should let the logging configuration to the discretion of the dev
# who uses the library. However since it can also be provided as an app, the end-users should be provided
# with an useful default logging in standard output without messing with the complex python logging system
import logging
USE_LOGGING = True
if USE_LOGGING:
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%S')
    # add some colours for an easier log experience
    logging.addLevelName(logging.DEBUG, "\033[0;34m%s\033[0;0m" % logging.getLevelName(logging.DEBUG))
    logging.addLevelName(logging.INFO, "\033[0;37m%s\033[0;0m" % logging.getLevelName(logging.INFO))
    logging.addLevelName(logging.WARNING, "\033[0;33m%s\033[0;0m" % logging.getLevelName(logging.WARNING))
    logging.addLevelName(logging.ERROR, "\033[1;31m%s\033[1;0m" % logging.getLevelName(logging.ERROR))
    logging.addLevelName(logging.CRITICAL, "\033[1;35m%s\033[1;0m" % logging.getLevelName(logging.CRITICAL))
else:
    # deactivate all log calls for use as a library
    logging.getLogger('ibllib').addHandler(logging.NullHandler())

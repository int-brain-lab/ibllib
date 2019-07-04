#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date:   2018-07-24 18:02:52
from ibllib.dsp import savitzky_golay, smooth
from ibllib.io import globus, one, params, raw_data_loaders, flags
from ibllib.misc import flatten, misc, timing, logger_config
import ibllib.time as time


# if this becomes a full-blown library we should let the logging configuration to the discretion of the dev
# who uses the library. However since it can also be provided as an app, the end-users should be provided
# with an useful default logging in standard output without messing with the complex python logging system
# -*- coding:utf-8 -*-
from sys import platform
import logging
USE_LOGGING = True
#%(asctime)s,%(msecs)d
if USE_LOGGING:
    logger_config(name='ibllib')
else:
    # deactivate all log calls for use as a library
    logging.getLogger('ibllib').addHandler(logging.NullHandler())

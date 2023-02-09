"""Library implementing the International Brain Laboratory data pipeline."""
import logging
import warnings

__version__ = '2.21.0'
warnings.filterwarnings('always', category=DeprecationWarning, module='ibllib')

# if this becomes a full-blown library we should let the logging configuration to the discretion of the dev
# who uses the library. However since it can also be provided as an app, the end-users should be provided
# with an useful default logging in standard output without messing with the complex python logging system
# -*- coding:utf-8 -*-
USE_LOGGING = True
#%(asctime)s,%(msecs)d
if USE_LOGGING:
    try:  # TODO Remove after release of iblutil v1.4
        from iblutil.util import get_logger
        get_logger(name='ibllib')
        warnings.warn('Please run `pip install -U iblutil` to update to v1.4', category=DeprecationWarning)
    except ImportError:
        from iblutil.util import setup_logger
        setup_logger(name='ibllib')
else:
    # deactivate all log calls for use as a library
    logging.getLogger('ibllib').addHandler(logging.NullHandler())

import warnings
import traceback

for line in traceback.format_stack():
    print(line.strip())

warnings.warn('ibllib.ephys.neuropixel has moved and functionality will be removed'
              ', change your imports to neuropixel !', DeprecationWarning)

from neuropixel import *  # noqa

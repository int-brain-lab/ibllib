import warnings
import traceback

for line in traceback.format_stack():
    if 'ibllib' in line:
        print(line.strip())

warnings.warn('ibllib.io.spikeglx has moved and functionality will be removed'
              ', change your imports to spikeglx !', DeprecationWarning)

from spikeglx import *  # noqa

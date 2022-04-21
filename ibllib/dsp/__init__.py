import warnings
import traceback

for line in traceback.format_stack():
    print(line.strip())

warnings.warn('ibllib.dsp is deprecated and functionality will be removed'
              ', change your imports to neurodsp ! See stack above', DeprecationWarning)

from neurodsp.fourier import fscale, freduce, fexpand, lp, hp, bp, fshift, dephas, fit_phase
from neurodsp.utils import rms, WindowGenerator, rises, falls, fronts, fcn_cosine
from neurodsp.voltage import destripe

import neurodsp.utils as utils
import neurodsp.fourier as fourier
import neurodsp.voltage as voltage

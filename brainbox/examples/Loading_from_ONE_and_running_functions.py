'''
This script gives examples of how to load data from ONE and call brainbox functions.

*Note*: This module assumes that the required data for a particular eid is already saved in the
CACHE_DIR specified by `.one_params` (the default location to which ONE saves data when running the
`load` method). It is recommended to download *all* data for a particular eid, e.g.:
    `from oneibl.one import ONE`
    `one = ONE()`
    # get eid
    `eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]`
    # download data
    one.load(eid, dataset_types=one.list(), clobber=False, download_only=True)
'''

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from oneibl.one import ONE
import alf.io as aio
import brainbox as bb

# Set eid and probe name #
#------------------------#
one = ONE()
eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
probe = 'probe_right'  # *Note: new probe naming convention is 'probe00', 'probe01', etc.

# Get important directories from `eid` #
#--------------------------------------#
spikes_path = one.load(eid, dataset_types='spikes.amps', clobber=False, download_only=True)[0]
alf_dir_part = np.where([part == 'alf' for part in Path(spikes_path).parts])[0][0]
session_path = os.path.join(*Path(spikes_path).parts[:alf_dir_part])
alf_dir = os.path.join(session_path, 'alf')
alf_probe_dir = os.path.join(alf_dir, probe)
ephys_file_dir = os.path.join(session_path, 'raw_ephys_data', probe)
# Find 'ap' ephys file in `ephys_file_dir`
for i in os.listdir(ephys_file_dir):
        if 'ap' in i and 'bin' in i:
            ephys_file_path = os.path.join(ephys_file_dir, i)
# Ensure directories and paths can be found
assert os.path.isdir(ephys_file_dir) and os.path.isdir(alf_probe_dir) \
    and os.path.isabs(ephys_file_path), 'Directories set incorrectly'
    
# Call brainbox functions #
#-------------------------#

# Change variable names to same names used in brainbox docstrings
path_to_ephys_file = ephys_file_path
path_to_alf_out = alf_probe_dir
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

# Load alf objects:
spks_b = aio.load_object(path_to_alf_out, 'spikes')
clstrs_b = aio.load_object(path_to_alf_out, 'clusters')
chnls_b = aio.load_object(path_to_alf_out, 'channels')
tmplts_b = aio.load_object(path_to_alf_out, 'templates')

# Convert spikes bunch into a units bunch
units_b = bb.processing.get_units_bunch(spks_b)
unit4_amps = units_b['amps']['4']  # get amplitudes for unit 4.

# Filter units according to some parameters
filtered_units_mask = bb.processing.filter_units(spks_b, params={'min_amp': 100, 'min_fr': 0.5,
                                                                 'max_fpr': 0.1, 'rp': 0.002})
filtered_units = np.where(filtered_units_mask)[0]  # get an array of the filtered units` ids.

# Extract waveforms from binary ephys file
# Get the timestamps and 20 channels around the max amp channel for unit1, and extract the
# two sets of waveforms.
ts = units_b['times']['1']
max_ch = max_ch = clstrs_b['channels'][1]
if max_ch < 10:  # take only channels greater than `max_ch`.
    ch = np.arange(max_ch, max_ch + 20)
elif (max_ch + 10) > 385:  # take only channels less than `max_ch`.
    ch = np.arange(max_ch - 20, max_ch)
else:  # take `n_c_ch` around `max_ch`.
    ch = np.arange(max_ch - 10, max_ch + 10)
wf = bb.io.extract_waveforms(path_to_ephys_file, ts, ch, t=2.0, car=False)
wf_car = bb.io.extract_waveforms(path_to_ephys_file, ts, ch, t=2.0, car=True)

# Plot variances of a spike feature for all units and for a subset of units
fig1, var_vals, p_vals = bb.plot.feat_vars(spks_b, units=[], feat_name='amps')
fig2, var_vals, p_vals = bb.plot.feat_vars(spks_b, units=filtered_units, feat_name='amps')

# Plot distribution cutoff of a spike feature for a single unit
fig3, fraction_missing = bb.plot.feat_cutoff(spks_b, unit=1, feat_name='amps')

# Plot and compare two sets of waveforms from two different time epochs for a single unit
ts = units_b['times']['1']
ts1 = ts[np.where(ts<60)[0]]
ts2 = ts[np.where(ts>180)[0][:len(ts1)]]
fig4, wf_1, wf_2, s = bb.plot.single_unit_wf_comp(path_to_ephys_file, spks_b, clstrs_b, unit=1,
                                                 ts1=ts1, ts2=ts2, n_ch=20, car=True)

# Plot the instantaneous firing rate and its coefficient of variation for a single unit
fig5, fr, cv, cvs = bb.plot.firing_rate(spks_b, unit=1, t='all', hist_win=0.01, fr_win=0.5,
                                        n_bins=10, show_fr_cv=True)

# Save figs in a directory
fig_dir = os.getcwd()  # current working directory
fig_list = [fig1, fig2, fig3, fig4, fig5]
[f.savefig(os.path.join('fig'+ str(i + 1))) for i,f in enumerate(fig_list)]

'''
This script gives examples of how to load data from ONE and call brainbox functions.

*Note*: This module assumes that your python path has access to 'ibllib', and that the required
data for a particular eid is already saved in the CACHE_DIR specified by `.one_params` (the default
location to which ONE saves data when running the `load` method). It is recommended to download
*all* data for a particular eid, e.g.:
    `from oneibl.one import ONE`
    `one = ONE()`
    # get eid
    `eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]`
    # download data
    one.load(eid, dataset_types=one.list(), clobber=False, download_only=True)
'''

import os
from pathlib import Path
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import scipy.stats as stats
import pandas as pd
from oneibl.one import ONE
import alf.io as aio
import brainbox as bb
from ibllib.io.spikeglx import glob_ephys_files

# Set eid and probe name #
# ------------------------#
one = ONE()
eid = one.search(subject='ZM_2104', date='2019-09-19', number=1)[0]
probe = 'probe_right'  # *Note: new probe naming convention is 'probe00', 'probe01', etc.

# Get important directories from `eid` #
# --------------------------------------#
session_path = one.path_from_eid(eid)
alf_dir = Path.joinpath(session_path, 'alf')
alf_probe_dir = Path.joinpath(alf_dir, probe)
ephys_files = glob_ephys_files(session_path)
ephys_file_path = None
if ephys_files:
    ephys_file_path = ephys_files[0]['ap']
else:
    warn('Ephys file not found! Some of the examples in this script require ephys file,')

# Ensure at least alf directory can be found
assert Path.exists(alf_probe_dir), \
    'Directories set incorrectly. {} not found'.format(alf_probe_dir)

# Call brainbox functions #
# -------------------------#
# Change variable names to same names used in brainbox docstrings
path_to_ephys_file = ephys_file_path
path_to_alf_out = alf_probe_dir

# Load alf objects
spks_b = aio.load_object(path_to_alf_out, 'spikes')
clstrs_b = aio.load_object(path_to_alf_out, 'clusters')
chnls_b = aio.load_object(path_to_alf_out, 'channels')
tmplts_b = aio.load_object(path_to_alf_out, 'templates')

# Convert spikes bunch into a units bunch
units_b = bb.processing.get_units_bunch(spks_b)  # this may take a few mins
unit4_amps = units_b['amps']['4']  # get amplitudes for unit 4.

# Filter units according to some parameters
T = spks_b['times'][-1] - spks_b['times'][0]
filtered_units = bb.processing.filter_units(units_b, T, min_amp=0, min_fr=0.5, max_fpr=1, rp=0.002)

# Extract waveforms from binary ephys file
# Get the timestamps and 10 channels around the max amp channel for unit1, and extract the
# two sets of waveforms.
n_ch_wf = 10  # number of channels on which to extract waveforms
n_ch_probe = 385  # number of channels in recording
ts = units_b['times']['1']
max_ch = max_ch = clstrs_b['channels'][1]
if max_ch < (n_ch_wf // 2):  # take only channels greater than `max_ch`.
    ch = np.arange(max_ch, max_ch + n_ch_wf)
elif (max_ch + (n_ch_wf // 2)) > n_ch_probe:  # take only channels less than `max_ch`.
    ch = np.arange(max_ch - n_ch_wf, max_ch)
else:  # take `n_c_ch` around `max_ch`.
    ch = np.arange(max_ch - (n_ch_wf // 2), max_ch + (n_ch_wf // 2))

# Waveform extraction may take a few mins (check size of `ts` before hand. If you're extracting
# e.g. 10000+ spikes across 10 channels, this may take quite some time.)
wf = bb.io.extract_waveforms(path_to_ephys_file, ts, ch, t=2.0, car=False)
wf_car = bb.io.extract_waveforms(path_to_ephys_file, ts, ch, t=2.0, car=True)

# Plot amplitude heatmap for a unit with and without car
V_vals = bb.plot.amp_heatmap(path_to_ephys_file, ts, ch, car=False)
fig1 = plt.gcf()
V_vals_car = bb.plot.amp_heatmap(path_to_ephys_file, ts, ch, car=True)  # may take a few mins
fig2 = plt.gcf()

# Plot variances of a spike feature for all units and for a subset of units
var_vals, p_vals = bb.plot.feat_vars(units_b, feat_name='amps')
fig3 = plt.gcf()
var_vals_f, p_vals_f = bb.plot.feat_vars(units_b, units=filtered_units, feat_name='amps')
fig4 = plt.gcf()

# Plot distribution cutoff of a spike feature for a single unit
amps = units_b['amps']['1']
fraction_missing = bb.plot.missed_spikes_est(amps, feat_name='amps')
fig5 = plt.gcf()

# Plot and compare two sets of waveforms from two different time epochs for a single unit
ts = units_b['times']['1']
ts1 = ts[np.where(ts < 60)[0]]
ts2 = ts[np.where(ts > 180)[0][:len(ts1)]]
wf_1, wf_2, s = bb.plot.wf_comp(path_to_ephys_file, ts1, ts2, ch, car=True)  # may take a few mins
fig6 = plt.gcf()

# Plot the instantaneous firing rate and its coefficient of variation for a single unit
fr, cv, cvs = bb.plot.firing_rate(ts, hist_win=0.01, fr_win=0.5, n_bins=10, show_fr_cv=True)
fig7 = plt.gcf()

# Plot the presence ratio for a single unit.
pr, pr_bins = bb.plot.pres_ratio(ts)
fig8 = plt.gcf()

# Plot the amplitude and depth driftmaps for a single unit.
cum_drift_amps, max_drift_amps = bb.plot.driftmap(ts, amps)
fig9 = plt.gcf()
fig9.axes[0].set_xlabel('Time (s)')
fig9.axes[0].set_xlabel('Voltage (V)')

depths = units_b.depths['1']
cum_drift_depth, max_drift_depth = bb.plot.driftmap(ts, depths)
fig10 = plt.gcf()
fig10.axes[0].set_xlabel('Time (s)')
fig10.axes[0].set_ylabel('Depth (mm)')

# Plot the peth for a single unit based on trial events. (This example requires '_ibl_trials')
eid = one.search(subject='KS022', date='2019-12-10', number=1)[0]  # have to use a task session
dtypes = [
    'clusters.amps',
    'clusters.channels',
    'clusters.depths',
    'clusters.metrics',
    'clusters.peakToTrough',
    'clusters.uuids',
    'clusters.waveforms',
    'clusters.waveformsChannels',
    'spikes.amps',
    'spikes.clusters',
    'spikes.depths',
    'spikes.samples',
    'spikes.templates',
    'spikes.times',
    'trials.contrastLeft',
    'trials.contrastRight',
    'trials.feedback_times',
    'trials.feedbackType',
    'trials.goCue_times',
    'trials.goCueTrigger_times',
    'trials.included',
    'trials.intervals',
    'trials.itiDuration',
    'trials.probabilityLeft',
    'trials.repNum',
    'trials.response_times',
    'trials.rewardVolume',
    'trials.stimOn_times',
]
# get appropriate paths
one.load(eid, dataset_types=dtypes, clobber=False, download_only=True)
session_path = one.path_from_eid(eid)
alf_dir = Path.joinpath(session_path, 'alf')
alf_probe_dir = Path.joinpath(alf_dir, probe)

# get trials bunch
trials = aio.load_object(alf_dir, 'trials')

# plot peth without raster (spike times, all cluster ids, event times, cluster id)
bb.plot.peri_event_time_histogram(spks_b.times, spks_b.clusters, trials.goCue_times, 1)
fig11 = plt.gcf()

# plot peth with underlaid raster for each event, showing spikes 0.25 seconds before and after
# each event
bb.plot.peri_event_time_histogram(
    spks_b.times, spks_b.clusters, trials.goCue_times, 1, t_before=0.25, t_after=0.25,
    include_raster=True)
fig12 = plt.gcf()

# Save figs in a directory
fig_dir = Path.cwd()  # current working directory
fig_list = [fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11, fig12]
[f.savefig(Path.joinpath(fig_dir, 'fig' + str(i + 1))) for i, f in enumerate(fig_list)]

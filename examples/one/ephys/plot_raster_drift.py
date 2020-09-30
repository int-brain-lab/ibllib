"""
Compute drift for example sessions using:
https://github.com/int-brain-lab/ibllib/blob/master/brainbox/metrics/electrode_drift.py
and display raster plot below
"""
# Authors: Gaelle, Olivier

from brainbox.metrics.electrode_drift import estimate_drift
from oneibl.one import ONE
import brainbox.plot as bbplot
import matplotlib.pyplot as plt

one = ONE()

# Find sessions
dataset_types = ['spikes.times',
                 'spikes.amps',
                 'spikes.depths']

eids = one.search(dataset_types=dataset_types,
                  project='ibl_neuropixel_brainwide_01',
                  task_protocol='_iblrig_tasks_ephysChoiceWorld')

eid = eids[0]  # Test

# Get dataset

spike_times, spike_amps, spike_depths = \
    one.load(eid, dataset_types=dataset_types)

drift = estimate_drift(spike_times, spike_amps, spike_depths, display=False)

# PLOT
fig, axs = plt.subplots(2, 1)
# Drift
axs[0].plot(drift)
# Raster plot -- Brainbox
bbplot.driftmap(spike_times,
                spike_depths,
                ax=axs[1], plot_style='bincount')

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

# eids = one.search(dataset_types=dataset_types,
#                   project='ibl_neuropixel_brainwide_01',
#                   task_protocol='_iblrig_tasks_ephysChoiceWorld')
#
# eid = eids[0]  # Test with little drift: '7cdb71fb-928d-4eea-988f-0b655081f21c'

eid = '89f0d6ff-69f4-45bc-b89e-72868abb042a'  # Test with huge drift

# Get dataset

spike_times, spike_amps, spike_depths = \
    one.load(eid, dataset_types=dataset_types)

drift = estimate_drift(spike_times, spike_amps, spike_depths, display=False)

# PLOT
# Tight layout
fig3 = plt.figure(constrained_layout=True)
gs = fig3.add_gridspec(3, 3)
f3_ax0 = fig3.add_subplot(gs[0, :])
f3_ax0.plot(drift)
f3_ax1 = fig3.add_subplot(gs[1:, :])
bbplot.driftmap(spike_times,
                spike_depths,
                ax=f3_ax1, plot_style='bincount')
f3_ax0.set_xlim(f3_ax1.get_xlim())

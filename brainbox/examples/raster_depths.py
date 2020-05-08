"""
author OW
last reviewed/run 19-04-2020
"""
import matplotlib.pyplot as plt
import numpy as np

from oneibl.one import ONE
import ibllib.plots as iblplt

from brainbox.processing import bincount2D
from brainbox.io import one as bbone

T_BIN = 0.05
D_BIN = 5

# get the data from flatiron and the current folder
one = ONE()
eid = one.search(subject='CSHL045', date='2020-02-26', number=1)[0]

spikes, clusters, trials = bbone.load_ephys_session(eid, one=one, dataset_types=['spikes.depth'])

pname = list(spikes.keys())[0]

# compute raster map as a function of site depth
R, times, depths = bincount2D(spikes[pname]['times'], spikes[pname]['depths'], T_BIN, D_BIN)

# plot raster map
plt.imshow(R, aspect='auto', cmap='binary', vmax=T_BIN / 0.001 / 4,
           extent=np.r_[times[[0, -1]], depths[[0, -1]]], origin='lower')
# plot trial start and reward time
reward = trials['feedback_times'][trials['feedbackType'] == 1]
iblplt.vertical_lines(trials['intervals'][:, 0], ymin=0, ymax=depths[-1],
                      color='k', linewidth=0.5, label='trial starts')
iblplt.vertical_lines(reward, ymin=0, ymax=depths[-1], color='m', linewidth=0.5,
                      label='valve openings')
plt.xlabel('Time (s)')
plt.ylabel('Cluster #')
plt.legend()

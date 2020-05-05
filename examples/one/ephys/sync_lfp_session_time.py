"""
Reads in and display a chunk of raw LFP synchronized on session time
"""
# Author: Olivier

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from ibllib.io import spikeglx

lf_file = ('/datadisk/FlatIron/churchlandlab/Subjects/CSHL045/2020-02-26/001/'
           'raw_ephys_data/probe01/_spikeglx_ephysData_g0_t0.imec.lf.cbin')

sr = spikeglx.Reader(lf_file)

sync_file = sr.file_bin.parent.joinpath(sr.file_bin.stem.replace('.lf', '.sync.npy'))
sync = np.load(sync_file)
sample2time = scipy.interpolate.interp1d(sync[:, 0] * sr.fs, sync[:, 1])

data = sr[105000:109000, :-1]
data = data - np.mean(data)
tscale = sample2time(np.array([105000, 109000]))

plt.figure()
im = plt.imshow(data.transpose(), aspect='auto',
                extent=[*tscale, data.shape[1], 0])
plt.xlabel('session time (sec)')
plt.ylabel('channel')

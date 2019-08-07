from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import alf.io
from brainbox.core import Bunch
from ibllib.io.extractors.ephys_fpga import (SYNC_CHANNEL_MAP,
                                             _get_ephys_files,
                                             _get_sync_fronts)

FS = 30000

ses_path = Path('/mnt/s0/Data/Subjects/ZM_1887/2019-07-19/001')
ephys_files = _get_ephys_files(ses_path)

nprobes = len(ephys_files)
assert(nprobes >= 2)

d = Bunch({'left_camera': None, 'right_camera': None, 'body_camera': None,
           'nsync': np.zeros(nprobes,)})
for ind, ephys_file in enumerate(ephys_files):
    sync = alf.io.load_object(ephys_file.ap.parent, '_spikeglx_sync', short_keys=True)
    d.nsync[ind] = len(sync.channels)
    # init the array if necessary
    lc = _get_sync_fronts(sync, SYNC_CHANNEL_MAP['left_camera'])['times']
    rc = _get_sync_fronts(sync, SYNC_CHANNEL_MAP['right_camera'])['times']
    bc = _get_sync_fronts(sync, SYNC_CHANNEL_MAP['body_camera'])['times']
    if ind == 0:
        d['left_camera'] = np.zeros((lc.size, nprobes))
        d['right_camera'] = np.zeros((rc.size, nprobes))
        d['body_camera'] = np.zeros((bc.size, nprobes))
    d['left_camera'][:, ind] = lc
    d['body_camera'][:, ind] = bc
    d['right_camera'][:, ind] = rc

# the reference probe is the one with the most sync pulses detected
iref = np.argmax(d.nsync)

dlc = np.diff(d['left_camera'], axis=1)
drc = np.diff(d['right_camera'], axis=1)
dbc = np.diff(d['body_camera'], axis=1)

plt.plot(d.left_camera[:, iref], dlc)

slope = np.polyfit(d.left_camera[:, iref], dlc, 1)[0]

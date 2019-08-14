from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import alf.io
from brainbox.core import Bunch
import ibllib.io.extractors.ephys_fpga as ephys_fpga
import ibllib.io.spikeglx as spikeglx

FS = 30000

ses_path = Path('/mnt/s0/Data/Subjects/ZM_1887/2019-07-19/001')
ephys_files = ephys_fpga._get_ephys_files(ses_path)

nprobes = len(ephys_files)
assert(nprobes >= 2)

d = Bunch({'times': None, 'nsync': np.zeros(nprobes,)})

for ind, ephys_file in enumerate(ephys_files):
    sync = alf.io.load_object(ephys_file.ap.parent, '_spikeglx_sync', short_keys=True)
    sync_map = ephys_fpga.get_sync_map(ephys_file.ap.parent)
    isync = np.in1d(sync['channels'], np.array([sync_map['right_camera'],
                                                sync_map['left_camera'],
                                                sync_map['body_camera']]))
    d.nsync[ind] = len(sync.channels)
    # this is designed to break if the number of fronts are different for one probe to the other
    if ind == 0:
        d['times'] = np.zeros((np.sum(isync), nprobes))
    d['times'][:, ind] = sync['times'][isync]

##
# the reference probe is the one with the most sync pulses detected
iref = np.argmax(d.nsync)
islave = np.setdiff1d(np.arange(nprobes), iref)
# get the sampling rate from the reference probe using metadata file
meta = spikeglx.read_meta_data(Path(ephys_files[iref].ap).with_suffix('.meta'))
sr = meta['imSampRate']


# plt.plot(d.times[:, iref, np.newaxis] - d.times[:, islave])


# output timestamps files as per ALF convention
for ind, ephys_file in enumerate(ephys_files):
    if ind == iref:
        timestamps = np.array([[0, 0], [sr, 1]])
    else:
        timestamps = sync_probes(d.times[:, iref], d.times[:, ind], sr)
    np.save(ephys_file.ap.parent / '_spikeglx_sync.timestamps.npy', timestamps)

##
def sync_probes(t, tref, sr):
    """
    From 2 timestamps vectors of equivalent length, output timestamps array to be used for
    linear interpolation
    :param t: time-serie to be synchronized
    :param tref: time-serie of the reference
    :param sr: sampling rate of the slave probe
    :return: a 2 columns by n-sync points array where is row corresponds
    to a sync point sample_index (0 based), tref
    """
    # as of now this is done through linear regression for 3A. The precision is enforced
    # by ensuring that each point lies less than one sampling rate away from the predicted
    # regression value. If this is the case, higher order smoothing function will be needed.
    pol = np.polyfit(t, tref, 1)  # higher order terms first: slope / int for linear

    plt.plot(tref, (np.polyval(pol, t) - tref) * sr)
    plt.ylabel('Residual drift (samples @ 30kHz)')
    plt.xlabel('time (sec)')

    return np.c_[np.array([0, sr]), np.polyval(pol, np.array([0, 1]))]


sync_probes(d.times[:, ind], d.times[:, iref], sr)

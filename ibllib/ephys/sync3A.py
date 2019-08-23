from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import alf.io
import ibllib.io.spikeglx
from brainbox.core import Bunch
import ibllib.io.spikeglx as spikeglx


def sync_probe_folders_3A(ses_path):
    """
    From a session path with _spikeglx_sync arrays extracted, locate ephys files for 3A and
     outputs one sync.timestamps.probeN.npy file per acquired probe. By convention the reference
     probe is the one with the most synchronisation pulses.
    :param ses_path:
    :return: None
    """
    ephys_files = ibllib.io.spikeglx.glob_ephys_files(ses_path)
    nprobes = len(ephys_files)
    assert (nprobes >= 2)

    d = Bunch({'times': None, 'nsync': np.zeros(nprobes, )})

    for ind, ephys_file in enumerate(ephys_files):
        sync = alf.io.load_object(ephys_file.ap.parent, '_spikeglx_sync', short_keys=True)
        sync_map = ibllib.io.spikeglx.get_sync_map(ephys_file.ap.parent)
        isync = np.in1d(sync['channels'], np.array([sync_map['right_camera'],
                                                    sync_map['left_camera'],
                                                    sync_map['body_camera']]))
        d.nsync[ind] = len(sync.channels)
        # this is designed to break if the number of fronts per probe are not equal
        if ind == 0:
            d['times'] = np.zeros((np.sum(isync), nprobes))
        d['times'][:, ind] = sync['times'][isync]

    # the reference probe is the one with the most sync pulses detected
    iref = np.argmax(d.nsync)
    # islave = np.setdiff1d(np.arange(nprobes), iref)
    # get the sampling rate from the reference probe using metadata file
    meta = spikeglx.read_meta_data(Path(ephys_files[iref].ap).with_suffix('.meta'))
    sr = meta['imSampRate']

    # output timestamps files as per ALF convention
    for ind, ephys_file in enumerate(ephys_files):
        if ind == iref:
            timestamps = np.array([[0, 0], [sr, 1]])
        else:
            timestamps = sync_probe_front_times(d.times[:, iref], d.times[:, ind], sr)
        alf.io.save_object_npy(ephys_file.ap.parent, {'timestamps': timestamps},
                               object='sync', parts=ephys_file.label)


def sync_probe_folders_3B(ses_path):
    """
    From a session path with _spikeglx_sync arrays extraccted, locate ephys files for 3A and
     outputs one sync.timestamps.probeN.npy file per acquired probe. By convention the reference
     probe is the one with the most synchronisation pulses.
    :param ses_path:
    :return: None
    """
    pass


def sync_probe_front_times(t, tref, sr, display=False):
    """
    From 2 timestamps vectors of equivalent length, output timestamps array to be used for
    linear interpolation
    :param t: time-serie to be synchronized
    :param tref: time-serie of the reference
    :param sr: sampling rate of the slave probe
    :return: a 2 columns by n-sync points array where each row corresponds
    to a sync point: sample_index (0 based), tref
    """
    COMPUTE_RESIDUAL = True
    # the main drift is computed through linear regression. A further step compute a smoothed
    # version of the residual to add to the linear drift. The precision is enforced
    # by ensuring that each point lies less than one sampling rate away from the predicted.
    pol = np.polyfit(t, tref, 1)  # higher order terms first: slope / int for linear
    residual = (tref - np.polyval(pol, t))
    if COMPUTE_RESIDUAL:
        # the interp function from camera fronts is not smooth due to the locking of detections
        # to the sampling rate of digital channels. The residual is fit using frequency domain
        # smoothing
        import ibllib.dsp as dsp
        CAMERA_UPSAMPLING_RATE_HZ = 300
        PAD_LENGTH_SECS = 60
        STAT_LENGTH_SECS = 30  # median length to compute padding value
        SYNC_SAMPLING_RATE_SECS = 20
        t_upsamp = np.arange(tref[0], tref[-1], 1 / CAMERA_UPSAMPLING_RATE_HZ)
        res_upsamp = np.interp(t_upsamp, tref, residual)
        # padding needs extra care as the function oscillates
        lpad = int(sr * PAD_LENGTH_SECS)
        res_filt = np.pad(res_upsamp, lpad, mode='median', stat_length=sr * STAT_LENGTH_SECS)
        fbounds = 1 / SYNC_SAMPLING_RATE_SECS * np.array([2, 4])
        res_filt = dsp.lp(res_filt, 1 / sr, fbounds)[lpad:-lpad]
        tout = np.arange(0, np.max(tref) + SYNC_SAMPLING_RATE_SECS, 20)
        sync_points = np.c_[tout * sr, np.polyval(pol, tout) - np.interp(tout, t_upsamp, res_filt)]
        if display:
            plt.plot(tref, residual * sr)
            plt.plot(t_upsamp, res_filt * sr)
            plt.plot(tout, np.interp(tout, t_upsamp, res_filt) * sr, '*')
            plt.ylabel('Residual drift (samples @ 30kHz)')
            plt.xlabel('time (sec)')
    else:
        sync_points = np.c_[np.array([0, sr]), np.polyval(pol, np.array([0, 1]))]
        if display:
            plt.plot(tref, residual * sr)
            plt.ylabel('Residual drift (samples @ 30kHz)')
            plt.xlabel('time (sec)')
    return sync_points


if __name__ == '__main__':
    ses_path = Path('/mnt/s0/Data/Subjects/ZM_1887/2019-07-19/001')
    sync_probe_folders_3A(ses_path)

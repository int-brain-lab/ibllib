from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import alf.io
import ibllib.io.spikeglx
from brainbox.core import Bunch
import ibllib.io.spikeglx as spikeglx
from ibllib.io.extractors.ephys_fpga import CHMAPS


def sync_probe_folders_3A(ses_path, display=False):
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
        sync_map = ibllib.io.spikeglx.get_sync_map(ephys_file.ap.parent) or CHMAPS['3A']
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
            timestamps = np.array([[0, 0], [1, 1]])
        else:
            timestamps = sync_probe_front_times(d.times[:, iref], d.times[:, ind], sr,
                                                display=display)
        assert(ephys_file.ap.name.endswith('.ap.bin'))
        file_out = ephys_file.ap.parent / ephys_file.ap.name.replace('.ap.bin', '.sync.npy')
        np.save(file_out, timestamps)


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
        # padding needs extra care as the function oscillates and numpy fft performance is
        # abysmal for non prime sample sizes
        nech = res_upsamp.size + (CAMERA_UPSAMPLING_RATE_HZ * PAD_LENGTH_SECS)
        lpad = 2 ** np.ceil(np.log2(nech)) - res_upsamp.size
        lpad = [int(np.floor(lpad / 2) + lpad % 2), int(np.floor(lpad / 2))]
        res_filt = np.pad(res_upsamp, lpad, mode='median',
                          stat_length=CAMERA_UPSAMPLING_RATE_HZ * STAT_LENGTH_SECS)
        fbounds = [0.001, 0.002]
        res_filt = dsp.lp(res_filt, 1 / CAMERA_UPSAMPLING_RATE_HZ, fbounds)[lpad[0]:-lpad[1]]
        tout = np.arange(0, np.max(tref) + SYNC_SAMPLING_RATE_SECS, 20)
        sync_points = np.c_[tout, np.polyval(pol, tout) - np.interp(tout, t_upsamp, res_filt)]
        if display:
            plt.plot(tref, residual * sr)
            plt.plot(t_upsamp, res_filt * sr)
            plt.plot(tout, np.interp(tout, t_upsamp, res_filt) * sr, '*')
            plt.ylabel('Residual drift (samples @ 30kHz)')
            plt.xlabel('time (sec)')
    else:
        sync_points = np.c_[np.array([0, 1]), np.polyval(pol, np.array([0, 1]))]
        if display:
            plt.plot(tref, residual * sr)
            plt.ylabel('Residual drift (samples @ 30kHz)')
            plt.xlabel('time (sec)')
    return sync_points


if __name__ == '__main__':
    ses_path = Path('/mnt/s0/Data/Subjects/ZM_1887/2019-07-19/001')
    sync_probe_folders_3A(ses_path)

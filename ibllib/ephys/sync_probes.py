import logging
from pathlib import Path

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

import alf.io
from brainbox.core import Bunch
import ibllib.io.spikeglx as spikeglx
from ibllib.misc import log2session_static
from ibllib.ephys import spikes
from ibllib.io import flags
from ibllib.io.extractors.ephys_fpga import CHMAPS, _get_sync_fronts

_logger = logging.getLogger('ibllib')


@log2session_static('ephys')
def sync_merge(session_path, dry=False, force=False):
    """
    Sync probes and merge spike sorting output.
    For single probe dataset, output ks2 as ALF dataset
    """
    session_path = Path(session_path)
    sync(session_path, display=False)
    spikes.merge_probes(session_path)
    flags.write_flag_file(session_path.joinpath('register_me.flag'))


@log2session_static('ephys')
def sync(ses_path, **kwargs):
    """
    Wrapper for sync_probes.version3A and sync_probes.version3B that automatically determines
    the version
    :param ses_path:
    :return: bool True on a a successful sync
    """
    version = spikeglx.get_neuropixel_version_from_folder(ses_path)
    if version == '3A':
        version3A(ses_path, **kwargs)
    elif version == '3B':
        version3B(ses_path, **kwargs)


def version3A(ses_path, display=True, linear=False, tol=1.5):
    """
    From a session path with _spikeglx_sync arrays extracted, locate ephys files for 3A and
     outputs one sync.timestamps.probeN.npy file per acquired probe. By convention the reference
     probe is the one with the most synchronisation pulses.
     Assumes the _spikeglx_sync datasets are already extracted from binary data
    :param ses_path:
    :return: bool True on a a successful sync
    """
    ephys_files = spikeglx.glob_ephys_files(ses_path)
    nprobes = len(ephys_files)
    if nprobes <= 1:
        _logger.warning(f"Skipping single probe session: {ses_path}")
        return True
    d = Bunch({'times': [], 'nsync': np.zeros(nprobes, )})
    for ind, ephys_file in enumerate(ephys_files):
        sync = alf.io.load_object(ephys_file.ap.parent, '_spikeglx_sync', short_keys=True)
        sync_map = spikeglx.get_sync_map(ephys_file.ap.parent) or CHMAPS['3A']
        isync = np.in1d(sync['channels'], np.array([sync_map['right_camera']]))
        d.nsync[ind] = len(sync.channels)
        d['times'].append(sync['times'][isync])
    # chop off to the lowest number of sync points
    nsyncs = [t.size for t in d['times']]
    if len(set(nsyncs)) > 1:
        _logger.warning("Probes don't have the same number of synchronizations pulses")
    d['times'] = np.r_[[t[:min(nsyncs)] for t in d['times']]].transpose()

    # the reference probe is the one with the most sync pulses detected
    iref = np.argmax(d.nsync)
    # islave = np.setdiff1d(np.arange(nprobes), iref)
    # get the sampling rate from the reference probe using metadata file
    sr = _get_sr(ephys_files[iref])
    qc_all = True
    # output timestamps files as per ALF convention
    for ind, ephys_file in enumerate(ephys_files):
        if ind == iref:
            timestamps = np.array([[0, 0], [1, 1]])
        else:
            timestamps, qc = sync_probe_front_times(d.times[:, iref], d.times[:, ind], sr,
                                                    display=display, linear=linear, tol=tol)
            qc_all &= qc
        _save_timestamps_npy(ephys_file, timestamps)
    return qc_all


def version3B(ses_path, display=True, linear=False, tol=2.5):
    """
    From a session path with _spikeglx_sync arrays extraccted, locate ephys files for 3A and
     outputs one sync.timestamps.probeN.npy file per acquired probe. By convention the reference
     probe is the one with the most synchronisation pulses.
     Assumes the _spikeglx_sync datasets are already extracted from binary data
    :param ses_path:
    :return: None
    """
    ephys_files = spikeglx.glob_ephys_files(ses_path)
    for ef in ephys_files:
        ef['sync'] = alf.io.load_object(ef.path, '_spikeglx_sync', short_keys=True)
        ef['sync_map'] = spikeglx.get_sync_map(ef['path']) or CHMAPS['3B']
    nidq_file = [ef for ef in ephys_files if ef.get('nidq')]
    ephys_files = [ef for ef in ephys_files if not ef.get('nidq')]
    nprobes = len(ephys_files)
    # should have at least 2 probes and only one nidq
    if nprobes <= 1:
        return True
    assert(len(nidq_file) == 1)
    nidq_file = nidq_file[0]
    sync_nidq = _get_sync_fronts(nidq_file.sync, nidq_file.sync_map['imec_sync'])

    qc_all = True
    for ef in ephys_files:
        sync_probe = _get_sync_fronts(ef.sync, ef.sync_map['imec_sync'])
        sr = _get_sr(ef)
        assert(sync_nidq.times.size == sync_probe.times.size)
        timestamps, qc = sync_probe_front_times(sync_probe.times, sync_nidq.times, sr,
                                                display=display, linear=linear, tol=tol)
        qc_all &= qc
        _save_timestamps_npy(ef, timestamps)
    return qc_all


def sync_probe_front_times(t, tref, sr, display=False, linear=False, tol=2.0):
    """
    From 2 timestamps vectors of equivalent length, output timestamps array to be used for
    linear interpolation
    :param t: time-serie to be synchronized
    :param tref: time-serie of the reference
    :param sr: sampling rate of the slave probe
    :return: a 2 columns by n-sync points array where each row corresponds
    to a sync point: sample_index (0 based), tref
    :return: quality Bool. False if tolerance is exceeded
    """
    qc = True
    # the main drift is computed through linear regression. A further step compute a smoothed
    # version of the residual to add to the linear drift. The precision is enforced
    # by ensuring that each point lies less than one sampling rate away from the predicted.
    pol = np.polyfit(t, tref, 1)  # higher order terms first: slope / int for linear
    residual = (tref - np.polyval(pol, t))
    if not linear:
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
        sync_points = np.c_[tout, np.polyval(pol, tout) + np.interp(tout, t_upsamp, res_filt)]
        if display:
            if isinstance(display, matplotlib.axes.Axes):
                ax = display
            else:
                ax = plt.axes()
            ax.plot(tref, residual * sr)
            ax.plot(t_upsamp, res_filt * sr)
            ax.plot(tout, np.interp(tout, t_upsamp, res_filt) * sr, '*')
            ax.set_xlabel('time (sec)')
            ax.set_ylabel('Residual drift (samples @ 30kHz)')
    else:
        sync_points = np.c_[np.array([0, 1]), np.polyval(pol, np.array([0, 1]))]
        if display:
            plt.plot(tref, residual * sr)
            plt.ylabel('Residual drift (samples @ 30kHz)')
            plt.xlabel('time (sec)')
    # test that the interp is within tol sample
    fcn = interp1d(sync_points[:, 0], sync_points[:, 1], fill_value='extrapolate')
    if np.any(np.abs((tref - fcn(t)) * sr) > (tol)):
        _logger.error(f'Synchronization check exceeds tolerance of {tol} samples. Check !!')
        qc = False
        # plt.plot((tref - fcn(t)) * sr)
    return sync_points, qc


def _get_sr(ephys_file):
    meta = spikeglx.read_meta_data(ephys_file.ap.with_suffix('.meta'))
    return spikeglx._get_fs_from_meta(meta)


def _save_timestamps_npy(ephys_file, timestamps):
    assert (ephys_file.ap.name.endswith('.ap.bin'))
    file_out = ephys_file.ap.parent / ephys_file.ap.name.replace('.ap.bin', '.sync.npy')
    np.save(file_out, timestamps)

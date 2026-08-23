import logging

import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
import one.alf.io as alfio
import one.alf.exceptions
from iblutil.util import Bunch
import spikeglx

from ibllib.exceptions import Neuropixel3BSyncFrontsNonMatching, SyncFrontsAnomaly
from ibllib.io.extractors.ephys_fpga import get_sync_fronts, get_ibl_sync_map

_logger = logging.getLogger(__name__)


def apply_sync(sync_file, times, forward=True):
    """
    :param sync_file: probe sync file (usually of the form _iblrig_ephysData.raw.imec1.sync.npy)
    :param times: times in seconds to interpolate
    :param forward: if True goes from probe time to session time, from session time to probe time
    otherwise
    :return: interpolated times
    """
    sync_points = np.load(sync_file)
    if forward:
        fcn = interp1d(sync_points[:, 0], sync_points[:, 1], fill_value='extrapolate')
    else:
        fcn = interp1d(sync_points[:, 1], sync_points[:, 0], fill_value='extrapolate')
    return fcn(times)


def sync(ses_path, **kwargs):
    """
    Wrapper for sync_probes.version3A and sync_probes.version3B that automatically determines
    the version
    :param ses_path:
    :return: bool True on a a successful sync
    """
    version = spikeglx.get_neuropixel_version_from_folder(ses_path)
    if version == '3A':
        return version3A(ses_path, **kwargs)
    elif version == '3B':
        return version3B(ses_path, **kwargs)


def version3A(ses_path, display=True, type='smooth', tol=2.1, probe_names=None, raise_on_anomaly=False):
    """
    From a session path with _spikeglx_sync arrays extracted, locate ephys files for 3A and
     outputs one sync.timestamps.probeN.npy file per acquired probe. By convention the reference
     probe is the one with the most synchronisation pulses.
     Assumes the _spikeglx_sync datasets are already extracted from binary data
    :param ses_path:
    :param type: linear, exact or smooth
    :param raise_on_anomaly: if True, raise SyncFrontsAnomaly when the sync_probe_front_times
     tolerance check fails for a probe, instead of only logging it and continuing with
     qc_all=False (default). No auto-switching between frame2ttl/right_camera is attempted.
    :return: bool True on a a successful sync
    """
    ephys_files = spikeglx.glob_ephys_files(ses_path, ext='meta', bin_exists=False)
    nprobes = len(ephys_files)
    if nprobes == 1:
        timestamps = np.array([[0.0, 0.0], [1.0, 1.0]])
        sr = _get_sr(ephys_files[0])
        out_files = _save_timestamps_npy(ephys_files[0], timestamps, sr)
        return True, out_files

    def get_sync_fronts(auxiliary_name):
        d = Bunch({
            'times': [],
            'nsync': np.zeros(
                nprobes,
            ),
        })
        # auxiliary_name: frame2ttl or right_camera
        for ind, ephys_file in enumerate(ephys_files):
            sync = alfio.load_object(ephys_file.ap.parent, 'sync', namespace='spikeglx', short_keys=True)
            sync_map = get_ibl_sync_map(ephys_file, '3A')
            # exits if sync label not found for current probe
            if auxiliary_name not in sync_map:
                return
            isync = np.in1d(sync['channels'], np.array([sync_map[auxiliary_name]]))
            # only returns syncs if we get fronts for all probes
            if np.all(~isync):
                return
            d.nsync[ind] = len(sync.channels)
            d['times'].append(sync['times'][isync])
        return d

    aux_used = 'frame2ttl'
    d = get_sync_fronts('frame2ttl')
    if not d:
        _logger.warning('Ephys sync: frame2ttl not detected on both probes, using camera sync')
        aux_used = 'right_camera'
        d = get_sync_fronts('right_camera')
        if not min([t[0] for t in d['times']]) > 0.2:
            raise ValueError('Cameras started before ephys, no sync possible')
    # chop off to the lowest number of sync points
    nsyncs = [t.size for t in d['times']]
    if len(set(nsyncs)) > 1:
        _logger.warning("Probes don't have the same number of synchronizations pulses")
    d['times'] = np.r_[[t[: min(nsyncs)] for t in d['times']]].transpose()

    # the reference probe is the one with the most sync pulses detected
    iref = np.argmax(d.nsync)
    # islave = np.setdiff1d(np.arange(nprobes), iref)
    # get the sampling rate from the reference probe using metadata file
    sr = _get_sr(ephys_files[iref])
    qc_all = True
    # output timestamps files as per ALF convention
    for ind, ephys_file in enumerate(ephys_files):
        if ind == iref:
            timestamps = np.array([[0.0, 0.0], [1.0, 1.0]])
        else:
            timestamps, qc = sync_probe_front_times(d.times[:, ind], d.times[:, iref], sr, display=display, type=type, tol=tol)
            qc_all &= qc
            if not qc:
                probe_name = ephys_file.path.parts[-1]
                _logger.error(f'{ses_path} probe {probe_name}: sync anomaly using aux channel "{aux_used}"')
                if raise_on_anomaly:
                    raise SyncFrontsAnomaly(f'{ses_path} probe {probe_name}: aux channel "{aux_used}"')
        out_files = _save_timestamps_npy(ephys_file, timestamps, sr)
    return qc_all, out_files


def version3B(ses_path, display=True, type=None, tol=2.5, probe_names=None, raise_on_anomaly=False):
    """
    From a session path with _spikeglx_sync arrays extracted, locate ephys files for 3A and
     outputs one sync.timestamps.probeN.npy file per acquired probe. By convention the reference
     probe is the one with the most synchronisation pulses.
     Assumes the _spikeglx_sync datasets are already extracted from binary data
    :param ses_path:
    :param type: linear, exact or smooth
    :param probe_names: by default will rglob all probes in the directory. If specified, this will filter
     the probes on which to perform the synchronisation, defaults to None, optional
    :param raise_on_anomaly: if True, raise SyncFrontsAnomaly when a classifiable raw-pulse
     pathology (dropped_edges / duplicate_burst / single_edge_glitch) is found on the probe or
     reference channel, instead of only logging it and continuing with qc_all=False (default).
    :return: None
    """
    DEFAULT_TYPE = 'smooth'
    ephys_files = spikeglx.glob_ephys_files(ses_path, ext='meta', bin_exists=False)
    for ef in ephys_files:
        try:
            ef['sync'] = alfio.load_object(ef.path, 'sync', namespace='spikeglx', short_keys=True)
            ef['sync_map'] = get_ibl_sync_map(ef, '3B')
        except one.alf.exceptions.ALFObjectNotFound as e:
            if probe_names is None or ef.path.parts[-1] in probe_names:
                raise e
    nidq_file = [ef for ef in ephys_files if ef.get('nidq')]
    ephys_files = [ef for ef in ephys_files if not ef.get('nidq')]
    if probe_names is not None:
        ephys_files = [ef for ef in ephys_files if ef.path.parts[-1] in probe_names]
    # should have at least 2 probes and only one nidq
    assert len(nidq_file) == 1
    nidq_file = nidq_file[0]
    sync_nidq = get_sync_fronts(nidq_file.sync, nidq_file.sync_map['imec_sync'])

    qc_all = True
    out_files = []
    for ef in ephys_files:
        sync_probe = get_sync_fronts(ef.sync, ef.sync_map['imec_sync'])
        sr = _get_sr(ef)
        try:
            # we say that the number of pulses should be within 10 %
            assert np.isclose(sync_nidq.times.size, sync_probe.times.size, rtol=0.1)
        except AssertionError:
            raise Neuropixel3BSyncFrontsNonMatching(f'{ses_path}')

        # Find the indexes in case the sizes don't match
        if sync_nidq.times.size != sync_probe.times.size:
            _logger.warning(
                f'Sync mismatch by {np.abs(sync_nidq.times.size - sync_probe.times.size)} '
                f'NIDQ sync times: {sync_nidq.times.size}, Probe sync times {sync_probe.times.size}'
            )
        sync_idx = np.min([sync_nidq.times.size, sync_probe.times.size])

        # if the qc of the diff finds anomalies, do not attempt to smooth the interp function
        qcdiff = _check_diff_3b(sync_probe)
        if not qcdiff:
            qc_all = False
            type_probe = type or 'exact'
            # _check_diff_3b filters to rising edges only (polarities == 1); match that here so
            # the diagnosis is computed on the same front subset that triggered the anomaly.
            label = (classify_sync_anomaly(sync_probe.times[sync_probe.polarities == 1])
                     or classify_sync_anomaly(sync_nidq.times[sync_nidq.polarities == 1]))
            _logger.error(f'{ses_path} probe {ef.path.parts[-1]}: sync anomaly detected: {label}')
            if raise_on_anomaly and label is not None:
                raise SyncFrontsAnomaly(f'{ses_path} probe {ef.path.parts[-1]}: {label}')
        else:
            type_probe = type or DEFAULT_TYPE
        timestamps, qc = sync_probe_front_times(
            sync_probe.times[:sync_idx], sync_nidq.times[:sync_idx], sr, display=display, type=type_probe, tol=tol
        )
        qc_all &= qc
        out_files.extend(_save_timestamps_npy(ef, timestamps, sr))
    return qc_all, out_files


def sync_probe_front_times(t, tref, sr, display=False, type='smooth', tol=2.0):
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
    """
    the main drift is computed through linear regression. A further step compute a smoothed
    version of the residual to add to the linear drift. The precision is enforced
    by ensuring that each point lies less than one sampling rate away from the predicted.
    """
    pol = np.polyfit(t, tref, 1)  # higher order terms first: slope / int for linear
    residual = tref - np.polyval(pol, t)
    if type == 'smooth':
        """
        the interp function from camera fronts is not smooth due to the locking of detections
        to the sampling rate of digital channels. The residual is fit using frequency domain
        smoothing
        """
        import ibldsp.fourier

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
        res_filt = np.pad(res_upsamp, lpad, mode='median', stat_length=CAMERA_UPSAMPLING_RATE_HZ * STAT_LENGTH_SECS)
        fbounds = [0.001, 0.002]
        res_filt = ibldsp.fourier.lp(res_filt, 1 / CAMERA_UPSAMPLING_RATE_HZ, fbounds)[lpad[0] : -lpad[1]]
        tout = np.arange(0, np.max(tref) + SYNC_SAMPLING_RATE_SECS, 20)
        sync_points = np.c_[tout, np.polyval(pol, tout) + np.interp(tout, t_upsamp, res_filt)]
        if display:
            if isinstance(display, matplotlib.axes.Axes):
                ax = display
            else:
                ax = plt.axes()
            ax.plot(tref, residual * sr, label='residual')
            ax.plot(t_upsamp, res_filt * sr, label='smoothed residual')
            ax.plot(tout, np.interp(tout, t_upsamp, res_filt) * sr, '*', label='interp timestamps')
            ax.legend()
            ax.set_xlabel('time (sec)')
            ax.set_ylabel('Residual drift (samples @ 30kHz)')
    elif type == 'exact':
        sync_points = np.c_[t, tref]
        if display:
            plt.plot(tref, residual * sr, label='residual')
            plt.ylabel('Residual drift (samples @ 30kHz)')
            plt.xlabel('time (sec)')
            pass
    elif type == 'linear':
        sync_points = np.c_[np.array([0.0, 1.0]), np.polyval(pol, np.array([0.0, 1.0]))]
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
        # plt.plot( (sync_points[:, 0] - fcn(sync_points[:, 1])) * sr)
    return sync_points, qc


def _get_sr(ephys_file):
    meta = spikeglx.read_meta_data(ephys_file.ap.with_suffix('.meta'))
    return spikeglx._get_fs_from_meta(meta)


def _save_timestamps_npy(ephys_file, tself_tref, sr):
    # this is the file with self_time_secs, ref_time_secs output
    file_sync = ephys_file.ap.parent.joinpath(ephys_file.ap.name.replace('.ap.', '.sync.')).with_suffix('.npy')
    np.save(file_sync, tself_tref)
    # this is the timestamps file
    file_ts = ephys_file.ap.parent.joinpath(ephys_file.ap.name.replace('.ap.', '.timestamps.')).with_suffix('.npy')
    timestamps = np.copy(tself_tref)
    timestamps[:, 0] *= np.float64(sr)
    np.save(file_ts, timestamps)
    return [file_sync, file_ts]


def classify_sync_anomaly(times, block=20, burst_factor=0.2, glitch_abs_thresh=0.005):
    """
    Classify a raw-pulse pathology from a single channel's front times, independent of
    whichever knot representation ('smooth'/'exact') ends up chosen downstream.

    Three independent checks, in order of precedence:

    - ``dropped_edges``: a sustained pulse-rate change (block-median inter-pulse interval
      drifts more than 30% from the global median, and stays drifted through to the end --
      not a transient blip). Typically real edges dropped partway through the recording.
    - ``duplicate_burst``: an inter-pulse interval far shorter than nominal (< ``burst_factor``
      times the median). Typically a bounced/double-triggered edge.
    - ``single_edge_glitch``: one isolated interval deviating from the median by more than
      ``glitch_abs_thresh`` in absolute terms (not relative -- real cases found this way
      ranged from 2.4% to 28% of the nominal period, too variable for a relative threshold).

    Parameters
    ----------
    times : array_like
        Front times (seconds) of a single channel, one polarity.
    block : int
        Number of consecutive intervals averaged per block for the `dropped_edges` check.
    burst_factor : float
        Fraction of the median interval below which an interval is flagged as a burst.
    glitch_abs_thresh : float
        Absolute deviation (seconds) from the median interval above which a single interval
        is flagged as an isolated glitch.

    Returns
    -------
    str or None
        One of 'dropped_edges', 'duplicate_burst', 'single_edge_glitch', or None (clean).
    """
    dt = np.diff(np.asarray(times))
    med = np.median(dt)
    if med <= 0:
        return None

    if dt.size >= block * 3:
        block_med = np.array([np.median(dt[i:i + block]) for i in range(0, dt.size - block, block)])
        bad = np.where(np.abs(block_med - med) / med > 0.3)[0]
        if bad.size > 0 and bad[-1] >= len(block_med) - 2:  # sustained to the end, not a blip
            return 'dropped_edges'

    if np.any(dt < burst_factor * med):
        return 'duplicate_burst'

    if np.max(np.abs(dt - med)) > glitch_abs_thresh:
        return 'single_edge_glitch'

    return None


def _check_diff_3b(sync):
    """
    Checks that the diff between consecutive sync pulses is below 150 PPM
    Returns True on a pass result (all values below threshold)
    """
    THRESH_PPM = 150
    d = np.diff(sync.times[sync.polarities == 1])
    dt = np.median(d)
    qc_pass = np.all(np.abs((d - dt) / dt * 1e6) < THRESH_PPM)
    if not qc_pass:
        _logger.error(
            f'Synchronizations bursts over {THRESH_PPM} ppm between sync pulses. Sync using "exact" match between pulses.'
        )
    return qc_pass

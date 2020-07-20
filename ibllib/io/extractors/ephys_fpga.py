from collections import OrderedDict
import logging
from pathlib import Path, PureWindowsPath
import uuid

import matplotlib.pyplot as plt
import numpy as np
from pkg_resources import parse_version
from scipy import interpolate

import alf.io
from brainbox.core import Bunch
import ibllib.dsp as dsp
import ibllib.exceptions as err
from ibllib.io import raw_data_loaders, spikeglx
from ibllib.io.extractors import biased_trials
from ibllib.io.extractors.base import (
    BaseBpodTrialsExtractor,
    BaseExtractor,
    run_extractor_classes,
)
from ibllib.io.extractors.training_wheel import extract_wheel_moves
import ibllib.plots as plots

_logger = logging.getLogger('ibllib')

SYNC_BATCH_SIZE_SAMPLES = 2 ** 18  # number of samples to read at once in bin file for sync
WHEEL_RADIUS_CM = 1  # stay in radians
WHEEL_TICKS = 1024

BPOD_FPGA_DRIFT_THRESHOLD_PPM = 150

CHMAPS = {'3A':
          {'ap':
           {'left_camera': 2,
            'right_camera': 3,
            'body_camera': 4,
            'bpod': 7,
            'frame2ttl': 12,
            'rotary_encoder_0': 13,
            'rotary_encoder_1': 14,
            'audio': 15
            }
           },
          '3B':
          {'nidq':
           {'left_camera': 0,
            'right_camera': 1,
            'body_camera': 2,
            'imec_sync': 3,
            'frame2ttl': 4,
            'rotary_encoder_0': 5,
            'rotary_encoder_1': 6,
            'audio': 7,
            'bpod': 16},
           'ap':
           {'imec_sync': 6}
           },
          }


def get_ibl_sync_map(ef, version):
    """
    Gets default channel map for the version/binary file type combination
    :param ef: ibllib.io.spikeglx.glob_ephys_file dictionary with field 'ap' or 'nidq'
    :return: channel map dictionary
    """
    if version == '3A':
        default_chmap = CHMAPS['3A']['ap']
    elif version == '3B':
        if ef.get('nidq', None):
            default_chmap = CHMAPS['3B']['nidq']
        elif ef.get('ap', None):
            default_chmap = CHMAPS['3B']['ap']
    return spikeglx.get_sync_map(ef['path']) or default_chmap


def _sync_to_alf(raw_ephys_apfile, output_path=None, save=False, parts=''):
    """
    Extracts sync.times, sync.channels and sync.polarities from binary ephys dataset

    :param raw_ephys_apfile: bin file containing ephys data or spike
    :param output_path: output directory
    :param save: bool write to disk only if True
    :param parts: string or list of strings that will be appended to the filename before extension
    :return:
    """
    # handles input argument: support ibllib.io.spikeglx.Reader, str and pathlib.Path
    if isinstance(raw_ephys_apfile, spikeglx.Reader):
        sr = raw_ephys_apfile
    else:
        raw_ephys_apfile = Path(raw_ephys_apfile)
        sr = spikeglx.Reader(raw_ephys_apfile)
    # if no output, need a temp folder to swap for big files
    if not output_path:
        output_path = raw_ephys_apfile.parent
    file_ftcp = Path(output_path).joinpath(f'fronts_times_channel_polarity{str(uuid.uuid4())}.bin')

    # loop over chunks of the raw ephys file
    wg = dsp.WindowGenerator(sr.ns, SYNC_BATCH_SIZE_SAMPLES, overlap=1)
    fid_ftcp = open(file_ftcp, 'wb')
    for sl in wg.slice:
        ss = sr.read_sync(sl)
        ind, fronts = dsp.fronts(ss, axis=0)
        # a = sr.read_sync_analog(sl)
        sav = np.c_[(ind[0, :] + sl.start) / sr.fs, ind[1, :], fronts.astype(np.double)]
        sav.tofile(fid_ftcp)
        # print progress
        wg.print_progress()
    # close temp file, read from it and delete
    fid_ftcp.close()
    tim_chan_pol = np.fromfile(str(file_ftcp))
    tim_chan_pol = tim_chan_pol.reshape((int(tim_chan_pol.size / 3), 3))
    file_ftcp.unlink()
    sync = {'times': tim_chan_pol[:, 0],
            'channels': tim_chan_pol[:, 1],
            'polarities': tim_chan_pol[:, 2]}
    if save:
        out_files = alf.io.save_object_npy(output_path, sync, '_spikeglx_sync', parts=parts)
        return Bunch(sync), out_files
    else:
        return Bunch(sync)


def _assign_events_bpod(bpod_t, bpod_polarities, ignore_first_valve=True):
    """
    From detected fronts on the bpod sync traces, outputs the synchronisation events
    related to trial start and valve opening
    :param bpod_t: numpy vector containing times of fronts
    :param bpod_fronts: numpy vector containing polarity of fronts (1 rise, -1 fall)
    :param ignore_first_valve (True): removes detected valve events at indices le 2
    :return: numpy arrays of times t_trial_start, t_valve_open and t_iti_in
    """
    TRIAL_START_TTL_LEN = 2.33e-4
    ITI_TTL_LEN = 0.4
    # make sure that there are no 2 consecutive fall or consecutive rise events
    assert(np.all(np.abs(np.diff(bpod_polarities)) == 2))
    # make sure that the first event is a rise
    assert(bpod_polarities[0] == 1)
    # take only even time differences: ie. from rising to falling fronts
    dt = np.diff(bpod_t)[::2]
    # detect start trials event assuming length is 0.23 ms except the first trial
    i_trial_start = np.r_[0, np.where(dt <= TRIAL_START_TTL_LEN)[0] * 2]
    t_trial_start = bpod_t[i_trial_start]
    # # the first trial we detect the first falling edge to which we subtract 0.1ms
    # t_trial_start[0] -= 1e-4
    # the last trial is a dud and should be removed
    t_trial_start = t_trial_start[:-1]
    # valve open events are between 50ms to 300 ms
    i_valve_open = np.where(np.logical_and(dt > TRIAL_START_TTL_LEN,
                                           dt < ITI_TTL_LEN))[0] * 2
    if ignore_first_valve:
        i_valve_open = np.delete(i_valve_open, np.where(i_valve_open < 2))
    t_valve_open = bpod_t[i_valve_open]
    # ITI events are above 400 ms
    i_iti_in = np.where(dt > ITI_TTL_LEN)[0] * 2
    i_iti_in = np.delete(i_iti_in, np.where(i_valve_open < 2))
    i_iti_in = bpod_t[i_iti_in]
    # # some debug plots when needed
    # import matplotlib.pyplot as plt
    # import ibllib.plots as plots
    # plt.figure()
    # plots.squares(bpod_t, bpod_fronts)
    # plots.vertical_lines(t_valve_open, ymin=-0.2, ymax=1.2, linewidth=0.5, color='g')
    # plots.vertical_lines(t_trial_start, ymin=-0.2, ymax=1.2, linewidth=0.5, color='r')
    return t_trial_start, t_valve_open, i_iti_in


def _rotary_encoder_positions_from_fronts(ta, pa, tb, pb, ticks=WHEEL_TICKS, radius=1,
                                          coding='x4'):
    """
    Extracts the rotary encoder absolute position as function of time from fronts detected
    on the 2 channels. Outputs in units of radius parameters, by default radians
    Coding options detailed here: http://www.ni.com/tutorial/7109/pt/
    Here output is clockwise from subject perspective

    :param ta: time of fronts on channel A
    :param pa: polarity of fronts on channel A
    :param tb: time of fronts on channel B
    :param pb: polarity of fronts on channel B
    :param ticks: number of ticks corresponding to a full revolution (1024 for IBL rotary encoder)
    :param radius: radius of the wheel. Defaults to 1 for an output in radians
    :param coding: x1, x2 or x4 coding (IBL default is x4)
    :return: indices vector (ta) and position vector
    """
    if coding == 'x1':
        ia = np.searchsorted(tb, ta[pa == 1])
        ia = ia[ia < ta.size]
        ia = ia[pa[ia] == 1]
        ib = np.searchsorted(ta, tb[pb == 1])
        ib = ib[ib < tb.size]
        ib = ib[pb[ib] == 1]
        t = np.r_[ta[ia], tb[ib]]
        p = np.r_[ia * 0 + 1, ib * 0 - 1]
        ordre = np.argsort(t)
        t = t[ordre]
        p = p[ordre]
        p = np.cumsum(p) / ticks * np.pi * 2 * radius
        return t, p
    elif coding == 'x2':
        p = pb[np.searchsorted(tb, ta) - 1] * pa
        p = - np.cumsum(p) / ticks * np.pi * 2 * radius / 2
        return ta, p
    elif coding == 'x4':
        p = np.r_[pb[np.searchsorted(tb, ta) - 1] * pa, -pa[np.searchsorted(ta, tb) - 1] * pb]
        t = np.r_[ta, tb]
        ordre = np.argsort(t)
        t = t[ordre]
        p = p[ordre]
        p = - np.cumsum(p) / ticks * np.pi * 2 * radius / 4
        return t, p


def _assign_events_audio(audio_t, audio_polarities, return_indices=False):
    """
    From detected fronts on the audio sync traces, outputs the synchronisation events
    related to tone in

    :param audio_t: numpy vector containing times of fronts
    :param audio_fronts: numpy vector containing polarity of fronts (1 rise, -1 fall)
    :param return_indices (False): returns indices of tones
    :return: numpy arrays t_ready_tone_in, t_error_tone_in
    :return: numpy arrays ind_ready_tone_in, ind_error_tone_in if return_indices=True
    """
    # make sure that there are no 2 consecutive fall or consecutive rise events
    assert(np.all(np.abs(np.diff(audio_polarities)) == 2))
    # take only even time differences: ie. from rising to falling fronts
    dt = np.diff(audio_t)[::2]
    # detect ready tone by length below 110 ms
    i_ready_tone_in = np.r_[np.where(dt <= 0.11)[0] * 2]
    t_ready_tone_in = audio_t[i_ready_tone_in]
    # error tones are events lasting from 400ms to 600ms
    i_error_tone_in = np.where(np.logical_and(0.4 < dt, dt < 1.2))[0] * 2
    t_error_tone_in = audio_t[i_error_tone_in]
    if return_indices:
        return t_ready_tone_in, t_error_tone_in, i_ready_tone_in, i_error_tone_in
    else:
        return t_ready_tone_in, t_error_tone_in


def _frame2ttl_events_extraction(f2ttl_t, f2ttl_fronts):
    pass


def _assign_events_to_trial(t_trial_start, t_event, take='last'):
    """
    Assign events to a trial given trial start times and event times.

    Trials without an event
    result in nan value in output time vector.
    The output has a consistent size with t_trial_start and ready to output to alf.

    :param t_trial_start: numpy vector of trial start times
    :param t_event: numpy vector of event times to assign to trials
    :param take: 'last' or 'first' (optional, default 'last'): index to take in case of duplicates
    :return: numpy array of event times with the same shape of trial start.
    """
    # make sure the events are sorted
    try:
        assert(np.all(np.diff(t_trial_start) >= 0))
    except AssertionError:
        raise ValueError('Trial starts vector not sorted')
    try:
        assert(np.all(np.diff(t_event) >= 0))
    except AssertionError:
        raise ValueError('Events vector is not sorted')
    # remove events that happened before the first trial start
    t_event = t_event[t_event >= t_trial_start[0]]
    ind = np.searchsorted(t_trial_start, t_event) - 1
    t_event_nans = np.zeros_like(t_trial_start) * np.nan
    # select first or last element matching each trial start
    if take == 'last':
        iall, iu = np.unique(np.flip(ind), return_index=True)
        t_event_nans[iall] = t_event[- (iu - ind.size + 1)]
    elif take == 'first':
        iall, iu = np.unique(ind, return_index=True)
        t_event_nans[iall] = t_event[iu]

    return t_event_nans


def _get_sync_fronts(sync, channel_nb, tmin=None, tmax=None):
    selection = sync['channels'] == channel_nb
    selection = np.logical_and(selection, sync['times'] <= tmax) if tmax else selection
    selection = np.logical_and(selection, sync['times'] >= tmin) if tmin else selection
    return Bunch({'times': sync['times'][selection],
                  'polarities': sync['polarities'][selection]})


def bpod_fpga_sync(bpod_intervals=None, ephys_intervals=None, iti_duration=None):
    """
    Computes synchronization function from bpod to fpga
    :param bpod_intervals
    :param ephys_intervals
    :return: interpolation function
    """
    if iti_duration is None:
        iti_duration = 0.5
    # check consistency
    if bpod_intervals.size != ephys_intervals.size:
        # patching things up if the bpod and FPGA don't have the same recording span
        _logger.warning("BPOD/FPGA synchronization: Bpod and FPGA don't have the same amount of"
                        " trial start events. Patching alf files.")
        _, _, ibpod, ifpga = raw_data_loaders.sync_trials_robust(
            bpod_intervals[:, 0], ephys_intervals[:, 0], return_index=True)
        if ibpod.size == 0:
            raise err.SyncBpodFpgaException('Can not sync BPOD and FPGA - no matching sync pulses '
                                            'found.')
        bpod_intervals = bpod_intervals[ibpod, :]
        ephys_intervals = ephys_intervals[ifpga, :]
    else:
        ibpod, ifpga = [np.arange(bpod_intervals.shape[0]) for _ in np.arange(2)]
    tlen = (np.diff(bpod_intervals) - np.diff(ephys_intervals))[:-1] - iti_duration
    assert np.all(np.abs(tlen[np.invert(np.isnan(tlen))])[:-1] < 5 * 1e-3)
    # dt is the delta to apply to bpod times in order to be on the ephys clock
    dt = bpod_intervals[:, 0] - ephys_intervals[:, 0]
    # compute the clock drift bpod versus dt
    ppm = np.polyfit(bpod_intervals[:, 0], dt, 1)[0] * 1e6
    if ppm > BPOD_FPGA_DRIFT_THRESHOLD_PPM:
        _logger.warning(
            'BPOD/FPGA synchronization shows values greater than %i ppm',
            BPOD_FPGA_DRIFT_THRESHOLD_PPM)
        # plt.plot(trials['intervals'][:, 0], dt, '*')
    # so far 2 datasets concerned: goCueTrigger_times_bpod  and response_times_bpod
    fcn_bpod2fpga = interpolate.interp1d(bpod_intervals[:, 0], ephys_intervals[:, 0],
                                         fill_value="extrapolate")
    return ibpod, ifpga, fcn_bpod2fpga


def extract_camera_sync(sync, chmap=None):
    """
    Extract camera timestamps from the sync matrix

    :param sync: dictionary 'times', 'polarities' of fronts detected on sync trace
    :param chmap: dictionary containing channel indices. Default to constant.
    :return: dictionary containing camera timestamps
    """
    # NB: should we check we opencv the expected number of frames ?
    assert(chmap)
    sr = _get_sync_fronts(sync, chmap['right_camera'])
    sl = _get_sync_fronts(sync, chmap['left_camera'])
    sb = _get_sync_fronts(sync, chmap['body_camera'])
    return {'right_camera': sr.times[::2],
            'left_camera': sl.times[::2],
            'body_camera': sb.times[::2]}


def extract_wheel_sync(sync, chmap=None):
    """
    Extract wheel positions and times from sync fronts dictionary for all 16 chans
    Output position is in radians, mathematical convention
    :param sync: dictionary 'times', 'polarities' of fronts detected on sync trace
    :param chmap: dictionary containing channel indices. Default to constant.
        chmap = {'rotary_encoder_0': 13, 'rotary_encoder_1': 14}
    :return: timestamps (np.array)
    :return: positions (np.array)
    """
    wheel = {}
    channela = _get_sync_fronts(sync, chmap['rotary_encoder_0'])
    channelb = _get_sync_fronts(sync, chmap['rotary_encoder_1'])
    wheel['re_ts'], wheel['re_pos'] = _rotary_encoder_positions_from_fronts(
        channela['times'], channela['polarities'], channelb['times'], channelb['polarities'],
        ticks=WHEEL_TICKS, radius=1, coding='x4')
    return wheel['re_ts'], wheel['re_pos']


def extract_behaviour_sync(sync, chmap=None, display=False, tmax=np.inf):
    """
    Extract wheel positions and times from sync fronts dictionary

    :param sync: dictionary 'times', 'polarities' of fronts detected on sync trace for all 16 chans
    :param chmap: dictionary containing channel index. Default to constant.
        chmap = {'bpod': 7, 'frame2ttl': 12, 'audio': 15}
    :param display: bool or matplotlib axes: show the full session sync pulses display
    defaults to False
    :return: trials dictionary
    """
    bpod = _get_sync_fronts(sync, chmap['bpod'], tmax=tmax)
    if bpod.times.size == 0:
        raise err.SyncBpodFpgaException('No Bpod event found in FPGA. No behaviour extraction. '
                                        'Check channel maps.')
    frame2ttl = _get_sync_fronts(sync, chmap['frame2ttl'], tmax=tmax)
    audio = _get_sync_fronts(sync, chmap['audio'], tmax=tmax)
    # extract events from the fronts for each trace
    t_trial_start, t_valve_open, t_iti_in = _assign_events_bpod(
        bpod['times'], bpod['polarities'])
    t_ready_tone_in, t_error_tone_in = _assign_events_audio(
        audio['times'], audio['polarities'])
    # stim off time is the first frame2ttl rise/fall after the trial start
    # does not apply for 1st trial
    ind = np.searchsorted(frame2ttl['times'], t_iti_in, side='left')
    t_stim_off = frame2ttl['times'][np.minimum(ind, frame2ttl.times.size - 1)]
    t_stim_freeze = frame2ttl['times'][np.minimum(ind, frame2ttl.times.size - 2)]
    # t_stim_freeze = frame2ttl['times'][np.maximum(ind - 1, 0)]
    # stimOn_times: first fram2ttl change after trial start
    trials = Bunch({
        'goCue_times': _assign_events_to_trial(t_trial_start, t_ready_tone_in, take='first'),
        'errorCue_times': _assign_events_to_trial(t_trial_start, t_error_tone_in),
        'valveOpen_times': _assign_events_to_trial(t_trial_start, t_valve_open),
        'stimFreeze_times': _assign_events_to_trial(t_trial_start, t_stim_freeze),
        'stimOn_times': _assign_events_to_trial(t_trial_start, frame2ttl['times'], take='first'),
        'stimOff_times': _assign_events_to_trial(t_trial_start, t_stim_off),
        'itiIn_times': _assign_events_to_trial(t_trial_start, t_iti_in)
    })
    # feedback times are valve open on good trials and error tone in on error trials
    trials['feedback_times'] = np.copy(trials['valveOpen_times'])
    ind_err = np.isnan(trials['valveOpen_times'])
    trials['feedback_times'][ind_err] = trials['errorCue_times'][ind_err]
    trials['intervals'] = np.c_[t_trial_start, trials['itiIn_times']]

    if display:
        width = 0.5
        ymax = 5
        if isinstance(display, bool):
            plt.figure("Ephys FPGA Sync")
            ax = plt.gca()
        else:
            ax = display
        r0 = _get_sync_fronts(sync, chmap['rotary_encoder_0'])
        plots.squares(bpod['times'], bpod['polarities'] * 0.4 + 1,
                      ax=ax, color='k')
        plots.squares(frame2ttl['times'], frame2ttl['polarities'] * 0.4 + 2,
                      ax=ax, color='k')
        plots.squares(audio['times'], audio['polarities'] * 0.4 + 3,
                      ax=ax, color='k')
        plots.squares(r0['times'], r0['polarities'] * 0.4 + 4,
                      ax=ax, color='k')
        plots.vertical_lines(t_ready_tone_in, ymin=0, ymax=ymax,
                             ax=ax, label='goCue_times', color='b', linewidth=width)
        plots.vertical_lines(t_trial_start, ymin=0, ymax=ymax,
                             ax=ax, label='start_trial', color='m', linewidth=width)
        plots.vertical_lines(t_error_tone_in, ymin=0, ymax=ymax,
                             ax=ax, label='error tone', color='r', linewidth=width)
        plots.vertical_lines(t_valve_open, ymin=0, ymax=ymax,
                             ax=ax, label='valveOpen_times', color='g', linewidth=width)
        plots.vertical_lines(t_stim_freeze, ymin=0, ymax=ymax,
                             ax=ax, label='stimFreeze_times', color='y', linewidth=width)
        plots.vertical_lines(t_stim_off, ymin=0, ymax=ymax,
                             ax=ax, label='stim off', color='c', linewidth=width)
        plots.vertical_lines(trials['stimOn_times'], ymin=0, ymax=ymax,
                             ax=ax, label='stimOn_times', color='tab:orange', linewidth=width)
        ax.legend()
        ax.set_yticklabels(['', 'bpod', 'f2ttl', 'audio', 're_0', ''])
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_ylim([0, 5])

    return trials


def extract_sync(session_path, overwrite=False, ephys_files=None):
    """
    Reads ephys binary file (s) and extract sync within the binary file folder
    Assumes ephys data is within a `raw_ephys_data` folder

    :param session_path: '/path/to/subject/yyyy-mm-dd/001'
    :param overwrite: Bool on re-extraction, forces overwrite instead of loading existing files
    :return: list of sync dictionaries
    """
    session_path = Path(session_path)
    if not ephys_files:
        ephys_files = spikeglx.glob_ephys_files(session_path)
    syncs = []
    outputs = []
    for efi in ephys_files:
        glob_filter = f'*{efi.label}*' if efi.label else '*'
        bin_file = efi.get('ap', efi.get('nidq', None))
        if not bin_file:
            continue
        file_exists = alf.io.exists(bin_file.parent, object='_spikeglx_sync', glob=glob_filter)
        if not overwrite and file_exists:
            _logger.warning(f'Skipping raw sync: SGLX sync found for probe {efi.label} !')
            sync = alf.io.load_object(bin_file.parent, object='_spikeglx_sync', glob=glob_filter)
            out_files, _ = alf.io._ls(bin_file.parent, object='_spikeglx_sync', glob=glob_filter)
        else:
            sr = spikeglx.Reader(bin_file)
            sync, out_files = _sync_to_alf(sr, bin_file.parent, save=True, parts=efi.label)
        outputs.extend(out_files)
        syncs.extend([sync])

    return syncs, outputs


def _get_all_probes_sync(session_path, bin_exists=True):
    # round-up of all bin ephys files in the session, infer revision and get sync map
    ephys_files = spikeglx.glob_ephys_files(session_path, bin_exists=bin_exists)
    version = spikeglx.get_neuropixel_version_from_files(ephys_files)
    # attach the sync information to each binary file found
    for ef in ephys_files:
        ef['sync'] = alf.io.load_object(ef.path, '_spikeglx_sync', short_keys=True)
        ef['sync_map'] = get_ibl_sync_map(ef, version)
    return ephys_files


def _get_main_probe_sync(session_path, bin_exists=False):
    """
    From 3A or 3B multiprobe session, returns the main probe (3A) or nidq sync pulses
    with the attached channel map (default chmap if none)
    :param session_path:
    :return:
    """
    ephys_files = _get_all_probes_sync(session_path, bin_exists=bin_exists)
    if not ephys_files:
        raise FileNotFoundError(f"No ephys files found in {session_path}")
    version = spikeglx.get_neuropixel_version_from_files(ephys_files)
    if version == '3A':
        # the sync master is the probe with the most sync pulses
        sync_box_ind = np.argmax([ef.sync.times.size for ef in ephys_files])
    elif version == '3B':
        # the sync master is the nidq breakout box
        sync_box_ind = np.argmax([1 if ef.get('nidq') else 0 for ef in ephys_files])

    sync = ephys_files[sync_box_ind].sync
    sync_chmap = ephys_files[sync_box_ind].sync_map
    return sync, sync_chmap


def _get_pregenerated_events(bpod_trials, settings):
    num = settings.get("PRELOADED_SESSION_NUM", None)
    if num is None:
        num = settings.get("PREGENERATED_SESSION_NUM", None)
    if num is None:
        fn = settings.get('SESSION_LOADED_FILE_PATH', None)
        fn = PureWindowsPath(fn).name
        num = ''.join([d for d in fn if d.isdigit()])
        if num == '':
            raise ValueError("Can't extract left probability behaviour.")
    # Load the pregenerated file
    ntrials = len(bpod_trials)
    sessions_folder = Path(raw_data_loaders.__file__).parent.joinpath(
        "extractors", "ephys_sessions")
    fname = f"session_{num}_ephys_pcqs.npy"
    pcqsp = np.load(sessions_folder.joinpath(fname))
    pos = pcqsp[:, 0]
    con = pcqsp[:, 1]
    pos = pos[: ntrials]
    con = con[: ntrials]
    contrastRight = con.copy()
    contrastLeft = con.copy()
    contrastRight[pos < 0] = np.nan
    contrastLeft[pos > 0] = np.nan
    qui = pcqsp[:, 2]
    qui = qui[: ntrials]
    phase = pcqsp[:, 3]
    phase = phase[: ntrials]
    pLeft = pcqsp[:, 4]
    pLeft = pLeft[: ntrials]

    phase_path = sessions_folder.joinpath(f"session_{num}_stim_phase.npy")
    is_patched_version = parse_version(
        settings.get('IBLRIG_VERSION_TAG', 0)) > parse_version('6.4.0')
    if phase_path.exists() and is_patched_version:
        phase = np.load(phase_path)[:ntrials]

    return {"position": pos, "contrast": con, "quiescence": qui, "phase": phase,
            "prob_left": pLeft, 'contrast_right': contrastRight, 'contrast_left': contrastLeft}


class ProbaContrasts(BaseBpodTrialsExtractor):
    """
    Bpod pre-generated values for probabilityLeft, contrastLR, phase, quiescence
    """
    save_names = ('_ibl_trials.probabilityLeft.npy', '_ibl_trials.contrastLeft.npy',
                  '_ibl_trials.contrastRight.npy')
    var_names = ('probabilityLeft', 'contrastLeft', 'contrastRight')

    def _extract(self):
        """Extracts positions, contrasts, quiescent delay, stimulus phase and probability left
        from pregenerated session files.
        Optional: saves alf contrastLR and probabilityLeft npy files"""
        pe = _get_pregenerated_events(self.bpod_trials, self.settings)
        return pe['prob_left'], pe['contrast_left'], pe['contrast_right']


class CameraTimestamps(BaseExtractor):
    save_names = ['_ibl_rightCamera.times.npy', '_ibl_leftCamera.times.npy',
                  '_ibl_bodyCamera.times.npy']
    var_names = ['right_camera_timestamps', 'left_camera_timestamps', 'body_camera_timestamps']

    def _extract(self, sync=None, chmap=None):
        ts = extract_camera_sync(sync=sync, chmap=chmap)
        return ts['right_camera'], ts['left_camera'], ts['body_camera']


class FpgaTrials(BaseExtractor):
    save_names = ('_ibl_trials.probabilityLeft.npy', '_ibl_trials.contrastLeft.npy',
                  '_ibl_trials.contrastRight.npy', '_ibl_trials.feedbackType.npy',
                  '_ibl_trials.choice.npy', '_ibl_trials.rewardVolume.npy',
                  '_ibl_trials.intervals_bpod.npy', '_ibl_trials.intervals.npy',
                  '_ibl_trials.response_times.npy', '_ibl_trials.goCueTrigger_times.npy',
                  '_ibl_trials.stimOn_times.npy', '_ibl_trials.stimOff_times.npy',
                  '_ibl_trials.goCue_times.npy', '_ibl_trials.feedback_times.npy',
                  '_ibl_trials.firstMovement_times.npy', '_ibl_wheel.timestamps.npy',
                  '_ibl_wheel.position.npy', '_ibl_wheelMoves.intervals.npy',
                  '_ibl_wheelMoves.peakAmplitude.npy')
    var_names = ('probabilityLeft', 'contrastLeft', 'contrastRight', 'feedbackType', 'choice',
                 'rewardVolume', 'intervals_bpod', 'intervals', 'response_times',
                 'goCueTrigger_times', 'stimOn_times', 'stimOff_times', 'goCue_times',
                 'feedback_times', 'firstMovement_times', 'wheel_timestamps', 'wheel_position',
                 'wheelMoves_intervals', 'wheelMoves_peakAmplitude')

    def _extract(self, sync=None, chmap=None):
        # extracts trials
        # extract the behaviour data from bpod
        if sync is None or chmap is None:
            _sync, _chmap = _get_main_probe_sync(self.session_path, bin_exists=False)
            sync = sync or _sync
            chmap = chmap or _chmap
        bpod_raw = raw_data_loaders.load_data(self.session_path)
        tmax = bpod_raw[-1]['behavior_data']['States timestamps']['exit_state'][0][-1] + 60
        bpod_trials, _ = biased_trials.extract_all(
            session_path=self.session_path, save=False, bpod_trials=bpod_raw)
        bpod_trials['intervals_bpod'] = np.copy(bpod_trials['intervals'])
        fpga_trials = extract_behaviour_sync(sync=sync, chmap=chmap, tmax=tmax)
        # checks consistency and compute dt with bpod
        ibpod, ifpga, fcn_bpod2fpga = bpod_fpga_sync(
            bpod_trials['intervals_bpod'], fpga_trials['intervals'])
        # those fields get directly in the output
        bpod_fields = ['feedbackType', 'choice', 'rewardVolume', 'intervals_bpod']
        # those fields have to be resynced
        bpod_rsync_fields = ['intervals', 'response_times', 'goCueTrigger_times']
        # ephys fields to save in the output
        fpga_fields = ['stimOn_times', 'stimOff_times', 'goCue_times', 'feedback_times']
        # get ('probabilityLeft', 'contrastLeft', 'contrastRight') from the custom ephys extractors
        pclcr, _ = ProbaContrasts(self.session_path).extract(bpod_trials=bpod_raw, save=False)
        # build trials output
        out = OrderedDict()
        out.update({k: pclcr[i][ifpga] for i, k in enumerate(ProbaContrasts.var_names)})
        out.update({k: bpod_trials[k][ibpod] for k in bpod_fields})
        out.update({k: fcn_bpod2fpga(bpod_trials[k][ibpod]) for k in bpod_rsync_fields})
        out.update({k: fpga_trials[k][ifpga] for k in fpga_fields})

        # extract the wheel data
        from ibllib.io.extractors.training_wheel import extract_first_movement_times
        ts, pos = extract_wheel_sync(sync=sync, chmap=chmap)
        moves = extract_wheel_moves(ts, pos)
        settings = raw_data_loaders.load_settings(session_path=self.session_path)
        min_qt = settings.get('QUIESCENT_PERIOD', None)
        first_move_onsets, *_ = extract_first_movement_times(moves, out, min_qt=min_qt)
        out.update({'firstMovement_times': first_move_onsets})

        assert tuple(filter(lambda x: 'wheel' not in x, self.var_names)) == tuple(out.keys())
        return [out[k] for k in out] + [ts, pos, moves['intervals'], moves['peakAmplitude']]


def extract_all(session_path, save=True, bin_exists=False):
    """
    For the IBL ephys task, reads ephys binary file and extract:
        -   sync
        -   wheel
        -   behaviour
        -   video time stamps
    :param session_path: '/path/to/subject/yyyy-mm-dd/001'
    :param save: Bool, defaults to False
    :param version: bpod version, defaults to None
    :return: outputs, files
    """
    assert save  # fixme with wheel positions, this function can't work without saving the data
    sync, chmap = _get_main_probe_sync(session_path, bin_exists=bin_exists)
    outputs, files = run_extractor_classes(
        [CameraTimestamps, FpgaTrials], session_path=session_path,
        save=save, sync=sync, chmap=chmap)
    return outputs, files

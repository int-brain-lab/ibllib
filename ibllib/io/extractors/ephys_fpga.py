"""Data extraction from raw FPGA output
Complete FPGA data extraction depends on Bpod extraction
"""
from collections import OrderedDict
import logging
from pathlib import Path
import uuid

import matplotlib.pyplot as plt
import numpy as np

import spikeglx
import neurodsp.utils
import one.alf.io as alfio
from iblutil.util import Bunch
from iblutil.spacer import Spacer

import ibllib.exceptions as err
from ibllib.io import raw_data_loaders, session_params
from ibllib.io.extractors.bpod_trials import extract_all as bpod_extract_all
from ibllib.io.extractors.opto_trials import LaserBool
import ibllib.io.extractors.base as extractors_base
from ibllib.io.extractors.training_wheel import extract_wheel_moves
import ibllib.plots as plots
from ibllib.io.extractors.default_channel_maps import DEFAULT_MAPS

_logger = logging.getLogger(__name__)

SYNC_BATCH_SIZE_SECS = 100  # number of samples to read at once in bin file for sync
WHEEL_RADIUS_CM = 1  # stay in radians
WHEEL_TICKS = 1024

BPOD_FPGA_DRIFT_THRESHOLD_PPM = 150  # throws an error if bpod to fpga clock drift is higher
F2TTL_THRESH = 0.01  # consecutive pulses with less than this threshold ignored

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
            'bpod': 16,
            'laser': 17,
            'laser_ttl': 18},
           'ap':
           {'imec_sync': 6}
           },
          }


def data_for_keys(keys, data):
    """Check keys exist in 'data' dict and contain values other than None"""
    return data is not None and all(k in data and data.get(k, None) is not None for k in keys)


def get_ibl_sync_map(ef, version):
    """
    Gets default channel map for the version/binary file type combination
    :param ef: ibllib.io.spikeglx.glob_ephys_file dictionary with field 'ap' or 'nidq'
    :return: channel map dictionary
    """
    # Determine default channel map
    if version == '3A':
        default_chmap = CHMAPS['3A']['ap']
    elif version == '3B':
        if ef.get('nidq', None):
            default_chmap = CHMAPS['3B']['nidq']
        elif ef.get('ap', None):
            default_chmap = CHMAPS['3B']['ap']
    # Try to load channel map from file
    chmap = spikeglx.get_sync_map(ef['path'])
    # If chmap provided but not with all keys, fill up with default values
    if not chmap:
        return default_chmap
    else:
        if data_for_keys(default_chmap.keys(), chmap):
            return chmap
        else:
            _logger.warning("Keys missing from provided channel map, "
                            "setting missing keys from default channel map")
            return {**default_chmap, **chmap}


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
    opened = sr.is_open
    if not opened:  # if not (opened := sr.is_open)  # py3.8
        sr.open()
    # if no output, need a temp folder to swap for big files
    if not output_path:
        output_path = raw_ephys_apfile.parent
    file_ftcp = Path(output_path).joinpath(f'fronts_times_channel_polarity{str(uuid.uuid4())}.bin')

    # loop over chunks of the raw ephys file
    wg = neurodsp.utils.WindowGenerator(sr.ns, int(SYNC_BATCH_SIZE_SECS * sr.fs), overlap=1)
    fid_ftcp = open(file_ftcp, 'wb')
    for sl in wg.slice:
        ss = sr.read_sync(sl)
        ind, fronts = neurodsp.utils.fronts(ss, axis=0)
        # a = sr.read_sync_analog(sl)
        sav = np.c_[(ind[0, :] + sl.start) / sr.fs, ind[1, :], fronts.astype(np.double)]
        sav.tofile(fid_ftcp)
    # close temp file, read from it and delete
    fid_ftcp.close()
    tim_chan_pol = np.fromfile(str(file_ftcp))
    tim_chan_pol = tim_chan_pol.reshape((int(tim_chan_pol.size / 3), 3))
    file_ftcp.unlink()
    sync = {'times': tim_chan_pol[:, 0],
            'channels': tim_chan_pol[:, 1],
            'polarities': tim_chan_pol[:, 2]}
    # If opened Reader was passed into function, leave open
    if not opened:
        sr.close()
    if save:
        out_files = alfio.save_object_npy(output_path, sync, 'sync',
                                          namespace='spikeglx', parts=parts)
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
    TRIAL_START_TTL_LEN = 2.33e-4  # the TTL length is 0.1ms but this has proven to drift on
    # some bpods and this is the highest possible value that discriminates trial start from valve
    ITI_TTL_LEN = 0.4
    # make sure that there are no 2 consecutive fall or consecutive rise events
    assert np.all(np.abs(np.diff(bpod_polarities)) == 2)
    if bpod_polarities[0] == -1:
        bpod_t = np.delete(bpod_t, 0)
    # take only even time differences: ie. from rising to falling fronts
    dt = np.diff(bpod_t)[::2]
    # detect start trials event assuming length is 0.23 ms except the first trial
    i_trial_start = np.r_[0, np.where(dt <= TRIAL_START_TTL_LEN)[0] * 2]
    t_trial_start = bpod_t[i_trial_start]
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
    t_iti_in = bpod_t[i_iti_in]
    ## some debug plots when needed
    # import matplotlib.pyplot as plt
    # import ibllib.plots as plots
    # events = {'id': np.zeros(bpod_t.shape), 't': bpod_t, 'p': bpod_polarities}
    # events['id'][i_trial_start] = 1
    # events['id'][i_valve_open] = 2
    # events['id'][i_iti_in] = 3
    # i_abnormal = np.where(np.diff(events['id'][bpod_polarities != -1]) == 0)
    # t_abnormal = events['t'][bpod_polarities != -1][i_abnormal]
    # assert np.all(events != 0)
    # plt.figure()
    # plots.squares(bpod_t, bpod_polarities, label='raw fronts')
    # plots.vertical_lines(t_trial_start, ymin=-0.2, ymax=1.1, linewidth=0.5, label='trial start')
    # plots.vertical_lines(t_valve_open, ymin=-0.2, ymax=1.1, linewidth=0.5, label='valve open')
    # plots.vertical_lines(t_iti_in, ymin=-0.2, ymax=1.1, linewidth=0.5, label='iti_in')
    # plt.plot(t_abnormal, t_abnormal * 0 + .5, 'k*')
    # plt.legend()

    return t_trial_start, t_valve_open, t_iti_in


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


def _assign_events_audio(audio_t, audio_polarities, return_indices=False, display=False):
    """
    From detected fronts on the audio sync traces, outputs the synchronisation events
    related to tone in

    :param audio_t: numpy vector containing times of fronts
    :param audio_fronts: numpy vector containing polarity of fronts (1 rise, -1 fall)
    :param return_indices (False): returns indices of tones
    :param display (False): for debug mode, displays the raw fronts overlaid with detections
    :return: numpy arrays t_ready_tone_in, t_error_tone_in
    :return: numpy arrays ind_ready_tone_in, ind_error_tone_in if return_indices=True
    """
    # make sure that there are no 2 consecutive fall or consecutive rise events
    assert np.all(np.abs(np.diff(audio_polarities)) == 2)
    # take only even time differences: ie. from rising to falling fronts
    dt = np.diff(audio_t)
    # detect ready tone by length below 110 ms
    i_ready_tone_in = np.where(np.logical_and(dt <= 0.11, audio_polarities[:-1] == 1))[0]
    t_ready_tone_in = audio_t[i_ready_tone_in]
    # error tones are events lasting from 400ms to 1200ms
    i_error_tone_in = np.where(np.logical_and(np.logical_and(0.4 < dt, dt < 1.2), audio_polarities[:-1] == 1))[0]
    t_error_tone_in = audio_t[i_error_tone_in]
    if display:  # pragma: no cover
        from ibllib.plots import squares, vertical_lines
        squares(audio_t, audio_polarities, yrange=[-1, 1],)
        vertical_lines(t_ready_tone_in, ymin=-.8, ymax=.8)
        vertical_lines(t_error_tone_in, ymin=-.8, ymax=.8)

    if return_indices:
        return t_ready_tone_in, t_error_tone_in, i_ready_tone_in, i_error_tone_in
    else:
        return t_ready_tone_in, t_error_tone_in


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
        assert np.all(np.diff(t_trial_start) >= 0)
    except AssertionError:
        raise ValueError('Trial starts vector not sorted')
    try:
        assert np.all(np.diff(t_event) >= 0)
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
    else:  # if the index is arbitrary, needs to be numeric (could be negative if from the end)
        iall = np.unique(ind)
        minsize = take + 1 if take >= 0 else - take
        # for each trial, take the takenth element if there are enough values in trial
        for iu in iall:
            match = t_event[iu == ind]
            if len(match) >= minsize:
                t_event_nans[iu] = match[take]
    return t_event_nans


def get_sync_fronts(sync, channel_nb, tmin=None, tmax=None):
    """
    Return the sync front polarities and times for a given channel.

    Parameters
    ----------
    sync : dict
        'polarities' of fronts detected on sync trace for all 16 channels and their 'times'.
    channel_nb : int
        The integer corresponding to the desired sync channel.
    tmin : float
        The minimum time from which to extract the sync pulses.
    tmax : float
        The maximum time up to which we extract the sync pulses.

    Returns
    -------
    Bunch
        Channel times and polarities.
    """
    selection = sync['channels'] == channel_nb
    selection = np.logical_and(selection, sync['times'] <= tmax) if tmax else selection
    selection = np.logical_and(selection, sync['times'] >= tmin) if tmin else selection
    return Bunch({'times': sync['times'][selection],
                  'polarities': sync['polarities'][selection]})


def _clean_audio(audio, display=False):
    """
    one guy wired the 150 Hz camera output onto the soundcard. The effect is to get 150 Hz periodic
    square pulses, 2ms up and 4.666 ms down. When this happens we remove all of the intermediate
    pulses to repair the audio trace
    Here is some helper code
        dd = np.diff(audio['times'])
        1 / np.median(dd[::2]) # 2ms up
        1 / np.median(dd[1::2])  # 4.666 ms down
        1 / (np.median(dd[::2]) + np.median(dd[1::2])) # both sum to 150 Hx
    This only runs on sessions when the bug is detected and leaves others untouched
    """
    DISCARD_THRESHOLD = 0.01
    average_150_hz = np.mean(1 / np.diff(audio['times'][audio['polarities'] == 1]) > 140)
    naudio = audio['times'].size
    if average_150_hz > 0.7 and naudio > 100:
        _logger.warning("Soundcard signal on FPGA seems to have been mixed with 150Hz camera")
        keep_ind = np.r_[np.diff(audio['times']) > DISCARD_THRESHOLD, False]
        keep_ind = np.logical_and(keep_ind, audio['polarities'] == -1)
        keep_ind = np.where(keep_ind)[0]
        keep_ind = np.sort(np.r_[0, keep_ind, keep_ind + 1, naudio - 1])

        if display:  # pragma: no cover
            from ibllib.plots import squares
            squares(audio['times'], audio['polarities'], ax=None, yrange=[-1, 1])
            squares(audio['times'][keep_ind], audio['polarities'][keep_ind], yrange=[-1, 1])
        audio = {'times': audio['times'][keep_ind],
                 'polarities': audio['polarities'][keep_ind]}
    return audio


def _clean_frame2ttl(frame2ttl, display=False):
    """
    Frame 2ttl calibration can be unstable and the fronts may be flickering at an unrealistic
    pace. This removes the consecutive frame2ttl pulses happening too fast, below a threshold
    of F2TTL_THRESH
    """
    dt = np.diff(frame2ttl['times'])
    iko = np.where(np.logical_and(dt < F2TTL_THRESH, frame2ttl['polarities'][:-1] == -1))[0]
    iko = np.unique(np.r_[iko, iko + 1])
    frame2ttl_ = {'times': np.delete(frame2ttl['times'], iko),
                  'polarities': np.delete(frame2ttl['polarities'], iko)}
    if iko.size > (0.1 * frame2ttl['times'].size):
        _logger.warning(f'{iko.size} ({iko.size / frame2ttl["times"].size:.2%} %) '
                        f'frame to TTL polarity switches below {F2TTL_THRESH} secs')
    if display:  # pragma: no cover
        from ibllib.plots import squares
        plt.figure()
        squares(frame2ttl['times'] * 1000, frame2ttl['polarities'], yrange=[0.1, 0.9])
        squares(frame2ttl_['times'] * 1000, frame2ttl_['polarities'], yrange=[1.1, 1.9])
        import seaborn as sns
        sns.displot(dt[dt < 0.05], binwidth=0.0005)

    return frame2ttl_


def extract_wheel_sync(sync, chmap=None, tmin=None, tmax=None):
    """
    Extract wheel positions and times from sync fronts dictionary for all 16 channels.
    Output position is in radians, mathematical convention.

    Parameters
    ----------
    sync : dict
        'polarities' of fronts detected on sync trace for all 16 chans and their 'times'
    chmap : dict
        Map of channel names and their corresponding index.  Default to constant.
    tmin : float
        The minimum time from which to extract the sync pulses.
    tmax : float
        The maximum time up to which we extract the sync pulses.

    Returns
    -------
    np.array
        Wheel timestamps in seconds.
    np.array
        Wheel positions in radians.
    """
    wheel = {}
    channela = get_sync_fronts(sync, chmap['rotary_encoder_0'], tmin=tmin, tmax=tmax)
    channelb = get_sync_fronts(sync, chmap['rotary_encoder_1'], tmin=tmin, tmax=tmax)
    wheel['re_ts'], wheel['re_pos'] = _rotary_encoder_positions_from_fronts(
        channela['times'], channela['polarities'], channelb['times'], channelb['polarities'],
        ticks=WHEEL_TICKS, radius=1, coding='x4')
    return wheel['re_ts'], wheel['re_pos']


def extract_behaviour_sync(sync, chmap=None, display=False, bpod_trials=None, tmin=None, tmax=None):
    """
    Extract task related event times from the sync.

    Parameters
    ----------
    sync : dict
        'polarities' of fronts detected on sync trace for all 16 chans and their 'times'
    chmap : dict
        Map of channel names and their corresponding index.  Default to constant.
    display : bool, matplotlib.pyplot.Axes
        Show the full session sync pulses display
    bpod_trials : dict
        The same trial events as recorded through Bpod. Assumed to contain an 'intervals_bpod' key.
    tmin : float
        The minimum time from which to extract the sync pulses.
    tmax : float
        The maximum time up to which we extract the sync pulses.

    Returns
    -------
    dict
        A map of trial event timestamps.
    """
    bpod = get_sync_fronts(sync, chmap['bpod'], tmin=tmin, tmax=tmax)
    if bpod.times.size == 0:
        raise err.SyncBpodFpgaException('No Bpod event found in FPGA. No behaviour extraction. '
                                        'Check channel maps.')
    frame2ttl = get_sync_fronts(sync, chmap['frame2ttl'], tmin=tmin, tmax=tmax)
    frame2ttl = _clean_frame2ttl(frame2ttl)
    audio = get_sync_fronts(sync, chmap['audio'], tmin=tmin, tmax=tmax)
    audio = _clean_audio(audio)
    # extract events from the fronts for each trace
    t_trial_start, t_valve_open, t_iti_in = _assign_events_bpod(bpod['times'], bpod['polarities'])
    # one issue is that sometimes bpod pulses may not have been detected, in this case
    # perform the sync bpod/FPGA, and add the start that have not been detected
    if bpod_trials:
        bpod_start = bpod_trials['intervals_bpod'][:, 0]
        fcn, drift, ibpod, ifpga = neurodsp.utils.sync_timestamps(
            bpod_start, t_trial_start, return_indices=True)
        # if it's drifting too much
        if drift > 200 and bpod_start.size != t_trial_start.size:
            raise err.SyncBpodFpgaException('sync cluster f*ck')
        missing_bpod = fcn(bpod_start[np.setxor1d(ibpod, np.arange(len(bpod_start)))])
        t_trial_start = np.sort(np.r_[t_trial_start, missing_bpod])
    else:
        _logger.warning('Deprecation Warning: calling FPGA trials extraction without a bpod trials'
                        ' dictionary will result in an error.')
    t_ready_tone_in, t_error_tone_in = _assign_events_audio(
        audio['times'], audio['polarities'])
    trials = Bunch({
        'goCue_times': _assign_events_to_trial(t_trial_start, t_ready_tone_in, take='first'),
        'errorCue_times': _assign_events_to_trial(t_trial_start, t_error_tone_in),
        'valveOpen_times': _assign_events_to_trial(t_trial_start, t_valve_open),
        'stimFreeze_times': _assign_events_to_trial(t_trial_start, frame2ttl['times'], take=-2),
        'stimOn_times': _assign_events_to_trial(t_trial_start, frame2ttl['times'], take='first'),
        'stimOff_times': _assign_events_to_trial(t_trial_start, frame2ttl['times']),
        'itiIn_times': _assign_events_to_trial(t_trial_start, t_iti_in)
    })
    # feedback times are valve open on good trials and error tone in on error trials
    trials['feedback_times'] = np.copy(trials['valveOpen_times'])
    ind_err = np.isnan(trials['valveOpen_times'])
    trials['feedback_times'][ind_err] = trials['errorCue_times'][ind_err]
    trials['intervals'] = np.c_[t_trial_start, trials['itiIn_times']]

    if display:  # pragma: no cover
        width = 0.5
        ymax = 5
        if isinstance(display, bool):
            plt.figure("Ephys FPGA Sync")
            ax = plt.gca()
        else:
            ax = display
        r0 = get_sync_fronts(sync, chmap['rotary_encoder_0'], tmin=tmin, tmax=tmax)
        plots.squares(bpod['times'], bpod['polarities'] * 0.4 + 1, ax=ax, color='k')
        plots.squares(frame2ttl['times'], frame2ttl['polarities'] * 0.4 + 2, ax=ax, color='k')
        plots.squares(audio['times'], audio['polarities'] * 0.4 + 3, ax=ax, color='k')
        plots.squares(r0['times'], r0['polarities'] * 0.4 + 4, ax=ax, color='k')
        plots.vertical_lines(t_ready_tone_in, ymin=0, ymax=ymax,
                             ax=ax, label='goCue_times', color='b', linewidth=width)
        plots.vertical_lines(t_trial_start, ymin=0, ymax=ymax,
                             ax=ax, label='start_trial', color='m', linewidth=width)
        plots.vertical_lines(t_error_tone_in, ymin=0, ymax=ymax,
                             ax=ax, label='error tone', color='r', linewidth=width)
        plots.vertical_lines(t_valve_open, ymin=0, ymax=ymax,
                             ax=ax, label='valveOpen_times', color='g', linewidth=width)
        plots.vertical_lines(trials['stimFreeze_times'], ymin=0, ymax=ymax,
                             ax=ax, label='stimFreeze_times', color='y', linewidth=width)
        plots.vertical_lines(trials['stimOff_times'], ymin=0, ymax=ymax,
                             ax=ax, label='stim off', color='c', linewidth=width)
        plots.vertical_lines(trials['stimOn_times'], ymin=0, ymax=ymax,
                             ax=ax, label='stimOn_times', color='tab:orange', linewidth=width)
        c = get_sync_fronts(sync, chmap['left_camera'], tmin=tmin, tmax=tmax)
        plots.squares(c['times'], c['polarities'] * 0.4 + 5, ax=ax, color='k')
        c = get_sync_fronts(sync, chmap['right_camera'], tmin=tmin, tmax=tmax)
        plots.squares(c['times'], c['polarities'] * 0.4 + 6, ax=ax, color='k')
        c = get_sync_fronts(sync, chmap['body_camera'], tmin=tmin, tmax=tmax)
        plots.squares(c['times'], c['polarities'] * 0.4 + 7, ax=ax, color='k')
        ax.legend()
        ax.set_yticklabels(['', 'bpod', 'f2ttl', 'audio', 're_0', ''])
        ax.set_yticks([0, 1, 2, 3, 4, 5])
        ax.set_ylim([0, 5])

    return trials


def extract_sync(session_path, overwrite=False, ephys_files=None, namespace='spikeglx'):
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
        bin_file = efi.get('ap', efi.get('nidq', None))
        if not bin_file:
            continue
        alfname = dict(object='sync', namespace=namespace)
        if efi.label:
            alfname['extra'] = efi.label
        file_exists = alfio.exists(bin_file.parent, **alfname)
        if not overwrite and file_exists:
            _logger.warning(f'Skipping raw sync: SGLX sync found for {efi.label}!')
            sync = alfio.load_object(bin_file.parent, **alfname)
            out_files, _ = alfio._ls(bin_file.parent, **alfname)
        else:
            sr = spikeglx.Reader(bin_file)
            sync, out_files = _sync_to_alf(sr, bin_file.parent, save=True, parts=efi.label)
            sr.close()
        outputs.extend(out_files)
        syncs.extend([sync])

    return syncs, outputs


def _get_all_probes_sync(session_path, bin_exists=True):
    # round-up of all bin ephys files in the session, infer revision and get sync map
    ephys_files = spikeglx.glob_ephys_files(session_path, bin_exists=bin_exists)
    version = spikeglx.get_neuropixel_version_from_files(ephys_files)
    # attach the sync information to each binary file found
    for ef in ephys_files:
        ef['sync'] = alfio.load_object(ef.path, 'sync', namespace='spikeglx', short_keys=True)
        ef['sync_map'] = get_ibl_sync_map(ef, version)
    return ephys_files


def get_wheel_positions(sync, chmap, tmin=None, tmax=None):
    """
    Gets the wheel position from synchronisation pulses

    Parameters
    ----------
    sync : dict
        A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses and
        the corresponding channel numbers.
    chmap : dict[str, int]
        A map of channel names and their corresponding indices.
    tmin : float
        The minimum time from which to extract the sync pulses.
    tmax : float
        The maximum time up to which we extract the sync pulses.

    Returns
    -------
    Bunch
        A dictionary with keys ('timestamps', 'position'), containing the wheel event timestamps and
        position in radians
    Bunch
        A dictionary of detected movement times with keys ('intervals', 'peakAmplitude', 'peakVelocity_times').
    """
    ts, pos = extract_wheel_sync(sync=sync, chmap=chmap, tmin=tmin, tmax=tmax)
    moves = Bunch(extract_wheel_moves(ts, pos))
    wheel = Bunch({'timestamps': ts, 'position': pos})
    return wheel, moves


def get_main_probe_sync(session_path, bin_exists=False):
    """
    From 3A or 3B multiprobe session, returns the main probe (3A) or nidq sync pulses
    with the attached channel map (default chmap if none)

    Parameters
    ----------
    session_path : str, pathlib.Path
        The absolute session path, i.e. '/path/to/subject/yyyy-mm-dd/nnn'.
    bin_exists : bool
        Whether there is a .bin file present.

    Returns
    -------
    one.alf.io.AlfBunch
        A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses and
        the corresponding channel numbers.
    dict
        A map of channel names and their corresponding indices.
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


def get_protocol_period(session_path, protocol_number, bpod_sync):
    """

    Parameters
    ----------
    session_path : str, pathlib.Path
        The absolute session path, i.e. '/path/to/subject/yyyy-mm-dd/nnn'.
    protocol_number : int
        The order that the protocol was run in.
    bpod_sync : dict
        The sync times and polarities for Bpod BNC1.

    Returns
    -------
    float
        The time of the detected spacer for the protocol number.
    float, None
        The time of the next detected spacer or None if this is the last protocol run.
    """
    # The spacers are TTLs generated by Bpod at the start of each protocol
    spacer_times = Spacer().find_spacers_from_fronts(bpod_sync)
    # Ensure that the number of detected spacers matched the number of expected tasks
    if acquisition_description := session_params.read_params(session_path):
        n_tasks = len(acquisition_description.get('tasks', []))
        assert n_tasks == len(spacer_times), f'expected {n_tasks} spacers, found {len(spacer_times)}'
        assert n_tasks > protocol_number >= 0, f'protocol number must be between 0 and {n_tasks}'
    else:
        assert protocol_number < len(spacer_times)
    start = spacer_times[int(protocol_number)]
    end = None if len(spacer_times) - 1 == protocol_number else spacer_times[int(protocol_number + 1)]
    return start, end


class FpgaTrials(extractors_base.BaseExtractor):
    save_names = ('_ibl_trials.intervals_bpod.npy',
                  '_ibl_trials.goCueTrigger_times.npy', None, None, None, None, None, None, None,
                  '_ibl_trials.stimOff_times.npy', None, None, None, None,
                  '_ibl_trials.table.pqt', '_ibl_wheel.timestamps.npy',
                  '_ibl_wheel.position.npy', '_ibl_wheelMoves.intervals.npy',
                  '_ibl_wheelMoves.peakAmplitude.npy')
    var_names = ('intervals_bpod',
                 'goCueTrigger_times', 'stimOnTrigger_times',
                 'stimOffTrigger_times', 'stimFreezeTrigger_times', 'errorCueTrigger_times',
                 'errorCue_times', 'itiIn_times', 'stimFreeze_times', 'stimOff_times',
                 'valveOpen_times', 'phase', 'position', 'quiescence', 'table',
                 'wheel_timestamps', 'wheel_position',
                 'wheelMoves_intervals', 'wheelMoves_peakAmplitude')

    # Fields from bpod extractor that we want to resync to FPGA
    bpod_rsync_fields = ('intervals', 'response_times', 'goCueTrigger_times',
                         'stimOnTrigger_times', 'stimOffTrigger_times',
                         'stimFreezeTrigger_times', 'errorCueTrigger_times')

    # Fields from bpod extractor that we want to save
    bpod_fields = ('feedbackType', 'choice', 'rewardVolume', 'contrastLeft', 'contrastRight', 'probabilityLeft',
                   'intervals_bpod', 'phase', 'position', 'quiescence')

    def __init__(self, *args, **kwargs):
        """An extractor for all ephys trial data, in FPGA time"""
        super().__init__(*args, **kwargs)
        self.bpod2fpga = None

    def _extract(self, sync=None, chmap=None, sync_collection='raw_ephys_data', task_collection='raw_behavior_data', **kwargs):
        """Extracts ephys trials by combining Bpod and FPGA sync pulses"""
        # extract the behaviour data from bpod
        if sync is None or chmap is None:
            _sync, _chmap = get_sync_and_chn_map(self.session_path, sync_collection)
            sync = sync or _sync
            chmap = chmap or _chmap
        # load the bpod data and performs a biased choice world training extraction
        # TODO these all need to pass in the collection so we can load for different protocols in different folders
        bpod_raw = raw_data_loaders.load_data(self.session_path, task_collection=task_collection)
        assert bpod_raw is not None, "No task trials data in raw_behavior_data - Exit"

        bpod_trials = self._extract_bpod(bpod_raw, task_collection=task_collection, save=False)
        # Explode trials table df
        trials_table = alfio.AlfBunch.from_df(bpod_trials.pop('table'))
        table_columns = trials_table.keys()
        bpod_trials.update(trials_table)
        # synchronize
        bpod_trials['intervals_bpod'] = np.copy(bpod_trials['intervals'])

        # Get the spacer times for this protocol
        if (protocol_number := kwargs.get('protocol_number')) is not None:  # look for spacer
            # The spacers are TTLs generated by Bpod at the start of each protocol
            bpod = get_sync_fronts(sync, chmap['bpod'])
            tmin, tmax = get_protocol_period(self.session_path, protocol_number, bpod)
        else:
            tmin = tmax = None

        fpga_trials = extract_behaviour_sync(
            sync=sync, chmap=chmap, bpod_trials=bpod_trials, tmin=tmin, tmax=tmax)
        # checks consistency and compute dt with bpod
        self.bpod2fpga, drift_ppm, ibpod, ifpga = neurodsp.utils.sync_timestamps(
            bpod_trials['intervals_bpod'][:, 0], fpga_trials.pop('intervals')[:, 0],
            return_indices=True)
        nbpod = bpod_trials['intervals_bpod'].shape[0]
        npfga = fpga_trials['feedback_times'].shape[0]
        nsync = len(ibpod)
        _logger.info(f"N trials: {nbpod} bpod, {npfga} FPGA, {nsync} merged, sync {drift_ppm} ppm")
        if drift_ppm > BPOD_FPGA_DRIFT_THRESHOLD_PPM:
            _logger.warning('BPOD/FPGA synchronization shows values greater than %i ppm',
                            BPOD_FPGA_DRIFT_THRESHOLD_PPM)
        out = OrderedDict()
        out.update({k: bpod_trials[k][ibpod] for k in self.bpod_fields})
        out.update({k: self.bpod2fpga(bpod_trials[k][ibpod]) for k in self.bpod_rsync_fields})
        out.update({k: fpga_trials[k][ifpga] for k in sorted(fpga_trials.keys())})
        # extract the wheel data
        wheel, moves = get_wheel_positions(sync=sync, chmap=chmap, tmin=tmin, tmax=tmax)
        from ibllib.io.extractors.training_wheel import extract_first_movement_times
        settings = raw_data_loaders.load_settings(session_path=self.session_path, task_collection=task_collection)
        min_qt = settings.get('QUIESCENT_PERIOD', None)
        first_move_onsets, *_ = extract_first_movement_times(moves, out, min_qt=min_qt)
        out.update({'firstMovement_times': first_move_onsets})
        # Re-create trials table
        trials_table = alfio.AlfBunch({x: out.pop(x) for x in table_columns})
        out['table'] = trials_table.to_df()

        out = {k: out[k] for k in self.var_names if k in out}  # Reorder output
        assert tuple(filter(lambda x: 'wheel' not in x, self.var_names)) == tuple(out.keys())
        return [out[k] for k in out] + [wheel['timestamps'], wheel['position'],
                                        moves['intervals'], moves['peakAmplitude']]

    def _extract_bpod(self, bpod_trials, task_collection='raw_behavior_data', save=False):
        bpod_trials, *_ = bpod_extract_all(
            session_path=self.session_path, save=save, bpod_trials=bpod_trials, task_collection=task_collection)

        return bpod_trials


def extract_all(session_path, sync_collection='raw_ephys_data', save=True, task_collection='raw_behavior_data', save_path=None,
                protocol_number=None, **kwargs):
    """
    For the IBL ephys task, reads ephys binary file and extract:
        -   sync
        -   wheel
        -   behaviour
        -   video time stamps

    Parameters
    ----------
    session_path : str, pathlib.Path
        The absolute session path, i.e. '/path/to/subject/yyyy-mm-dd/nnn'.
    sync_collection : str
        The session subdirectory where the sync data are located.
    save : bool
        If true, save the extracted files to save_path.
    task_collection : str
        The location of the behaviour protocol data.
    save_path : str, pathlib.Path
        The save location of the extracted files, defaults to the alf directory of the session path.
    protocol_number : int
        The order that the protocol was run in.
    **kwargs
        Optional extractor keyword arguments.

    Returns
    -------
    list
        The extracted data.
    list of pathlib.Path, None
        If save is True, a list of file paths to the extracted data.
    """
    extractor_type = extractors_base.get_session_extractor_type(session_path, task_collection=task_collection)
    _logger.info(f"Extracting {session_path} as {extractor_type}")
    sync, chmap = get_sync_and_chn_map(session_path, sync_collection)
    # sync, chmap = get_main_probe_sync(session_path, bin_exists=bin_exists)
    base = [FpgaTrials]
    if extractor_type == 'ephys_biased_opto':
        base.append(LaserBool)
    outputs, files = extractors_base.run_extractor_classes(
        base, session_path=session_path, save=save, sync=sync, chmap=chmap, path_out=save_path,
        task_collection=task_collection, protocol_number=protocol_number, **kwargs)
    return outputs, files


def get_sync_and_chn_map(session_path, sync_collection):
    """
    Return sync and channel map for session based on collection where main sync is stored.

    Parameters
    ----------
    session_path : str, pathlib.Path
        The absolute session path, i.e. '/path/to/subject/yyyy-mm-dd/nnn'.
    sync_collection : str
        The session subdirectory where the sync data are located.

    Returns
    -------
    one.alf.io.AlfBunch
        A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses and
        the corresponding channel numbers.
    dict
        A map of channel names and their corresponding indices.
    """
    if sync_collection == 'raw_ephys_data':
        # Check to see if we have nidq files, if we do just go with this otherwise go into other function that deals with
        # 3A probes
        nidq_meta = next(session_path.joinpath(sync_collection).glob('*nidq.meta'), None)
        if not nidq_meta:
            sync, chmap = get_main_probe_sync(session_path)
        else:
            sync = load_sync(session_path, sync_collection)
            ef = Bunch()
            ef['path'] = session_path.joinpath(sync_collection)
            ef['nidq'] = nidq_meta
            chmap = get_ibl_sync_map(ef, '3B')

    else:
        sync = load_sync(session_path, sync_collection)
        chmap = load_channel_map(session_path, sync_collection)

    return sync, chmap


def load_channel_map(session_path, sync_collection):
    """
    Load syncing channel map for session path and collection

    Parameters
    ----------
    session_path : str, pathlib.Path
        The absolute session path, i.e. '/path/to/subject/yyyy-mm-dd/nnn'.
    sync_collection : str
        The session subdirectory where the sync data are located.

    Returns
    -------
    dict
        A map of channel names and their corresponding indices.
    """

    device = sync_collection.split('_')[1]
    default_chmap = DEFAULT_MAPS[device]['nidq']

    # Try to load channel map from file
    chmap = spikeglx.get_sync_map(session_path.joinpath(sync_collection))
    # If chmap provided but not with all keys, fill up with default values
    if not chmap:
        return default_chmap
    else:
        if data_for_keys(default_chmap.keys(), chmap):
            return chmap
        else:
            _logger.warning("Keys missing from provided channel map, "
                            "setting missing keys from default channel map")
            return {**default_chmap, **chmap}


def load_sync(session_path, sync_collection):
    """
    Load sync files from session path and collection.

    Parameters
    ----------
    session_path : str, pathlib.Path
        The absolute session path, i.e. '/path/to/subject/yyyy-mm-dd/nnn'.
    sync_collection : str
        The session subdirectory where the sync data are located.

    Returns
    -------
    one.alf.io.AlfBunch
        A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses and
        the corresponding channel numbers.
    """
    sync = alfio.load_object(session_path.joinpath(sync_collection), 'sync', namespace='spikeglx', short_keys=True)

    return sync

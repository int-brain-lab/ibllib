import logging
import tempfile
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import ibllib.plots as plots
import ibllib.behaviour.wheel as whl
import ibllib.io
import ibllib.dsp as dsp
import alf.io

_logger = logging.getLogger('ibllib')

SYNC_BATCH_SIZE_SAMPLES = 2 ** 18  # number of samples to read at once in bin file for sync
WHEEL_RADIUS_CM = 3.1
WHEEL_TICKS = 1024
DEBUG_PLOTS = False
# this is the mapping of synchronisation pulses coming out of the FPGA
AUXES = [
    (0, None),
    (1, None),
    (2, 'left_camera'),
    (3, 'right_camera'),
    (4, 'body_camera'),
    (5, None),
    (6, None),
    (7, 'bpod'),
    (8, None),
    (9, None),
    (10, None),
    (11, None),
    (12, 'frame2ttl'),
    (13, 'rotary_encoder_0'),
    (14, 'rotary_encoder_1'),
    (15, 'audio'),
]
SYNC_CHANNEL_MAP = {}
for aux in AUXES:
    if aux[1]:
        SYNC_CHANNEL_MAP[aux[1]] = aux[0]


def _sync_to_alf(raw_ephys_apfile, output_path=None, save=False):
    """
    Extracts sync.times, sync.channels and sync.polarities from binary ephys dataset

    :param raw_ephys_apfile: bin file containing ephys data or spike
    :param out_dir: output directory
    :return:
    """
    if not output_path:
        file_ftcp = tempfile.TemporaryFile()
    else:
        file_ftcp = Path(output_path) / 'fronts_times_channel_polarity.bin'
    if isinstance(raw_ephys_apfile, ibllib.io.spikeglx.Reader):
        sr = raw_ephys_apfile
    else:
        sr = ibllib.io.spikeglx.Reader(raw_ephys_apfile)
    # loop over chunks of the raw ephys file
    wg = dsp.WindowGenerator(sr.ns, SYNC_BATCH_SIZE_SAMPLES, overlap=1)
    fid_ftcp = open(file_ftcp, 'wb')
    for sl in wg.slice:
        ss = sr.read_sync(sl)
        ind, fronts = dsp.fronts(ss, axis=0)
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
        alf.io.save_object_npy(output_path, sync, '_spikeglx_sync')
    return sync


def _bpod_events_extraction(bpod_t, bpod_fronts):
    """
    From detected fronts on the bpod sync traces, outputs the synchronisation events
    related to trial start and valve opening
    :param bpod_t: numpy vector containing times of fronts
    :param bpod_fronts: numpy vector containing polarity of fronts (1 rise, -1 fall)
    :return: numpy arrays of times t_trial_start, t_valve_open and t_iti_in
    """
    # make sure that there are no 2 consecutive fall or consecutive rise events
    assert(np.all(np.abs(np.diff(bpod_fronts)) == 2))
    # make sure that the first event is a rise
    assert(bpod_fronts[0] == 1)
    # take only even time differences: ie. from rising to falling fronts
    dt = np.diff(bpod_t)[::2]
    # detect start trials event assuming length is 0.1 ms except the first trial
    i_trial_start = np.r_[0, np.where(dt <= 1.66e-4)[0] * 2]
    t_trial_start = bpod_t[i_trial_start]
    # # the first trial we detect the first falling edge to which we subtract 0.1ms
    # t_trial_start[0] -= 1e-4
    # the last trial is a dud and should be removed
    t_trial_start = t_trial_start[:-1]
    # valve open events are between 50ms to 300 ms
    i_valve_open = np.where(np.logical_and(dt > 1.66e-4, dt < 0.4))[0] * 2
    i_valve_open = np.delete(i_valve_open, np.where(i_valve_open < 2))
    t_valve_open = bpod_t[i_valve_open]
    # ITI events are above 400 ms
    i_iti_in = np.where(dt > 0.4)[0] * 2
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


def _rotary_encoder_positions_from_fronts(ta, pa, tb, pb):
    """
    Extracts the rotary encoder absolute position (cm) as function of time from fronts detected
    on the 2 channels

    :param ta: time of fronts on channel A
    :param pa: polarity of fronts on channel A
    :param tb: time of fronts on channel B
    :param pb: polarity of fronts on channel B
    :return: indices vector (ta) and position vector
    """
    p = pb[np.searchsorted(tb, ta) - 1] * pa
    p = np.cumsum(p) / WHEEL_TICKS * np.pi * WHEEL_RADIUS_CM
    return ta, p


def _rotary_encoder_positions_from_gray_code(channela, channelb):
    """
    Extracts the rotary encoder absolute position (cm) as function of time from digital recording
    of the 2 channels.

    Rotary Encoder implements X1 encoding: http://www.ni.com/tutorial/7109/en/
    rising A  & B high = +1
    rising A  & B low = -1
    falling A & B high = -1
    falling A & B low = +1

    :param channelA: Vector of rotary encoder digital recording channel A
    :type channelA: numpy array
    :param channelB: Vector of rotary encoder digital recording channel B
    :type channelB: numpy array
    :return: indices vector and position vector
    """
    # detect rising and falling fronts
    t, fronts = dsp.fronts(channela)
    # apply X1 logic to get positions in ticks
    p = (channelb[t] * 2 - 1) * fronts
    # convert position in cm
    p = np.cumsum(p) / WHEEL_TICKS * np.pi * WHEEL_RADIUS_CM
    return t, p


def _audio_events_extraction(audio_t, audio_fronts):
    """
    From detected fronts on the audio sync traces, outputs the synchronisation events
    related to tone in

    :param audio_t: numpy vector containing times of fronts
    :param audio_fronts: numpy vector containing polarity of fronts (1 rise, -1 fall)
    :return: numpy arrays t_ready_tone_in, t_error_tone_in
    """
    # make sure that there are no 2 consecutive fall or consecutive rise events
    assert(np.all(np.abs(np.diff(audio_fronts)) == 2))
    # make sure that the first event is a rise
    assert(audio_fronts[0] == 1)
    # take only even time differences: ie. from rising to falling fronts
    dt = np.diff(audio_t)[::2]
    # detect ready tone by length below 110 ms
    i_ready_tone_in = np.r_[1, np.where(dt <= 0.11)[0] * 2]
    t_ready_tone_in = audio_t[i_ready_tone_in]
    # error tones are events lasting from 400ms to 600ms
    i_error_tone_in = np.where(np.logical_and(0.4 < dt, dt < 0.6))[0] * 2
    t_error_tone_in = audio_t[i_error_tone_in]
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


def _get_sync_fronts(sync, channel_nb):
    return {'times': sync['times'][sync['channels'] == channel_nb],
            'polarities': sync['polarities'][sync['channels'] == channel_nb]}


def extract_wheel_sync(sync, output_path=None, save=False, chmap=SYNC_CHANNEL_MAP):
    """
    Extract wheel positions and times from sync fronts dictionary for all 16 chans

    :param sync: dictionary 'times', 'polarities' of fronts detected on sync trace
    :param output_path: where to save the data
    :param save: True/False
    :param chmap: dictionary containing channel indices. Default to constant.
        chmap = {'rotary_encoder_0': 13, 'rotary_encoder_1': 14}
    :return: dictionary containing wheel data, 'wheel_ts', 're_ts'
    """
    wheel = {}
    channela = _get_sync_fronts(sync, chmap['rotary_encoder_0'])
    channelb = _get_sync_fronts(sync, chmap['rotary_encoder_1'])
    wheel['re_ts'], wheel['re_pos'] = _rotary_encoder_positions_from_fronts(
        channela['times'], channela['polarities'], channelb['times'], channelb['polarities'])
    if save and output_path:
        output_path = Path(output_path)
        # last phase of the process is to save the alf data-files
        np.save(output_path / '_ibl_wheel.position.npy', wheel['re_pos'])
        np.save(output_path / '_ibl_wheel.times.npy', wheel['re_ts'])
        np.save(output_path / '_ibl_wheel.velocity.npy',
                whl.velocity(wheel['re_ts'], wheel['re_pos']))
    return wheel


def extract_behaviour_sync(sync, output_path=None, save=False, chmap=SYNC_CHANNEL_MAP):
    """
    Extract wheel positions and times from sync fronts dictionary

    :param sync: dictionary 'times', 'polarities' of fronts detected on sync trace for all 16 chans
    :param output_path: where to save the data
    :param save: True/False
    :param chmap: dictionary containing channel index. Default to constant.
        chmap = {'bpod': 7, 'frame2ttl': 12, 'audio': 15}
    :return: trials dictionary
    """
    bpod = _get_sync_fronts(sync, chmap['bpod'])
    frame2ttl = _get_sync_fronts(sync, chmap['frame2ttl'])
    audio = _get_sync_fronts(sync, chmap['audio'])
    # extract events from the fronts for each trace
    t_trial_start, t_valve_open, t_iti_in = _bpod_events_extraction(
        bpod['times'], bpod['polarities'])
    t_ready_tone_in, t_error_tone_in = _audio_events_extraction(
        audio['times'], audio['polarities'])
    # stim off time is the first frame2ttl rise/fall after the trial start
    # does not apply for 1st trial
    ind = np.searchsorted(frame2ttl['times'], t_trial_start[1:], side='left')
    t_stim_off = frame2ttl['times'][ind]
    # the t_stim_off happens 100ms after trial start
    assert(np.all((t_trial_start[1:] - t_stim_off) > -0.1))
    t_stim_freeze = frame2ttl['times'][ind - 1]

    if DEBUG_PLOTS:
        plt.figure()
        ax = plt.gca()
        plots.squares(bpod['times'], bpod['polarities'] * 0.4 + 1,
                      ax=ax, label='bpod=1', color='k')
        plots.squares(frame2ttl['times'], frame2ttl['polarities'] * 0.4 + 2,
                      ax=ax, label='frame2ttl=2', color='k')
        plots.squares(audio['times'], audio['polarities'] * 0.4 + 3,
                      ax=ax, label='audio=3', color='k')
        plots.vertical_lines(t_ready_tone_in, ymin=0, ymax=4,
                             ax=ax, label='ready tone in', color='b', linewidth=0.5)
        plots.vertical_lines(t_trial_start, ymin=0, ymax=4,
                             ax=ax, label='start_trial', color='m', linewidth=0.5)
        plots.vertical_lines(t_error_tone_in, ymin=0, ymax=4,
                             ax=ax, label='error tone', color='r', linewidth=0.5)
        plots.vertical_lines(t_valve_open, ymin=0, ymax=4,
                             ax=ax, label='valve open', color='g', linewidth=0.5)
        plots.vertical_lines(t_stim_freeze, ymin=0, ymax=4,
                             ax=ax, label='stim freeze', color='y', linewidth=0.5)
        plots.vertical_lines(t_stim_off, ymin=0, ymax=4,
                             ax=ax, label='stim off', color='c', linewidth=0.5)
        ax.legend()

    # stimOn_times: first fram2ttl change after trial start
    trials = {
        'ready_tone_in': _assign_events_to_trial(t_trial_start, t_ready_tone_in),
        'error_tone_in': _assign_events_to_trial(t_trial_start, t_error_tone_in),
        'valve_open': _assign_events_to_trial(t_trial_start, t_valve_open),
        'stim_freeze': _assign_events_to_trial(t_trial_start, t_stim_freeze),
        'stimOn_times': _assign_events_to_trial(t_trial_start, frame2ttl['times'], take='first'),
        'iti_in': _assign_events_to_trial(t_trial_start, t_iti_in)
    }
    # goCue_times corresponds to the tone_in event
    trials['goCue_times'] = trials['ready_tone_in']
    # response_times is TONE_IN to STIM freeze
    trials['response_times'] = trials['stim_freeze'] - trials['ready_tone_in']
    # feedback times are valve open on good trials and error tone in on error trials
    trials['feedback_times'] = trials['valve_open']
    ind_err = np.isnan(trials['valve_open'])
    trials['feedback_times'][ind_err] = trials['error_tone_in'][ind_err]
    # # # # this is specific to version 4
    trials['iti_in'] = trials['valve_open'] + 1.
    trials['iti_in'][ind_err] = trials['error_tone_in'][ind_err] + 2.
    trials['intervals'] = np.c_[t_trial_start, trials['iti_in']]
    # # # # end of specific to version 4
    if save and output_path:
        output_path = Path(output_path)
        np.save(output_path / '_ibl_trials.goCue_times.npy', trials['goCue_times'])
        np.save(output_path / '_ibl_trials.response_times.npy', trials['response_times'])
        np.save(output_path / '_ibl_trials.stimOn_times.npy', trials['stimOn_times'])
        np.save(output_path / '_ibl_trials.intervals.npy', trials['intervals'])
        np.save(output_path / '_ibl_trials.feedback_times.npy', trials['feedback_times'])
    return trials


def align_with_bpod(session_path):
    """
    Reads in trials.intervals ALF dataset from bpod and fpga.
    Asserts consistency between datasets and compute the median time difference

    :param session_path:
    :return: dt: median time difference of trial start times (fpga - bpod)
    """
    # check consistency
    output_path = Path(session_path) / 'alf'
    trials = alf.io.load_object(output_path, '_ibl_trials')
    assert(ibllib.io.check_dimensions(trials) == 0)
    dt = (np.diff(trials['intervalsBpod']) - np.diff(trials['intervals']))
    assert(np.all(np.abs(dt[np.invert(np.isnan(dt))]) < 5 * 1e-3))
    dt = trials['intervals'][:, 0] - trials['intervalsBpod'][:, 0]
    # plt.plot(np.diff(trials['intervalsBpod']), '*')
    # plt.plot(np.diff(trials['intervals']), '.')
    return np.median(dt)


def extract_all(session_path, save=False, version=None):
    """
    Reads ephys binary file and extract sync, wheel and behaviour ALF files

    :param session_path: '/path/to/subject/yyyy-mm-dd/001'
    :param save: Bool, defaults to False
    :param version: bpod version, defaults to None
    :return: None
    """
    session_path = Path(session_path)
    output_path = session_path / 'alf'
    raw_ephys_path = session_path / 'raw_ephys_data'
    if not output_path.exists():
        output_path.mkdir()

    ephys_files = list(raw_ephys_path.rglob('*.ap.bin'))
    if len(ephys_files) > 2:
        raise NotImplementedError("Multiple probes/files extraction not implemented. Contact us !")
    # TODO Extract channel maps from meta-data
    raw_ephys_apfile = ephys_files[0]
    sr = ibllib.io.spikeglx.Reader(raw_ephys_apfile)
    sync = _sync_to_alf(sr, output_path, save=save)
    # TODO Extract camera time-stamps
    extract_wheel_sync(sync, output_path, save=save)
    extract_behaviour_sync(sync, output_path, save=save)
    align_with_bpod(session_path)  # checks consistency and compute dt with bpod

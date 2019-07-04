#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Training wheel extractor from Pybpod output.
"""
import os
import logging

import numpy as np
from scipy import interpolate

import ibllib.io.raw_data_loaders as raw
from ibllib.misc import structarr
import ibllib.behaviour.wheel as wheel

logger_ = logging.getLogger('ibllib.alf')
WHEEL_RADIUS_CM = 3.1
THRESHOLD_RAD_PER_SEC = 10
THRESHOLD_CONSECUTIVE_SAMPLES = 0.001
EPS = 7. / 3 - 4. / 3 - 1


def check_alf_folder(session_path):
    """
    Check if alf folder exists, creates it if it doesn't.

    :param session_path: absolute path of session folder
    :type session_path: str
    """
    alf_folder = os.path.join(session_path, 'alf')
    if not os.path.exists(alf_folder):
        os.mkdir(alf_folder)


def get_trial_start_times(session_path, data=None):
    if not data:
        data = raw.load_data(session_path)
    trial_start_times = []
    for tr in data:
        trial_start_times.extend(
            [x[0] for x in tr['behavior_data']['States timestamps']['trial_start']])
    return np.array(trial_start_times)


def get_trial_start_times_re(session_path, evt=None):
    if not evt:
        evt = raw.load_encoder_events(session_path)
    trial_start_times_re = evt.re_ts[evt.sm_ev[evt.sm_ev == 1].index].values / 1e6
    return trial_start_times_re[:-1]


def time_converter_session(session_path, kind):
    """
    Create interp1d functions to convert values from one clock to another given a
    set of synchronization pulses.

    The task global sync pulse is at trial_start from Bpod to:
    Rotary Encoder, Cameras and e-phys system.
    Depends on getter functions that extract from the raw data the timestamps
    of the trial_start sync pulse event for each clock.

    +---------+----------+-----------+-----------+-----------+
    | 2b      |   -      |   re2b    | cam2b,    |  ephys2b  |
    +---------+----------+-----------+-----------+-----------+
    | 2re     |   b2re   |   -       | cam2re    |  ephys2re |
    +---------+----------+-----------+-----------+-----------+
    | 2cam    |   b2cam  |   re2cam  |  -        |  ephys2cam|
    +---------+----------+-----------+-----------+-----------+
    | 2ephys  |   b2ephys|   re2ephys| cam2ephys |  -        |
    +---------+----------+-----------+-----------+-----------+

    Default converters for times are assumed to be of kind *2b* unless ephys data
    is present in that case converters for 'times' will be of kind *2ephys*

    :param session_path: absolute path of session folder
    :type session_path: str
    :param kind: ['re2b', 'b2re'], defaults to 're2b'
    :type kind: str, optional
    :return: Function that converts from clock A to clock B defined by kind.
    :rtype: scipy.interpolate.interpolate.interp1d
    """
    # there should be a way to input data if already in memory
    if kind == 're2b':
        target = get_trial_start_times(session_path)
        tref = get_trial_start_times_re(session_path)
    elif kind == 'b2re':
        tref = get_trial_start_times(session_path)
        target = get_trial_start_times_re(session_path)

    return time_interpolation(tref, target)


def time_interpolation(tref, target):
    """
    From 2 arrays of timestamps, return an interpolation function that allows to go
    from one to the other.
    If sizes are different, only work with the first elements.
    """
    if tref.size != target.size:
        logger_.warning('Time-stamp arrays have inconsistent size. Trimming to the smallest size')
        siz = min(tref.size, target.size)
        tref = tref[:siz]
        target = target[:siz]
    if tref.size == target.size == 1:
        logger_.error('Wheel time-stamp arrays have only one value ?!!?. This is a dud. ABORT')
        raise(ValueError)
    func = interpolate.interp1d(tref, target, fill_value="extrapolate")
    return func


def get_wheel_data(session_path, bp_data=None, save=False):
    """
    Get wheel data from raw files and converts positions into centimeters and
    timestamps into seconds.
    **Optional:** saves _ibl_wheel.times.npy and _ibl_wheel.position.npy

    Times:
    Gets Rotary Encoder timestamps (ms) for each position and converts to times.

    Uses time_converter to extract and convert timstamps (ms) to times (s).

    Positions:
    Positions are in (cm) of RE perimeter relative to 0. The 0 resets every trial.

    cmtick = radius (cm) * 2 * pi / n_ticks
    cmtick = 3.1 * 2 * np.pi / 1024

    :param session_path: absolute path of session folder
    :type session_path: str
    :param data: dictionary containing the contents pybppod jsonable file read with raw.load_data
    :type data: dict, optional
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: Numpy structured array.
    :rtype: numpy.ndarray
    """
    ##
    status = 0
    if not bp_data:
        bp_data = raw.load_data(session_path)
    df = raw.load_encoder_positions(session_path)
    if df is None:
        logger_.error('No wheel data for ' + str(session_path))
        return None
    data = structarr(['re_ts', 're_pos', 'bns_ts'],
                     shape=(df.shape[0],), formats=['f8', 'f8', np.object])
    data['re_ts'] = df.re_ts.values
    data['re_pos'] = df.re_pos.values
    data['re_pos'] = data['re_pos'] / 1024 * 2 * np.pi  # convert positions to radians
    trial_starts = get_trial_start_times(session_path)
    # need a flag if the data resolution is 1ms due to the old version of rotary encoder firmware
    if np.all(np.mod(data['re_ts'], 1e3) == 0):
        status = 1
    data['re_ts'] = data['re_ts'] / 1e6  # convert ts to seconds
    # get the converter function to translate re_ts into behavior times
    convtime = time_converter_session(session_path, kind='re2b')
    data['re_ts'] = convtime(data['re_ts'])
    return data

    def get_reset_trace_compensation_with_state_machine_times():
        # this is the preferred way of getting resets using the state machine time information
        # it will not always work depending on firmware versions, new bugs
        iwarn = []
        ns = len(data['re_pos'])
        tr_dc = np.zeros_like(data['re_pos'])  # trial dc component
        for bp_dat in bp_data:
            restarts = np.sort(np.array(
                bp_dat['behavior_data']['States timestamps']['reset_rotary_encoder'] +
                bp_dat['behavior_data']['States timestamps']['reset2_rotary_encoder'])[:, 0])
            ind = np.unique(np.searchsorted(data['re_ts'], restarts, side='left') - 1)
            # the rotary encoder doesn't always reset right away, and the reset sample given the
            # timestamp can be ambiguous: look for zeros
            for i in np.where(data['re_pos'][ind] != 0)[0]:
                # handle boundary effects
                if ind[i] > ns - 2:
                    continue
                # it happens quite often that we have to lock in to next sample to find the reset
                if data['re_pos'][ind[i] + 1] == 0:
                    ind[i] = ind[i] + 1
                    continue
                # also case where the rotary doesn't reset to 0, but erratically to -1/+1
                if data['re_pos'][ind[i]] <= (1 / 1024 * 2 * np.pi):
                    ind[i] = ind[i] + 1
                    continue
                # compounded with the fact that the reset may have happened at next sample.
                if np.abs(data['re_pos'][ind[i] + 1]) <= (1 / 1024 * 2 * np.pi):
                    ind[i] = ind[i] + 1
                    continue
                # sometimes it is also the last trial that has this behaviour
                if (bp_data[-1] is bp_dat) or (bp_data[0] is bp_dat):
                    continue
                iwarn.append(ind[i])
                # at which point we are running out of possible bugs and calling it
            tr_dc[ind] = data['re_pos'][ind - 1]
        if iwarn:  # if a warning flag was caught in the loop throw a single warning
            logger_.warning('Rotary encoder reset events discrepancy. Doing my best to merge.')
            logger_.debug('Offending inds: ' + str(iwarn) + ' times: ' + str(data['re_ts'][iwarn]))
        # exit status 0 is fine, 1 something went wrong
        return tr_dc, len(iwarn) != 0

    # attempt to get the resets properly unless the unit is ms which means precision is
    # not good enough to match SM times to wheel samples time
    if not status:
        tr_dc, status = get_reset_trace_compensation_with_state_machine_times()

    # if something was wrong or went wrong agnostic way of getting resets: just get zeros values
    if status:
        tr_dc = np.zeros_like(data['re_pos'])  # trial dc component
        i0 = np.where(data['re_pos'] == 0)[0]
        tr_dc[i0] = data['re_pos'][i0 - 1]
    # even if things went ok, rotary encoder may not log the whole session. Need to fix outside
    else:
        i0 = np.where(np.bitwise_and(np.bitwise_or(data['re_ts'] >= trial_starts[-1],
                                                   data['re_ts'] <= trial_starts[0]),
                                     data['re_pos'] == 0))[0]
    # make sure the bounds are not included in the current list
    i0 = np.delete(i0, np.where(np.bitwise_or(i0 >= len(data['re_pos']) - 1, i0 == 0)))
    # a 0 sample is not a reset if 2 conditions are met:
    # 1/2 no inflexion (continuous derivative)
    c1 = np.abs(np.sign(data['re_pos'][i0 + 1] - data['re_pos'][i0]) -
                np.sign(data['re_pos'][i0] - data['re_pos'][i0 - 1])) == 2
    # 2/2 needs to be below threshold
    c2 = np.abs((data['re_pos'][i0] - data['re_pos'][i0 - 1]) /
                (EPS + (data['re_ts'][i0] - data['re_ts'][i0 - 1]))) < THRESHOLD_RAD_PER_SEC
    # apply reset to points identified as resets
    i0 = i0[np.where(np.bitwise_not(np.bitwise_and(c1, c2)))]
    tr_dc[i0] = data['re_pos'][i0 - 1]

    # unwrap the rotation (in radians) and then add the DC component from restarts
    data['re_pos'] = np.unwrap(data['re_pos']) + np.cumsum(tr_dc)

    # Also forgot to mention that time stamps may be repeated or very close to one another.
    # Find them as they will induce large jitters on the velocity function or errors in
    # attempts of interpolation
    rep_idx = np.where(np.diff(data['re_ts']) <= THRESHOLD_CONSECUTIVE_SAMPLES)[0]
    # Change the value of the repeated position
    data['re_pos'][rep_idx] = (data['re_pos'][rep_idx] +
                               data['re_pos'][rep_idx + 1]) / 2
    data['re_ts'][rep_idx] = (data['re_ts'][rep_idx] +
                              data['re_ts'][rep_idx + 1]) / 2
    # Now remove the repeat times that are rep_idx + 1
    data = np.delete(data, rep_idx + 1)

    # convert to cm
    data['re_pos'] = data['re_pos'] * WHEEL_RADIUS_CM

    # #  DEBUG PLOTS START HERE ########################
    # # if you are experiencing a new bug here is some plot tools
    # # do not forget to increment the wasted dev hours counter below
    # WASTED_HOURS_ON_THIS_WHEEL_FORMAT = 16
    #
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = plt.axes()
    # tstart = get_trial_start_times(session_path)
    # tts = np.c_[tstart, tstart, tstart + np.nan].flatten()
    # vts = np.c_[tstart * 0 + 100, tstart * 0 - 100, tstart + np.nan].flatten()
    # ax.plot(tts, vts, label='Trial starts')
    # ax.plot(convtime(df.re_ts.values/1e6), df.re_pos.values / 1024 * 2 * np.pi,
    #         '.-', label='Raw data')
    # i0 = np.where(df.re_pos.values == 0)
    # ax.plot(convtime(df.re_ts.values[i0] / 1e6), df.re_pos.values[i0] / 1024 * 2 * np.pi,
    #         'r*', label='Raw data zero samples')
    # ax.plot(convtime(df.re_ts.values / 1e6) , tr_dc, label='reset compensation')
    # ax.set_xlabel('Bpod Time')
    # ax.set_ylabel('radians')
    # #
    # restarts = np.array(bp_data[10]['behavior_data']['States timestamps']\
    #                         ['reset_rotary_encoder']).flatten()
    # # x__ = np.c_[restarts, restarts, restarts + np.nan].flatten()
    # # y__ = np.c_[restarts * 0 + 1, restarts * 0 - 1, restarts+ np.nan].flatten()
    # #
    # # ax.plot(x__, y__, 'k', label='Restarts')
    #
    # ax.plot(data['re_ts'], data['re_pos'] / WHEEL_RADIUS_CM, '.-', label='Output Trace')
    # ax.legend()
    # # plt.hist(np.diff(data['re_ts']), 400, range=[0, 0.01])
    # #  DEBUG PLOTS STOP HERE ########################

    check_alf_folder(session_path)
    if raw.save_bool(save, '_ibl_wheel.timestamps.npy'):
        tpath = os.path.join(session_path, 'alf', '_ibl_wheel.timestamps.npy')
        np.save(tpath, data['re_ts'])
    if raw.save_bool(save, '_ibl_wheel.position.npy'):
        ppath = os.path.join(session_path, 'alf', '_ibl_wheel.position.npy')
        np.save(ppath, data['re_pos'])
    return data


def get_velocity(session_path, save=False, data_wheel=None):
    """
    Compute velocity from non-uniformly acquired positions and timestamps.
    **Optional:** save _ibl_trials.velocity.npy

    Uses signed_contrast to create left and right contrast vectors.

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    if not isinstance(data_wheel, np.ndarray):
        data_wheel = get_wheel_data(session_path, save=False)
    if data_wheel is None:
        logger_.error('No wheel data for ' + str(session_path))
        return None

    velocity = wheel.velocity(data_wheel['re_ts'], data_wheel['re_pos'])

    if raw.save_bool(save, '_ibl_wheel.velocity.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_wheel.velocity.npy')
        np.save(fpath, velocity)

    return velocity


def extract_all(session_path, bp_data=None, save=False):
    data = get_wheel_data(session_path, bp_data=bp_data, save=save)
    velocity = get_velocity(session_path, save=save, data_wheel=data)
    return data, velocity

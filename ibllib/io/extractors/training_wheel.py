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
import brainbox.behavior.wheel as wheel
import ibllib.exceptions as err

logger_ = logging.getLogger('ibllib.alf')
WHEEL_RADIUS_CM = 1  # we want the output in radians
THRESHOLD_RAD_PER_SEC = 10
THRESHOLD_CONSECUTIVE_SAMPLES = 0
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


def sync_rotary_encoder(session_path, bpod_data=None, re_events=None):
    if not bpod_data:
        bpod_data = raw.load_data(session_path)
    if not re_events:
        evt = raw.load_encoder_events(session_path)
    # we work with stim_on (2) and closed_loop (3) states for the synchronization with bpod
    tre = evt.re_ts.values / 1e6  # convert to seconds
    # the first trial on the rotary encoder is a dud
    rote = {'stim_on': tre[evt.sm_ev == 2][:-1],
            'closed_loop': tre[evt.sm_ev == 3][:-1]}
    bpod = {
        'stim_on': np.array([tr['behavior_data']['States timestamps']
                             ['stim_on'][0][0] for tr in bpod_data]),
        'closed_loop': np.array([tr['behavior_data']['States timestamps']
                                 ['closed_loop'][0][0] for tr in bpod_data]),
    }
    if rote['closed_loop'].size <= 1:
        raise err.SyncBpodWheelException("Not enough Rotary Encoder events to perform wheel"
                                         " synchronization. Wheel data not extracted")
    # bpod bug that spits out events in ms instead of us
    if np.diff(bpod['closed_loop'][[-1, 0]])[0] / np.diff(rote['closed_loop'][[-1, 0]])[0] > 900:
        logger_.error("Rotary encoder stores values in ms instead of us. Wheel timing inaccurate")
        rote['stim_on'] *= 1e3
        rote['closed_loop'] *= 1e3
    # just use the closed loop for synchronization
    # handle different sizes in synchronization:
    sz = min(rote['closed_loop'].size, bpod['closed_loop'].size)
    # if all the sample are contiguous and first samples match
    diff_first_match = np.diff(rote['closed_loop'][:sz]) - np.diff(bpod['closed_loop'][:sz])
    # if all the sample are contiguous and last samples match
    diff_last_match = np.diff(rote['closed_loop'][-sz:]) - np.diff(bpod['closed_loop'][-sz:])
    # 99% of the pulses match for a first sample lock
    DIFF_THRESHOLD = 0.005
    if np.mean(np.abs(diff_first_match) < DIFF_THRESHOLD) > 0.99:
        re = rote['closed_loop'][:sz]
        bp = bpod['closed_loop'][:sz]
        indko = np.where(np.abs(diff_first_match) >= DIFF_THRESHOLD)[0]
    # 99% of the pulses match for a last sample lock
    elif np.mean(np.abs(diff_last_match) < DIFF_THRESHOLD) > 0.99:
        re = rote['closed_loop'][-sz:]
        bp = bpod['closed_loop'][-sz:]
        indko = np.where(np.abs(diff_last_match) >= DIFF_THRESHOLD)[0]
    # last resort is to use ad-hoc sync function
    else:
        bp, re = raw.sync_trials_robust(bpod['closed_loop'], rote['closed_loop'],
                                        diff_threshold=DIFF_THRESHOLD, max_shift=5)
        indko = np.array([])
        # raise ValueError("Can't sync bpod and rotary encoder: non-contiguous sync pulses")
    # remove faulty indices due to missing or bad syncs
    indko = np.int32(np.unique(np.r_[indko + 1, indko]))
    re = np.delete(re, indko)
    bp = np.delete(bp, indko)
    # check the linear drift
    assert bp.size > 1
    poly = np.polyfit(bp, re, 1)
    assert np.all(np.abs(np.polyval(poly, bp) - re) < 0.002)
    return interpolate.interp1d(re, bp, fill_value="extrapolate")


def get_wheel_data(session_path, bp_data=None, save=False, display=False):
    """
    Get wheel data from raw files and converts positions into radians mathematical convention
     (anti-clockwise = +) and timestamps into seconds relative to Bpod clock.
    **Optional:** saves _ibl_wheel.times.npy and _ibl_wheel.position.npy

    Times:
    Gets Rotary Encoder timestamps (us) for each position and converts to times.
    Synchronize with Bpod and outputs

    Positions:
    Radians mathematical convention

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
    data['re_pos'] = df.re_pos.values * -1  # anti-clockwise is positive in our output
    data['re_pos'] = data['re_pos'] / 1024 * 2 * np.pi  # convert positions to radians
    trial_starts = get_trial_start_times(session_path)
    # need a flag if the data resolution is 1ms due to the old version of rotary encoder firmware
    if np.all(np.mod(data['re_ts'], 1e3) == 0):
        status = 1
    data['re_ts'] = data['re_ts'] / 1e6  # convert ts to seconds
    # # get the converter function to translate re_ts into behavior times
    re2bpod = sync_rotary_encoder(session_path)
    data['re_ts'] = re2bpod(data['re_ts'])

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

    #  DEBUG PLOTS START HERE ########################
    if display:
        import matplotlib.pyplot as plt
        plt.figure()
        ax = plt.axes()
        tstart = get_trial_start_times(session_path)
        tts = np.c_[tstart, tstart, tstart + np.nan].flatten()
        vts = np.c_[tstart * 0 + 100, tstart * 0 - 100, tstart + np.nan].flatten()
        ax.plot(tts, vts, label='Trial starts')
        ax.plot(re2bpod(df.re_ts.values / 1e6), df.re_pos.values / 1024 * 2 * np.pi,
                '.-', label='Raw data')
        i0 = np.where(df.re_pos.values == 0)
        ax.plot(re2bpod(df.re_ts.values[i0] / 1e6), df.re_pos.values[i0] / 1024 * 2 * np.pi,
                'r*', label='Raw data zero samples')
        ax.plot(re2bpod(df.re_ts.values / 1e6), tr_dc, label='reset compensation')
        ax.set_xlabel('Bpod Time')
        ax.set_ylabel('radians')
        # restarts = np.array(bp_data[10]['behavior_data']['States timestamps']
        #                             ['reset_rotary_encoder']).flatten()
        # x__ = np.c_[restarts, restarts, restarts + np.nan].flatten()
        # y__ = np.c_[restarts * 0 + 1, restarts * 0 - 1, restarts+ np.nan].flatten()
        # ax.plot(x__, y__, 'k', label='Restarts')
        ax.plot(data['re_ts'], data['re_pos'] / WHEEL_RADIUS_CM, '.-', label='Output Trace')
        ax.legend()
        # plt.hist(np.diff(data['re_ts']), 400, range=[0, 0.01])

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

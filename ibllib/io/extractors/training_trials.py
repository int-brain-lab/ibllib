#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Wednesday, July 18th 2018, 9:53:59 am
"""**ALF extractors** are a collection of functions that extract alf files from
the PyBpod rig raw data.

Each DatasetType in the IBL pipeline should have one extractor function.
"""
import ibllib.io.raw_data_loaders as raw
import numpy as np
import os
import logging
from pathlib import Path
from ibllib.misc import version

logger_ = logging.getLogger('ibllib.alf')


def get_port_events(trial: dict, name: str = '') -> list:
    out: list = []
    for k in trial['behavior_data']['Events timestamps']:
        if name in k:
            out.extend(trial['behavior_data']['Events timestamps'][k])
    out = sorted(out)

    return out


def check_alf_folder(session_path):
    """
    Check if alf folder exists, creates it if it doesn't.

    :param session_path: absolute path of session folder
    :type session_path: str
    """
    alf_folder = os.path.join(session_path, 'alf')
    if not os.path.exists(alf_folder):
        os.mkdir(alf_folder)


def get_feedbackType(session_path, save=False, data=False, settings=False):
    """
    Get the feedback that was delivered to subject.
    **Optional:** saves _ibl_trials.feedbackType.npy

    Checks in raw datafile for error and reward state.
    Will raise an error if more than one of the mutually exclusive states have
    been triggered.

    Sets feedbackType to -1 if error state was trigered (applies to no-go)
    Sets feedbackType to +1 if reward state was triggered

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('int64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    feedbackType = np.empty(len(data))
    feedbackType.fill(np.nan)
    reward = []
    error = []
    no_go = []
    for t in data:
        reward.append(~np.isnan(t['behavior_data']['States timestamps']['reward'][0][0]))
        error.append(~np.isnan(t['behavior_data']['States timestamps']['error'][0][0]))
        no_go.append(~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0]))

    if not all(np.sum([reward, error, no_go], axis=0) == np.ones(len(data))):
        raise ValueError

    feedbackType[reward] = 1
    feedbackType[error] = -1
    feedbackType[no_go] = -1
    feedbackType = feedbackType.astype('int64')
    if raw.save_bool(save, '_ibl_trials.feedbackType.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.feedbackType.npy')
        np.save(fpath, feedbackType)
    return feedbackType


def get_contrastLR(session_path, save=False, data=False, settings=False):
    """
    Get left and right contrasts from raw datafile. Optionally, saves
    _ibl_trials.contrastLeft.npy and _ibl_trials.contrastRight.npy to alf folder.

    Uses signed_contrast to create left and right contrast vectors.

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    contrastLeft = np.array([t['contrast']['value'] if np.sign(
        t['position']) < 0 else np.nan for t in data])
    contrastRight = np.array([t['contrast']['value'] if np.sign(
        t['position']) > 0 else np.nan for t in data])
    # save if needed
    check_alf_folder(session_path)
    if raw.save_bool(save, '_ibl_trials.contrastLeft.npy'):
        lpath = os.path.join(session_path, 'alf', '_ibl_trials.contrastLeft.npy')
        np.save(lpath, contrastLeft)

    if raw.save_bool(save, '_ibl_trials.contrastRight.npy'):
        rpath = os.path.join(session_path, 'alf', '_ibl_trials.contrastRight.npy')
        np.save(rpath, contrastRight)

    return (contrastLeft, contrastRight)


def get_probabilityLeft(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    pLeft = np.array([t['stim_probability_left'] for t in data])
    if raw.save_bool(save, '_ibl_trials.probabilityLeft.npy'):
        lpath = Path(session_path).joinpath('alf', '_ibl_trials.probabilityLeft.npy')
        np.save(lpath, pLeft)
    return pLeft


def get_choice(session_path, save=False, data=False, settings=False):
    """
    Get the subject's choice in every trial.
    **Optional:** saves _ibl_trials.choice.npy to alf folder.

    Uses signed_contrast and trial_correct.
    -1 is a CCW turn (towards the left)
    +1 is a CW turn (towards the right)
    0 is a no_go trial
    If a trial is correct the choice of the animal was the inverse of the sign
    of the position.

    >>> choice[t] = -np.sign(position[t]) if trial_correct[t]

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('int64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    sitm_side = np.array([np.sign(t['position']) for t in data])
    trial_correct = np.array([t['trial_correct'] for t in data])
    trial_nogo = np.array(
        [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
         for t in data])
    choice = sitm_side.copy()
    choice[trial_correct] = -choice[trial_correct]
    choice[trial_nogo] = 0
    choice = choice.astype(int)
    if raw.save_bool(save, '_ibl_trials.choice.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.choice.npy')
        np.save(fpath, choice)
    return choice


def get_repNum(session_path, save=False, data=False, settings=False):
    """
    Count the consecutive repeated trials.
    **Optional:** saves _ibl_trials.repNum.npy to alf folder.

    Creates trial_repeated from trial['contrast']['type'] == 'RepeatContrast'

    >>> trial_repeated = [0, 1, 1, 0, 1, 0, 1, 1, 1, 0]
    >>> repNum =         [0, 1, 2, 0, 1, 0, 1, 2, 3, 0]

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('int64')
    """

    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    trial_repeated = np.array(
        [t['contrast']['type'] == 'RepeatContrast' for t in data])
    trial_repeated = trial_repeated.astype(int)
    repNum = trial_repeated.copy()
    c = 0
    for i in range(len(trial_repeated)):
        if trial_repeated[i] == 0:
            c = 0
            repNum[i] = 0
            continue
        c += 1
        repNum[i] = c
    if raw.save_bool(save, '_ibl_trials.repNum.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.repNum.npy')
        np.save(fpath, repNum)
    return repNum


def get_rewardVolume(session_path, save=False, data=False, settings=False):
    """
    Load reward volume delivered for each trial.
    **Optional:** saves _ibl_trials.rewardVolume.npy

    Uses reward_current to accumulate the amount of

    :param session_path: Absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('int64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    trial_volume = [x['reward_amount']
                    if x['trial_correct'] else 0 for x in data]
    rewardVolume = np.array(trial_volume).astype(np.float64)
    assert len(rewardVolume) == len(data)
    if raw.save_bool(save, '_ibl_trials.rewardVolume.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.rewardVolume.npy')
        np.save(fpath, rewardVolume)
    return rewardVolume


def get_feedback_times_lt5(session_path, data=False):
    if not data:
        data = raw.load_data(session_path)
    rw_times = [tr['behavior_data']['States timestamps']['reward'][0][0]
                for tr in data]
    err_times = [tr['behavior_data']['States timestamps']['error'][0][0]
                 for tr in data]
    nogo_times = [tr['behavior_data']['States timestamps']['no_go'][0][0]
                  for tr in data]
    assert sum(np.isnan(rw_times) &
               np.isnan(err_times) & np.isnan(nogo_times)) == 0
    merge = np.array([np.array(times)[~np.isnan(times)] for times in
                      zip(rw_times, err_times, nogo_times)]).squeeze()

    return np.array(merge)


def get_feedback_times_ge5(session_path, data=False):
    # ger err and no go trig times -- look for BNC2High of trial -- verify
    # only 2 onset times go tone and noise, select 2nd/-1 OR select the one
    # that is grater than the nogo or err trial onset time
    if not data:
        data = raw.load_data(session_path)
    rw_times = [tr['behavior_data']['States timestamps']['reward'][0][0]
                for tr in data]
    sound_times = [tr['behavior_data']['Events timestamps']['BNC2High'] for tr in data]

    err_sound_times = [x[-1] if len(x) == 2 else np.nan for x in sound_times]

    assert sum(np.isnan(rw_times) &
               np.isnan(err_sound_times)) == 0
    merge = np.array([np.array(times)[~np.isnan(times)] for times in
                      zip(rw_times, err_sound_times)]).squeeze()
    return np.array(merge)


def get_feedback_times(session_path, save=False, data=False, settings=False):
    """
    Get the times the water or error tone was delivered to the animal.
    **Optional:** saves _ibl_trials.feedback_times.npy

    Gets reward  and error state init times vectors,
    checks if theintersection of nans is empty, then
    merges the 2 vectors.

    :param session_path: Absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    # Version check
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        merge = get_feedback_times_ge5(session_path, data=data)
    else:
        merge = get_feedback_times_lt5(session_path, data=data)

    if raw.save_bool(save, '_ibl_trials.feedback_times.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.feedback_times.npy')
        np.save(fpath, merge)
    return np.array(merge)


def get_stimOnTrigger_times(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}
    # Get the stim_on_state that triggers the onset of the stim
    stim_on_state = np.array([tr['behavior_data']['States timestamps']
                             ['stim_on'][0] for tr in data])
    stimOnTrigger_times = stim_on_state[:, 0].T

    if raw.save_bool(save, '_ibl_trials.stimOnTrigger_times.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.stimOnTrigger_times.npy')
        np.save(fpath, np.array(stimOnTrigger_times))

    return stimOnTrigger_times


def get_stimOn_times_ge5(session_path, data=False):
    """
    Find first and last stim_sync pulse of the trial.
    stimOn_times should be the first after the stim_on state.
    (Stim updates are in BNC1High and BNC1Low - frame2TTL device)
    Check that all trials have frame changes.
    Find length of stim_on_state [start, stop].
    If either check fails the HW device failed to detect the stim_sync square change
    Substitute that trial's missing or incorrect value with a NaN.
    return stimOn_times
    """
    if not data:
        data = raw.load_data(session_path)
    # Get all stim_sync events detected
    stim_sync_all = [raw.get_port_events(
        tr['behavior_data']['Events timestamps'], 'BNC1') for tr in data]
    stim_sync_all = [np.array(x) for x in stim_sync_all]
    # Get the stim_on_state that triggers the onset of the stim
    stim_on_state = np.array([tr['behavior_data']['States timestamps']
                             ['stim_on'][0] for tr in data])

    stimOn_times = np.array([])
    for sync, on, off in zip(
            stim_sync_all, stim_on_state[:, 0], stim_on_state[:, 1]):
        pulse = sync[np.where(np.bitwise_and((sync > on), (sync <= off)))]
        if pulse.size == 0:
            stimOn_times = np.append(stimOn_times, np.nan)
        else:
            stimOn_times = np.append(stimOn_times, pulse)

    nmissing = np.sum(np.isnan(stimOn_times))
    # Check if all stim_syncs have failed to be detected
    if np.all(np.isnan(stimOn_times)):
        logger_.error(f'{session_path}: Missing ALL BNC1 stimulus ({nmissing} trials')

    # Check if any stim_sync has failed be detected for every trial
    if np.any(np.isnan(stimOn_times)):
        logger_.warning(f'{session_path}: Missing BNC1 stimulus on {nmissing} trials')

    return stimOn_times


def get_stimOn_times_lt5(session_path, data=False):
    """
    Find the time of the statemachine command to turn on hte stim
    (state stim_on start or rotary_encoder_event2)
    Find the next frame change from the photodiodeafter that TS.
    Screen is not displaying anything until then.
    (Frame changes are in BNC1High and BNC1Low)
    """
    if not data:
        data = raw.load_data(session_path)
    stim_on = []
    bnc_h = []
    bnc_l = []
    for tr in data:
        stim_on.append(tr['behavior_data']['States timestamps']['stim_on'][0][0])
        if 'BNC1High' in tr['behavior_data']['Events timestamps'].keys():
            bnc_h.append(np.array(tr['behavior_data']
                                  ['Events timestamps']['BNC1High']))
        else:
            bnc_h.append(np.array([np.NINF]))
        if 'BNC1Low' in tr['behavior_data']['Events timestamps'].keys():
            bnc_l.append(np.array(tr['behavior_data']
                                  ['Events timestamps']['BNC1Low']))
        else:
            bnc_l.append(np.array([np.NINF]))

    stim_on = np.array(stim_on)
    bnc_h = np.array(bnc_h)
    bnc_l = np.array(bnc_l)

    count_missing = 0
    stimOn_times = np.zeros_like(stim_on)
    for i in range(len(stim_on)):
        hl = np.sort(np.concatenate([bnc_h[i], bnc_l[i]]))
        stot = hl[hl > stim_on[i]]
        if np.size(stot) == 0:
            stot = np.array([np.nan])
            count_missing += 1
        stimOn_times[i] = stot[0]

    if np.all(np.isnan(stimOn_times)):
        logger_.error(f'{session_path}: Missing ALL BNC1 stimulus ({count_missing} trials')

    if count_missing > 0:
        logger_.warning(f'{session_path}: Missing BNC1 stimulus on {count_missing} trials')

    return np.array(stimOn_times)


def get_stimOn_times(session_path, save=False, data=False, settings=False):
    """
    Find the time of the statemachine command to turn on hte stim
    (state stim_on start or rotary_encoder_event2)
    Find the next frame change from the photodiodeafter that TS.
    Screen is not displaying anything until then.
    (Frame changes are in BNC1High and BNC1Low)
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}
    # Version check
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        stimOn_times = get_stimOn_times_ge5(session_path, data=data)
    else:
        stimOn_times = get_stimOn_times_lt5(session_path, data=data)

    if raw.save_bool(save, '_ibl_trials.stimOn_times.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.stimOn_times.npy')
        np.save(fpath, np.array(stimOn_times))

    return np.array(stimOn_times)


def get_intervals(session_path, save=False, data=False, settings=False):
    """
    Trial start to trial end. Trial end includes 1 or 2 seconds after feedback,
    (depending on the feedback) and 0.5 seconds of iti.
    **Optional:** saves _ibl_trials.intervals.npy

    Uses the corrected Trial start and Trial end timpestamp values form PyBpod.

    :param session_path: Absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: 2D numpy.ndarray (col0 = start, col1 = end)
    :rtype: dtype('float64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    starts = [t['behavior_data']['Trial start timestamp'] for t in data]
    ends = [t['behavior_data']['Trial end timestamp'] for t in data]
    intervals = np.array([starts, ends]).T
    if raw.save_bool(save, '_ibl_trials.intervals.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.intervals.npy')
        np.save(fpath, intervals)
    return intervals


def get_iti_duration(session_path, save=False, data=False, settings=False):
    """
    Calculate duration of iti from state timestamps.
    **Optional:** saves _ibl_trials.iti_duration.npy

    Uses Trial end timestamp and get_response_times to calculate iti.

    :param session_path: Absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    rt = get_response_times(session_path, save=False, data=False)
    ends = np.array([t['behavior_data']['Trial end timestamp'] for t in data])

    iti_dur = ends - rt
    if raw.save_bool(save, '_ibl_trials.itiDuration.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.itiDuration.npy')
        np.save(fpath, iti_dur)
    return iti_dur


def get_response_times(session_path, save=False, data=False, settings=False):
    """
    Time (in absolute seconds from session start) when a response was recorded.
    **Optional:** saves _ibl_trials.response_times.npy

    Uses the timestamp of the end of the closed_loop state.

    :param session_path: Absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    rt = np.array([tr['behavior_data']['States timestamps']['closed_loop'][0][1]
                   for tr in data])
    if raw.save_bool(save, '_ibl_trials.response_times.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.response_times.npy')
        np.save(fpath, rt)
    return rt


def get_goCueTrigger_times(session_path, save=False, data=False, settings=False):
    """
    Get trigger times of goCue from state machine.

    Current software solution for triggering sounds uses PyBpod soft codes.
    Delays can be in the order of 10's of ms. This is the time when the command
    to play the sound was executed. To measure accurate time, either getting the
    sound onset from xonar soundcard sync pulse (latencies may vary).

    :param session_path: Absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    # Version check
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        goCue = np.array([tr['behavior_data']['States timestamps']
                          ['play_tone'][0][0] for tr in data])
    else:
        goCue = np.array([tr['behavior_data']['States timestamps']
                         ['closed_loop'][0][0] for tr in data])

    if raw.save_bool(save, '_ibl_trials.goCue_times.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.goCueTrigger_times.npy')
        np.save(fpath, goCue)
    return goCue


def get_goCueOnset_times(session_path, save=False, data=False, settings=False):
    """
    Get trigger times of goCue from state machine.

    Current software solution for triggering sounds uses PyBpod soft codes.
    Delays can be in the order of 10's of ms. This is the time when the command
    to play the sound was executed. To measure accurate time, either getting the
    sound onset from the future microphone OR the new xonar soundcard and
    setup developed by Sanworks guarantees a set latency (in testing).

    :param session_path: Absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    go_cue_times = []
    for tr in data:
        if get_port_events(tr, 'BNC2'):
            go_cue_times.append(tr['behavior_data']['Events timestamps']['BNC2High'][0])
        else:
            go_cue_times.append(np.nan)

    go_cue_times = np.array(go_cue_times)

    nmissing = np.sum(np.isnan(go_cue_times))
    # Check if all stim_syncs have failed to be detected
    if np.all(np.isnan(go_cue_times)):
        logger_.error(f'{session_path}: Missing ALL BNC1 stimulus ({nmissing} trials')

    # Check if any stim_sync has failed be detected for every trial
    if np.any(np.isnan(go_cue_times)):
        logger_.warning(f'{session_path}: Missing BNC1 stimulus on {nmissing} trials')

    if raw.save_bool(save, '_ibl_trials.goCue_times.npy'):
        check_alf_folder(session_path)
        fpath = Path(session_path).joinpath('alf', '_ibl_trials.goCue_times.npy')
        np.save(fpath, go_cue_times)
    return go_cue_times


def get_included_trials_lt5(session_path, data=False):
    if not data:
        data = raw.load_data(session_path)

    trials_included = np.array([True for t in data])

    return trials_included


def get_included_trials_ge5(session_path, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    trials_included = np.array([True for t in data])
    if ('SUBJECT_DISENGAGED_TRIGGERED' in settings.keys() and settings[
            'SUBJECT_DISENGAGED_TRIGGERED'] is not False):
        idx = settings['SUBJECT_DISENGAGED_TRIALNUM'] - 1
        trials_included[idx:] = False
    return trials_included


def get_included_trials(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        trials_included = get_included_trials_ge5(session_path, data=data, settings=settings)
    else:
        trials_included = get_included_trials_lt5(session_path, data=data)

    if raw.save_bool(save, '_ibl_trials.included'):
        fpath = Path(session_path).joinpath('alf', '_ibl_trials.included.npy')
        np.save(fpath, trials_included)
    return trials_included


def extract_all(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    # Version check
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        feedbackType = get_feedbackType(
            session_path, save=save, data=data, settings=settings)
        contrastLeft, contrastRight = get_contrastLR(
            session_path, save=save, data=data, settings=settings)
        probabilityLeft = get_probabilityLeft(
            session_path, save=save, data=data, settings=settings)
        choice = get_choice(
            session_path, save=save, data=data, settings=settings)
        repNum = get_repNum(
            session_path, save=save, data=data, settings=settings)
        rewardVolume = get_rewardVolume(
            session_path, save=save, data=data, settings=settings)
        feedback_times = get_feedback_times(
            session_path, save=save, data=data, settings=settings)
        stimOnTrigger_times = get_stimOnTrigger_times(
            session_path, save=save, data=data, settings=settings)
        stimOn_times = get_stimOn_times(
            session_path, save=save, data=data, settings=settings)
        intervals = get_intervals(
            session_path, save=save, data=data, settings=settings)
        response_times = get_response_times(
            session_path, save=save, data=data, settings=settings)
        trials_included = get_included_trials(
            session_path, save=save, data=data, settings=settings)
        go_cue_trig_times = get_goCueTrigger_times(
            session_path, save=save, data=data, settings=settings)
        go_cue_times = get_goCueOnset_times(
            session_path, save=save, data=data, settings=settings)
        out = {'feedbackType': feedbackType,
               'contrastLeft': contrastLeft,
               'contrastRight': contrastRight,
               'probabilityLeft': probabilityLeft,
               'session_path': session_path,
               'choice': choice,
               'repNum': repNum,
               'rewardVolume': rewardVolume,
               'feedback_times': feedback_times,
               'stimOnTrigger_times': stimOnTrigger_times,
               'stimOn_times': stimOn_times,
               'intervals': intervals,
               'response_times': response_times,
               'trials_included': trials_included,
               'goCue_times': go_cue_times,
               'goCueTrigger_times': go_cue_trig_times}
    else:
        feedbackType = get_feedbackType(
            session_path, save=save, data=data, settings=settings)
        contrastLeft, contrastRight = get_contrastLR(
            session_path, save=save, data=data, settings=settings)
        probabilityLeft = get_probabilityLeft(
            session_path, save=save, data=data, settings=settings)
        choice = get_choice(
            session_path, save=save, data=data, settings=settings)
        repNum = get_repNum(
            session_path, save=save, data=data, settings=settings)
        rewardVolume = get_rewardVolume(
            session_path, save=save, data=data, settings=settings)
        feedback_times = get_feedback_times(
            session_path, save=save, data=data, settings=settings)
        stimOn_times = get_stimOn_times(
            session_path, save=save, data=data, settings=settings)
        intervals = get_intervals(
            session_path, save=save, data=data, settings=settings)
        response_times = get_response_times(
            session_path, save=save, data=data, settings=settings)
        iti_dur = get_iti_duration(
            session_path, save=save, data=data, settings=settings)
        trials_included = get_included_trials(
            session_path, save=save, data=data, settings=settings)
        go_cue_trig_times = get_goCueTrigger_times(
            session_path, save=save, data=data, settings=settings)
        go_cue_times = get_goCueOnset_times(
            session_path, save=save, data=data, settings=settings)
        out = {'feedbackType': feedbackType,
               'contrastLeft': contrastLeft,
               'contrastRight': contrastRight,
               'probabilityLeft': probabilityLeft,
               'session_path': session_path,
               'choice': choice,
               'repNum': repNum,
               'rewardVolume': rewardVolume,
               'feedback_times': feedback_times,
               'stimOn_times': stimOn_times,
               'intervals': intervals,
               'response_times': response_times,
               'iti_dur': iti_dur,
               'trials_included': trials_included,
               'goCue_times': go_cue_times,
               'goCueTrigger_times': go_cue_trig_times}
    return out


if __name__ == "__main__":
    sess = '/mnt/s0/IntegrationTests/Subjects_init/ZM_1085/2019-02-12/002'
    sett = raw.load_settings(sess)
    if 'training' in sett['PYBPOD_PROTOCOL']:
        l, r = get_contrastLR(sess, save=False, data=False, settings=False)
        choice = get_choice(sess, save=False, data=False, settings=False)
    print('42')

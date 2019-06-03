#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Wednesday, July 18th 2018, 9:53:59 am
"""**ALF extractors** are a collection of functions that extract alf files from
the PyBpod rig raw data.

Each DatasetType in the IBL pipeline should have one extractor function.

:raises an: n/a
:raises ValueError: n/a
:return: n/a
:rtype: n/a
"""
import ibllib.io.raw_data_loaders as raw
import numpy as np
import os
import logging
from pathlib import Path

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


def get_feedbackType(session_path, save=False, data=False):
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
    feedbackType = np.empty(len(data))
    feedbackType.fill(np.nan)
    reward = []
    error = []
    no_go = []
    for t in data:
        reward.append(~np.isnan(t['behavior_data']
                                ['States timestamps']['reward'][0][0]))
        error.append(~np.isnan(t['behavior_data']
                               ['States timestamps']['error'][0][0]))
        no_go.append(~np.isnan(t['behavior_data']
                               ['States timestamps']['no_go'][0][0]))

    if not all(np.sum([reward, error, no_go], axis=0) == np.ones(len(data))):
        raise ValueError

    feedbackType[reward] = 1
    feedbackType[error] = -1
    feedbackType[no_go] = -1
    feedbackType = feedbackType.astype('int64')
    if raw.save_bool(save, '_ibl_trials.feedbackType.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.feedbackType.npy')
        np.save(fpath, feedbackType)
    return feedbackType


def get_contrastLR(session_path, save=False, data=False):
    """
    Get left and right contrasts from raw datafile
    **Optional:** save _ibl_trials.contrastLeft.npy and
        _ibl_trials.contrastRight.npy to alf folder.

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


def get_probaLR(session_path, save=False, data=False):
    if not data:
        data = raw.load_data(session_path)
    pLeft = np.array([t['stim_probability_left'] for t in data])
    pRight = 1 - pLeft
    if raw.save_bool(save, '_ibl_trials.probabilityLeft.npy'):
        lpath = Path(session_path).joinpath('alf', '_ibl_trials.probabilityLeft.npy')
        np.save(lpath, pLeft)
    return pLeft, pRight


def get_choice(session_path, save=False, data=False):
    """
    Get the subject's choice in every trial.
    **Optional:** saves _ibl_trials.choice.npy to alf folder.

    Uses signed_contrast and trial_correct.
    -1 is a CCW turn (towards the left)
    +1 is a CW turn (towards the right)
    0 is a no_go trial
    If a trial is correct the choice of the animal was the inverse of the sign
    of the contrast.

    >>> choice[t] = -np.sign(signed_contrast[t]) if trial_correct[t]

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
    sitm_side = np.array([np.sign(t['position']) for t in data])
    trial_correct = np.array([t['trial_correct'] for t in data])
    trial_nogo = []
    for t in data:
        trial_nogo.append(~np.isnan(t['behavior_data']['States timestamps']
                                    ['no_go'][0][0]))
    trial_nogo = np.array(trial_nogo)
    choice = sitm_side.copy()
    choice[trial_correct] = -choice[trial_correct]
    choice[trial_nogo] = 0
    choice = choice.astype(int)
    if raw.save_bool(save, '_ibl_trials.choice.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.choice.npy')
        np.save(fpath, choice)
    return choice


def get_repNum(session_path, save=False, data=False):
    """
    Count the consecutive repeated trials.
    **Optional:** saves _ibl_trials.repNum.npy to alf folder.

    Creates trial_repeated from trial['contrast']['type'] == 'repeat_contrast'

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
    trial_repeated = np.array(
        [t['contrast']['type'] == 'repeat_contrast' for t in data])
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


def get_rewardVolume(session_path, save=False, data=False):
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
    trial_volume = [x['reward_amount']
                    if x['trial_correct'] else 0 for x in data]
    rewardVolume = np.array(trial_volume).astype(np.float64)
    assert len(rewardVolume) == len(data)
    if raw.save_bool(save, '_ibl_trials.rewardVolume.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.rewardVolume.npy')
        np.save(fpath, rewardVolume)
    return rewardVolume


def get_feedback_times(session_path, save=False, data=False):
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
    if raw.save_bool(save, '_ibl_trials.feedback_times.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.feedback_times.npy')
        np.save(fpath, merge)
    return np.array(merge)


def get_stimOn_times(session_path, save=False, data=False):
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
        logger_.error(f'{session_path}: Missing ALL BNC1 stimulus ({count_missing}: trials')
        return None

    if count_missing > 0:
        logger_.warning(f'{session_path}: Missing BNC1 stimulus on {count_missing} trials')

    if raw.save_bool(save, '_ibl_trials.stimOn_times.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.stimOn_times.npy')
        np.save(fpath, np.array(stimOn_times))

    return np.array(stimOn_times)


def get_intervals(session_path, save=False, data=False):
    """
    Trial start to trial end. Trial end includes 1 or 2 seconds of iti depending
    on if the trial was correct or not.
    TODO: Nick suggested the that the iti be removed from this. In this case the
    end of a trial would be the same as the response time.
    Also consider adding _ibl_trials.iti and _ibl_trials.deadTime
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
    starts = [t['behavior_data']['Trial start timestamp'] for t in data]
    ends = [t['behavior_data']['Trial end timestamp'] for t in data]
    intervals = np.array([starts, ends]).T
    if raw.save_bool(save, '_ibl_trials.intervals.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.intervals.npy')
        np.save(fpath, intervals)
    return intervals


def get_iti_duration(session_path, save=False, data=False):
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
    rt = get_response_times(session_path, save=False, data=False)
    ends = np.array([t['behavior_data']['Trial end timestamp'] for t in data])

    iti_dur = ends - rt
    if raw.save_bool(save, '_ibl_trials.itiDuration.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.itiDuration.npy')
        np.save(fpath, iti_dur)
    return iti_dur


def get_deadTime(session_path, save=False, data=False):
    """
    Get the time between state machine exit and restart of next trial.

    Uses the corrected Trial start and Trial end timpestamp values form PyBpod.

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
    starts = [t['behavior_data']['Trial start timestamp'] for t in data]
    ends = [t['behavior_data']['Trial end timestamp'] for t in data]
    # trial_len = np.array(ends) - np.array(starts)
    deadTime = np.array(starts)[1:] - np.array(ends)[:-1]
    deadTime = np.append(np.array([0]), deadTime)
    if raw.save_bool(save, '_ibl_trials.deadTime.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.deadTime.npy')
        np.save(fpath, deadTime)
    return deadTime


def get_response_times(session_path, save=False, data=False):
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
    rt = np.array([tr['behavior_data']['States timestamps']['closed_loop'][0][1]
                   for tr in data])
    if raw.save_bool(save, '_ibl_trials.response_times.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.response_times.npy')
        np.save(fpath, rt)
    return rt


def get_goCueTrigger_times(session_path, save=False, data=False):
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
    goCue = np.array([tr['behavior_data']['States timestamps']
                      ['closed_loop'][0][0] for tr in data])
    if raw.save_bool(save, '_ibl_trials.goCue_times.npy'):
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.goCueTrigger_times.npy')
        np.save(fpath, goCue)
    return goCue


def get_goCueOnset_times(session_path, save=False, data=False):
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
    go_cue_times = []
    for tr in data:
        if get_port_events(tr, 'BNC2'):
            go_cue_times.append(tr['behavior_data']['Events timestamps']
                                ['BNC2High'][0])
        else:
            go_cue_times.append(np.nan)

    if all(np.isnan(go_cue_times)):
        return

    if raw.save_bool(save, '_ibl_trials.goCue_times.npy'):
        check_alf_folder(session_path)
        fpath = Path(session_path).joinpath('alf', '_ibl_trials.goCue_times.npy')
        np.save(fpath, go_cue_times)
    return go_cue_times


def get_included_trials(session_path, save=False, data=False):
    if not data:
        data = raw.load_data(session_path)
    trials_included = np.array([t['contrast']['type'] != "RepeatContrast" for t in data])
    if raw.save_bool(save, '_ibl_trials.included'):
        fpath = Path(session_path).joinpath('alf', '_ibl_trials.included.npy')
        np.save(fpath, trials_included)
    return trials_included


def extract_all(session_path, save=False, data=False):
    if not data:
        data = raw.load_data(session_path)
    feedbackType = get_feedbackType(session_path, save=save, data=data)
    contrastLeft, contrastRight = get_contrastLR(
        session_path, save=save, data=data)
    probabilityLeft, _ = get_probaLR(session_path, save=save, data=data)
    choice = get_choice(session_path, save=save, data=data)
    repNum = get_repNum(session_path, save=save, data=data)
    rewardVolume = get_rewardVolume(session_path, save=save, data=data)
    feedback_times = get_feedback_times(session_path, save=save, data=data)
    stimOn_times = get_stimOn_times(session_path, save=save, data=data)
    intervals = get_intervals(session_path, save=save, data=data)
    response_times = get_response_times(session_path, save=save, data=data)
    iti_dur = get_iti_duration(session_path, save=save, data=data)
    trials_included = get_included_trials(session_path, save=save, data=data)
    go_cue_trig_times = get_goCueTrigger_times(session_path, save=save, data=data)
    go_cue_times = get_goCueOnset_times(session_path, save=save, data=data)
    # Missing datasettypes
    # _ibl_trials.deadTime
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
        l, r = get_contrastLR(sess, save=False, data=False)
        choice = get_choice(sess, save=False, data=False)
    print('42')

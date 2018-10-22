# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Wednesday, July 18th 2018, 9:53:59 am
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 18-07-2018 09:54:02.022
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


def check_alf_folder(session_path):
    """
    Check if alf folder exists, creates it if it doesn't.

    :param session_path: absolute path of session folder
    :type session_path: str
    """
    alf_folder = os.path.join(session_path, 'alf')
    if not os.path.exists(alf_folder):
        os.mkdir(alf_folder)


def get_feedbackType(session_path, save=False):
    """
    Get the feedback that was delivered to subject.
    **Optional:** saves _ibl_trials.feedbackType.npy

    Checks in raw datafile for error and reward state.
    Will raise an error if more than one of the mutually exclusive states have
    been triggered.

    Sets feedbackType to -1 if error state was trigered
    Sets feedbackType to +1 if reward state was triggered
    Sets feedbackType to 0 if no_go state was triggered

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('int64')
    """
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
    feedbackType[no_go] = 0
    feedbackType = feedbackType.astype('int64')
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.feedbackType.npy')
        np.save(fpath, feedbackType)
    return feedbackType


def get_contrastLR(session_path, save=False):
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
    data = raw.load_data(session_path)
    contrastLeft = np.array([t['signed_contrast'] for t in data])
    contrastRight = contrastLeft.copy()
    contrastLeft[contrastLeft > 0] = np.nan
    contrastLeft = np.abs(contrastLeft)
    contrastRight[contrastRight < 0] = np.nan
    if save:
        check_alf_folder(session_path)
        lpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.contrastLeft.npy')
        rpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.contrastRight.npy')

        np.save(lpath, contrastLeft)
        np.save(rpath, contrastRight)
    return (contrastLeft, contrastRight)


def get_choice(session_path, save=False):
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
    data = raw.load_data(session_path)
    sitm_side = np.array([np.sign(t['signed_contrast']) for t in data])
    trial_correct = np.array([t['trial_correct'] for t in data])
    choice = sitm_side.copy()
    choice[trial_correct] = -choice[trial_correct]
    choice = choice.astype(int)
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.choice.npy')
        np.save(fpath, choice)
    return choice


def get_repNum(session_path, save=False):
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
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_trials.repNum.npy')
        np.save(fpath, repNum)
    return repNum


def get_rewardVolume(session_path, save=False):
    """
    Load reward volume delivered for each trial.
    **Optional:** saves _ibl_trials.rewardVolume.npy

    Uses reward_current to accumulate the amount of

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('int64')
    """
    data = raw.load_data(session_path)
    trial_volume = [x['reward_current']
                    if x['trial_correct'] else 0 for x in data]
    rewardVolume = np.array(trial_volume).astype(np.float64)
    assert len(rewardVolume) == len(data)
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.rewardVolume.npy')
        np.save(fpath, rewardVolume)
    return rewardVolume


def get_feedback_times(session_path, save=False):
    """
    Get the times the water or error tone was delivered to the animal.
    **Optional:** saves _ibl_trials.feedback_times.npy

    Gets reward  and error state init times vectors,
    checks if theintersection of nans is empty, then
    merges the 2 vectors.

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    data = raw.load_data(session_path)
    rw_times = [tr['behavior_data']['States timestamps']['reward'][0][0]
                for tr in data]
    err_times = [tr['behavior_data']['States timestamps']['error'][0][0]
                 for tr in data]
    assert sum(np.isnan(rw_times) & np.isnan(err_times)) == 0
    merge = [x if ~np.isnan(x) else y for x, y in zip(rw_times, err_times)]
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.feedback_times.npy')
        np.save(fpath, merge)
    return np.array(merge)


def get_stimOn_times(session_path, save=False):
    """
    Find the time of the statemachine command to turn on hte stim
    (state stim_on start or rotary_encoder_event2)
    Find the next frame change from the photodiodeafter that TS.
    Screen is not displaying anything until then.
    (Frame changes are in BNC1High and BNC1Low)
    """
    data = raw.load_data(session_path)
    stim_on = []
    bnc_h = []
    bnc_l = []
    for tr in data:
        stim_on.append(tr['behavior_data']
                        ['States timestamps']['stim_on'][0][0])
        if 'BNC1High' in tr['behavior_data']['Events timestamps'].keys():
            bnc_h.append(np.array(tr['behavior_data']
                         ['Events timestamps']['BNC1High']))
        else:
            bnc_h.append(np.nan)
        if 'BNC1Low' in tr['behavior_data']['Events timestamps'].keys():
            bnc_l.append(np.array(tr['behavior_data']
                         ['Events timestamps']['BNC1Low']))
        else:
            bnc_l.append(np.nan)

    stim_on = np.array(stim_on)
    bnc_h = np.array(bnc_h)
    bnc_l = np.array(bnc_l)

    stimOn_times = []
    for s, h, l in zip(stim_on, bnc_h, bnc_l):
        hl = np.concatenate([h, l])
        hl.sort()
        stimOn_times.extend([hl[hl > s][0]])

    # delays = np.asarray(stimOn_times) - np.asarray(stim_on)

    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.stimOn_times.npy')
        np.save(fpath, np.array(stimOn_times))

    return np.array(stimOn_times)


def get_intervals(session_path, save=False):
    """
    Trial start to trial end. Trial end includes 1 or 2 seconds of iti depending
    on if the trial was correct or not.
    TODO: Nick suggested the that the iti be removed from this. In this case the
    end of a trial would be the same as the response time.
    Also consider adding _ibl_trials.iti and _ibl_trials.deadTime
    **Optional:** saves _ibl_trials.intervals.npy

    Uses the corrected Trial start and Trial end timpestamp values form PyBpod.

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: 2D numpy.ndarray (col0 = start, col1 = end)
    :rtype: dtype('float64')
    """
    data = raw.load_data(session_path)
    starts = [t['behavior_data']['Trial start timestamp'] for t in data]
    ends = [t['behavior_data']['Trial end timestamp'] for t in data]
    intervals = np.array([starts, ends]).T
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.intervals.npy')
        np.save(fpath, intervals)
    return intervals


def get_iti_duration(session_path, save=False):
    """
    Calculate duration of iti from state timestamps.
    **Optional:** saves _ibl_trials.iti_duration.npy

    Uses Trial end timestamp and get_response_times to calculate iti.

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    data = raw.load_data(session_path)
    rt = get_response_times(session_path, save=False)
    ends = np.array([t['behavior_data']['Trial end timestamp'] for t in data])

    iti_dur = ends - rt
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.iti_duration.npy')
        np.save(fpath, iti_dur)
    return iti_dur


def get_deadTime(session_path, save=False):
    """
    Get the time between state machine exit and restart of next trial.

    Uses the corrected Trial start and Trial end timpestamp values form PyBpod.

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    data = raw.load_data(session_path)
    starts = [t['behavior_data']['Trial start timestamp'] for t in data]
    ends = [t['behavior_data']['Trial end timestamp'] for t in data]
    # trial_len = np.array(ends) - np.array(starts)
    deadTime = np.array(starts)[1:] - np.array(ends)[:-1]
    deadTime = np.append(np.array([0]), deadTime)
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.deadTime.npy')
        np.save(fpath, deadTime)
    return deadTime


def get_response_times(session_path, save=False):
    """
    Time (in absolute seconds from session start) when a response was recorded.
    **Optional:** saves _ibl_trials.response_times.npy

    Uses the timestamp of the end of the closed_loop state.

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    data = raw.load_data(session_path)
    rt = np.array([tr['behavior_data']['States timestamps']['closed_loop'][0][1]
                   for tr in data])
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.response_times.npy')
        np.save(fpath, rt)
    return rt


def get_goCueTrigger_times(session_path, save=False):
    """
    Get trigger times of goCue from state machine.

    Current software solution for triggering sounds uses PyBpod soft codes.
    Delays can be in the order of 10's of ms. This is the time when the command
    to play the sound was executed. To measure accurate time, either getting the
    sound onset from the future microphone OR the new xonar soundcard and
    setup developed by Sanworks guarantees a set latency (in testing).

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    data = raw.load_data(session_path)
    goCue = np.array([tr['behavior_data']['States timestamps']
                      ['closed_loop'][0][0] for tr in data])
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.goCue_times.npy')
        np.save(fpath, goCue)
    return goCue


def get_goCueOnset_times(session_path, save=False):
    """
    Get trigger times of goCue from state machine.

    Current software solution for triggering sounds uses PyBpod soft codes.
    Delays can be in the order of 10's of ms. This is the time when the command
    to play the sound was executed. To measure accurate time, either getting the
    sound onset from the future microphone OR the new xonar soundcard and
    setup developed by Sanworks guarantees a set latency (in testing).

    :param session_path: Absoulte path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :param save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    data = raw.load_data(session_path)
    goCue = np.array([tr['behavior_data']['States timestamps']
                      ['closed_loop'][0][0] for tr in data])
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.goCue_times.npy')
        np.save(fpath, goCue)
    return goCue


def extract_trials(session_path, save=False):

    feedbackType = get_feedbackType(session_path, save=save)
    contrastLeft, contrastRight = get_contrastLR(session_path, save=save)
    choice = get_choice(session_path, save=save)
    repNum = get_repNum(session_path, save=save)
    rewardVolume = get_rewardVolume(session_path, save=save)
    feedback_times = get_feedback_times(session_path, save=save)
    stimOn_times = get_stimOn_times(session_path, save=save)
    intervals = get_intervals(session_path, save=save)
    response_times = get_response_times(session_path, save=save)
    iti_dur = get_iti_duration(session_path, save=save)


    # Missing datasettypes
    # _ibl_trials.goCue_times
    # _ibl_trials.deadTime
    # _ibl_trials.probabilityLeft

if __name__ == '__main__':
    session_path = "/home/nico/Projects/IBL/IBL-github/iblrig/test_dataset/\
test_mouse/2018-10-02/1"
    save = False

    data = raw.load_data(session_path)

    feedbackType = get_feedbackType(session_path, save=save)
    contrastLeft, contrastRight = get_contrastLR(session_path, save=save)
    choice = get_choice(session_path, save=save)
    repNum = get_repNum(session_path, save=save)
    rewardVolume = get_rewardVolume(session_path, save=save)
    feedback_times = get_feedback_times(session_path, save=save)
    stimOn_times = get_stimOn_times(session_path, save=save)
    intervals = get_intervals(session_path, save=save)
    response_times = get_response_times(session_path, save=save)
    iti_dur = get_iti_duration(session_path, save=save)

    print("Done!")

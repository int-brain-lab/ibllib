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
import raw_data_loaders as raw
import numpy as np
import os


def get_trial_intervals(session_path):
    data = raw.load_data(session_path)
    shift = data[0]['behavior_data']['Bpod start timestamp']
    starts = [t['behavior_data']['Trial start timestamp'] for t in data]
    t['behavior_data']['States timestamps']['error'][0][0]
    t['behavior_data']['States timestamps']['reward'][0][0]


def check_alf_folder(session_path):
    """
    Check if alf folder exists, creates it if it doesn't.

    :param session_path: absolute path of session folder
    :type session_path: str
    """
    alf_folder = os.path.join(session_path, 'alf')
    if not os.path.exists(alf_folder):
        os.mkdir(alf_folder)


def get_trials_feedbackType(session_path, save=False):
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
    feedbackType = feedbackType.astype(int)
    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_trials.feedbackType.npy')
        np.save(fpath, feedbackType)
    return feedbackType


def get_trials_contrastLR(session_path, save=False):
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


def get_trials_choice(session_path, save=False):
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


def get_trials_repNum(session_path, save=False):
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


if __name__ == '__main__':
    SESSION_PATH = "/home/nico/Projects/IBL/IBL-github/IBL_root/pybpod_data/\
test_mouse/2018-07-11/11"

    data = raw.load_data(SESSION_PATH)
    starts = [t['behavior_data']['Trial start timestamp'] for t in data]
    ends = [t['behavior_data']['Trial end timestamp'] for t in data]
    trial_len = np.array(ends) - np.array(starts)
    dead_time = np.array(starts)[1:] - np.array(ends)[:-1]
    dead_time = np.append(np.array([0]), dead_time)

    feedbackType = get_trials_feedbackType(SESSION_PATH, save=False)
    contrastLeft, contrastRight = get_trials_contrastLR(
        SESSION_PATH, save=False)
    choice = get_trials_choice(SESSION_PATH, save=False)
    rep = np.array([t['contrast']['type'] == 'repeat_contrast' for t in data])
    repNum = get_trials_repNum(SESSION_PATH, save=False)
    print(list(zip(rep.astype(int), repNum)))

    ft = os.path.join(SESSION_PATH, 'alf', '_ibl_trials.feedbackType.npy')
    cl = os.path.join(SESSION_PATH, 'alf', '_ibl_trials.contrastLeft.npy')
    cr = os.path.join(SESSION_PATH, 'alf', '_ibl_trials.contrastRight.npy')
    sc = os.path.join(SESSION_PATH, 'alf', '_ibl_trials.choice.npy')
    rn = os.path.join(SESSION_PATH, 'alf', '_ibl_trials.repNum.npy')
    print(np.load(ft).dtype)
    print(np.load(cl).dtype)
    print(np.load(cr).dtype)
    print(np.load(sc).dtype)
    print(np.load(rn).dtype)
    print("Done!")

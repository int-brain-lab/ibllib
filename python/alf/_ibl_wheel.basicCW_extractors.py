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


def get_positions(session_path, save=False):
    """
    radius * 2 * pi / n_ticks
    3.1 * 2 * np.pi / 1024





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


def get_velocity(session_path, save=False):
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


def get_timestamps(session_path, save=False):
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



def extract_wheel(session_path, save=False):
    data = raw.load_data(session_path)
    position = get_positions(session_path, save=save)
    velocity = get_velocity(session_path, save=save)
    timestamps = get_timestamps(session_path, save=save)


if __name__ == '__main__':
    SESSION_PATH = "/home/nico/Projects/IBL/IBL-github/iblrig/Subjects/\
test_mouse/2018-09-19/1"
    save = False

    data = raw.load_data(SESSION_PATH)


    print("Done!")

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
from scipy import interpolate
import pandas as pd


# START of AUXILIARY FUNCS to be refactored out of the extracor files
def check_alf_folder(session_path):
    """
    Check if alf folder exists, creates it if it doesn't.

    :param session_path: absolute path of session folder
    :type session_path: str
    """
    alf_folder = os.path.join(session_path, 'alf')
    if not os.path.exists(alf_folder):
        os.mkdir(alf_folder)


def get_trial_start_times(session_path, save=False):
    data = raw.load_data(session_path)
    trial_start_times = []
    for tr in data:
        trial_start_times.extend(
            [x[0] for x in tr['behavior_data']['States timestamps']['trial_start']])
    return np.array(trial_start_times)


def get_trial_start_times_re(session_path, save=False):
    evt = raw.load_encoder_events(session_path)
    trial_start_times_re = evt.re_ts[evt.sm_ev[evt.sm_ev == 1].index].values / 1000
    return trial_start_times_re[:-1]


def time_converter(session_path, kind='re2b'):
    """
    Create interp1d functions to convert values from one clok to another given a
    set of syncronization pulses.

    The task global sync pulse is at trial_start from Bpod to:
    Rotary Encoder, Cameras and e-phys system.
    Depends on getter functions that extract from the raw data the timestamps
    of the trial_start sync pulse event for each clock.

    kinds:
    *2b:        _________   re2b        cam2b,      ephys2b
    *2re:       b2re        _________   cam2re,     ephys2re
    *2cam:      b2cam       re2cam      _________   ephys2cam
    *2ephys:    b2ephys     re2ephys    cam2ephys   _________

    Default converters for times are assumed to be of kind *2b unless ephys data
    is present in that case converters for 'times' will be of kind *2ephys

    TODO: implement new kinds as needed!

    :param session_path: absolute path of session folder
    :type session_path: str
    :param kind: ['re2b', 'b2re'], defaults to 're2b'
    :type kind: str, optional
    :return: Function that converts from clock A to clock B defined by kind.
    :rtype: scipy.interpolate.interpolate.interp1d
    """
    tst = get_trial_start_times(session_path)
    tst_re = get_trial_start_times_re(session_path)

    btimes_to_re_ts = interpolate.interp1d(
        tst, tst_re, fill_value="extrapolate")
    re_ts_to_times = interpolate.interp1d(
        tst_re, tst, fill_value="extrapolate")

    if kind == 're2b':
        func = re_ts_to_times
    elif kind == 'b2re':
        func = btimes_to_re_ts

    return func
# END of AUXILIARY FUNCS to be refactored out of the extracor files


def get_positions(session_path, save=False):
    """
    Positions in (cm) of RE relative to 0
    **Optional:** saves _ibl_wheel.position.npy

    cmtick = radius (cm) * 2 * pi / n_ticks
    cmtick = 3.1 * 2 * np.pi / 1024

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    pos = raw.load_encoder_positions(session_path)
    cmtick = 3.1 * 2 * np.pi / 1024
    cmpos = pos.re_pos.values * cmtick

    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_wheel.position.npy')
        np.save(fpath, cmpos)
    return cmpos


def get_times(session_path, save=False):
    """
    Gets Rotary Encoder timestamps (ms) for each position and converts to times.
    **Optional:** saves _ibl_wheel.times.npy

    Uses time_converter to extract and convert timstamps (ms) to times (s).

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    re_ts_to_times = time_converter(session_path, kind='re2b')
    pos_df = raw.load_encoder_positions(session_path)
    pos_re_ts = pos_df.re_ts.values / 1000

    rep_idx = np.where(np.diff(pos_re_ts) == 0)[0]


    pos_times = re_ts_to_times(pos_re_ts)

    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_wheel.times.npy')
        np.save(fpath, pos_times)
    return pos_times


def get_time_n_position(session_path, save=False):

    df = raw.load_encoder_positions(session_path)

    cmtick = 3.1 * 2 * np.pi / 1024
    # Convert position and timestamps to cm and seconds respectively
    df.re_pos = df.re_pos.values * cmtick
    df.re_ts = df.re_ts.values / 1000
    # Find timestamps that are repeated
    rep_idx = np.where(np.diff(df.re_ts) == 0)[0]
    df.re_pos[rep_idx] = (df.re_pos[rep_idx].values +
                          df.re_pos[rep_idx+1].values)/2
    # get the converter function to translate re_ts into behavior times
    convtime = time_converter(session_path, kind='re2b')
    df.re_ts = convtime(df.re_ts.values)
    # Now remove the repeted times that are rep_idx + 1


    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf', '_ibl_wheel.times.npy')
        np.save(fpath, pos_times)
    return

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
    cmpos = get_positions(session_path)
    stimes = get_times(session_path)
    dp = np.diff(cmpos)
    dt = np.diff(stimes)
    velocity = dp / dt

    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_wheel.velocity.npy')
        np.save(fpath, velocity)
    return cmpos


def extract_wheel(session_path, save=False):
    position = get_positions(session_path, save=save)
    velocity = get_velocity(session_path, save=save)
    times = get_times(session_path, save=save)


if __name__ == '__main__':
    session_path = "/home/nico/Projects/IBL/IBL-github/iblrig/test_dataset/\
test_mouse/2018-10-02/1"
    save = False

    data = raw.load_data(session_path)
    position = get_positions(session_path, save=save)
    times = get_times(session_path, save=save)
    velocity = get_velocity(session_path, save=save)


    # extract_wheel(session_path, save=save)

    cmpos = get_positions(session_path)
    stimes = get_times(session_path)
    dp = np.diff(cmpos)
    dt = np.diff(stimes)
    td = np.cumsum(dt) + dt/2

    f = interpolate.interp1d(
        np.cumsum(dt) + stimes[1]/2, velocity, fill_value="extrapolate")

    import matplotlib.pyplot as plt
    plt.plot(stimes[1:], dp/dt)


    print("Done!")

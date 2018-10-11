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
import os

import numpy as np
import pandas as pd
from scipy import interpolate

import ibllib.io.raw_data_loaders as raw
from ibllib.misc import structarr


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


def get_wheel_data(session_path, save=False):
    """
    Get wheel data from raw files and converts positions into centimeters and
    timestamps into seconds.
    **Optional:** saves _ibl_wheel.times.npy and _ibl_wheel.position.npy

    Times:
    Gets Rotary Encoder timestamps (ms) for each position and converts to times.

    Uses time_converter to extract and convert timstamps (ms) to times (s).

    Positions:
    Positions are in (cm) of RE perim relative to 0. The 0 resets every trial.

    cmtick = radius (cm) * 2 * pi / n_ticks
    cmtick = 3.1 * 2 * np.pi / 1024

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: wether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: Numpy structuresd array.
    :rtype: numpy.ndarray
    """
    # TODO: move this to raw_data_loaders (use structured arrays instead of
    # pandas DataFrames)
    ############################################################################
    df = raw.load_encoder_positions(session_path)
    names = df.columns.tolist()
    data = structarr(names, shape=(df.index.max() + 1,))
    data['re_ts'] = df.re_ts.values
    data['re_pos'] = df.re_pos.values
    data['bns_ts'] = df.bns_ts.values
    ############################################################################
    # ticks to cm factor
    cmtick = 3.1 * 2 * np.pi / 1024
    # Convert position and timestamps to cm and seconds respectively
    data['re_ts'] = data['re_ts'] / 1000.
    data['re_pos'] = data['re_pos'] * cmtick
    # Find timestamps that are repeated
    rep_idx = np.where(np.diff(data['re_ts']) == 0)[0]
    # Change the value of the repeated position
    data['re_pos'][rep_idx] = (data['re_pos'][rep_idx] +
                               data['re_pos'][rep_idx + 1]) / 2
    # get the converter function to translate re_ts into behavior times
    convtime = time_converter(session_path, kind='re2b')
    data['re_ts'] = convtime(data['re_ts'])
    # Now remove the repeted times that are rep_idx + 1
    data = np.delete(data, rep_idx + 1)

    if save:
        check_alf_folder(session_path)
        tpath = os.path.join(session_path, 'alf', '_ibl_wheel.times.npy')
        ppath = os.path.join(session_path, 'alf', '_ibl_wheel.position.npy')
        np.save(tpath, data['re_ts'])
        np.save(ppath, data['re_pos'])

    return data


def get_velocity(session_path, save=False):
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
    data = get_wheel_data(session_path, save=False)
    dp = np.diff(data['re_pos'])
    dt = np.diff(data['re_ts'])
    # Compute raw velocity
    vel = dp / dt
    # Compute velocity time scale
    td = data['re_ts'][:-1] + dt/2

    # Get the true velocity function
    velocity = interpolate.interp1d(td, vel, fill_value="extrapolate")

    if save:
        check_alf_folder(session_path)
        fpath = os.path.join(session_path, 'alf',
                             '_ibl_wheel.velocity.npy')
        np.save(fpath, velocity(data['re_ts']))

    return velocity(data['re_ts'])


if __name__ == '__main__':
    # function code plus plot on test_dataset
    import matplotlib.pyplot as plt
    session_path = "/home/nico/Projects/IBL/IBL-github/iblrig/test_dataset/\
test_mouse/2018-10-02/1"
    save = False

    data = get_wheel_data(session_path, save=save)
    velocity = get_velocity(session_path, save=save)

    df = raw.load_encoder_positions(session_path)
    names = df.columns.tolist()
    data = structarr(names, shape=(df.index.max() + 1,))
    data['re_ts'] = df.re_ts.values
    data['re_pos'] = df.re_pos.values
    data['bns_ts'] = df.bns_ts.values

    cmtick = 3.1 * 2 * np.pi / 1024
    # Convert position and timestamps to cm and seconds respectively
    data['re_ts'] = data['re_ts'] / 1000.
    data['re_pos'] = data['re_pos'] * cmtick
    # Find timestamps that are repeated
    rep_idx = np.where(np.diff(data['re_ts']) == 0)[0]
    # Change the value of the repeated position
    data['re_pos'][rep_idx] = (data['re_pos'][rep_idx] +
                               data['re_pos'][rep_idx + 1]) / 2
    # get the converter function to translate re_ts into behavior times
    convtime = time_converter(session_path, kind='re2b')
    data['re_ts'] = convtime(data['re_ts'])
    # Now remove the repeted times that are rep_idx + 1
    data = np.delete(data, rep_idx + 1)


    dp = np.diff(data['re_pos'])
    dt = np.diff(data['re_ts'])
    # Compute raw velocity
    vel = dp / dt
    # Compute velocity time scale
    td = data['re_ts'][:-1] + dt/2
    # Get the true velocity function
    velocity = interpolate.interp1d(td, vel, fill_value="extrapolate")
    # Checkit out
    plt.plot(td, vel, '-o')
    plt.plot(data['re_ts'], velocity(data['re_ts']), '-*')

    print("Done!")

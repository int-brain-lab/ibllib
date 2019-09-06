# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:08:01 2019

List of analysis functions for DLC

@author: guido, miles
"""

import numpy as np
from scipy import signal

def lick_times(dlc_data, dlc_data_camera_2=None):
    """
    Lick onset and offset frames.  If more than one dict is provided licks detected by one or the other are counted.
    :param dlc_data: tuple of tongue x and y positions.  If provided only contact with tube is counted as a lick.
    :param dlc_data_camera_2: tuple of tongue x and y positions from other camera angle

    :return: tuple of lick onset and offset frames
    """
    #  Extract tongue and tube arrays
    use_tube = any(map(lambda v: v.startswith('tube'), dlc_data.keys()))
    tongue_end_l = (dlc_data['tongue_end_l_x'], dlc_data['tongue_end_l_y'])
    tongue_end_r = (dlc_data['tongue_end_r_x'], dlc_data['tongue_end_r_y'])
    if use_tube:
        tube_top = (dlc_data['tube_top_y'], dlc_data['tube_top_y'])
        tube_bottom = (dlc_data['tube_bottom_y'], dlc_data['tube_bottom_y'])

    # Find indices of all non nan values
    idx = np.nonzero(np.isfinite(tongue_end_l[0]))
    # Find first and last for each index


    if use_tube:
        pass
    else:
        pass

    print(dlc_data.keys())


def sniff_times(dlc_dict):
    '''
    Sniff onset times.
    :param dlc_dict: Dict containing the following keys:
                  nostril_top_x, nostril_top_y, nostril_bottom_x, nostril_bottom_y
    :return: 1D array of sniff onset times
    '''

    dis = np.sqrt(((dlc_dict['nostril_top_x'] - dlc_dict['nostril_bottom_x'])**2)
                  + ((dlc_dict['nostril_top_y'] - dlc_dict['nostril_bottom_y'])**2))
    win = 4 * dlc_dict['sampling_rate']

    freqs, time, spec = signal.spectrogram(dis, fs=dlc_dict['sampling_rate'],
                                           nperseg=int(win))

    freqs_w, psd = signal.welch(dis, dlc_dict['sampling_rate'], nperseg=win)

    power = np.mean(spec[(freqs > 28) & (freqs < 30)], axis=0)


def fit_circle(x, y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    u = x - x_m
    v = y - y_m
    Suv = np.sum(u*v)
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suuv = np.sum(u**2 * v)
    Suvv = np.sum(u * v**2)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = np.array([Suuu + Suvv, Svvv + Suuv])/2.0
    uc, vc = np.linalg.solve(A, B)
    xc_1 = x_m + uc
    yc_1 = y_m + vc
    Ri_1 = np.sqrt((x-xc_1)**2 + (y-yc_1)**2)
    R_1 = np.mean(Ri_1)
    return xc_1, yc_1, R_1


def pupil_features(dlc_dict):
    vec_x = [dlc_dict['pupil_left_r_x'], dlc_dict['pupil_right_r_x'],
             dlc_dict['pupil_top_r_x'], dlc_dict['pupil_bottom_r_x']]
    vec_y = [dlc_dict['pupil_left_r_y'], dlc_dict['pupil_right_r_y'],
             dlc_dict['pupil_top_r_y'], dlc_dict['pupil_bottom_r_y']]
    x = np.zeros(len(vec_x[0]))
    y = np.zeros(len(vec_x[0]))
    diameter = np.zeros(len(vec_x[0]))
    for i in range(len(vec_x[0])):
        try:
            x[i], y[i], R = fit_circle([vec_x[0][i], vec_x[1][i], vec_x[2][i], vec_x[3][i]],
                                       [vec_y[0][i], vec_y[1][i], vec_y[2][i], vec_y[3][i]])
            diameter[i] = R*2
        except:
            x[i] = np.nan
            y[i] = np.nan
            diameter = np.nan
    return x, y, diameter

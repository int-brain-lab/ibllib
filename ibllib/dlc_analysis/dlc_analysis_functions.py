# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 14:08:01 2019

List of analysis functions for DLC

@author: miles
"""

import numpy as np

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
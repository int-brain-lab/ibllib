# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:48:15 2019

List of basis functions for DLC

@author: guido
"""

import numpy as np
from os.path import join
import alf.io


def load_dlc(folder_path, camera='left'):
    """
    Load in DLC traces and timestamps from FPGA and align them

    Parameters
    ----------
    folder_path: path to top-level folder of recording
    camera: which camera to use

    """

    # Load in DLC data
    dlc_data = alf.io.load_object(join(folder_path, 'alf'), '_ibl_%sCamera' % camera)
    dlc_data['camera'] = camera
    dlc_data['units'] = 'px'

    # Hard-coded hack because extraction of timestamps was wrong
    if camera == 'left':
        camera = 'body'

    # Load in FPGA timestamps
    timestamps = np.load(join(folder_path, 'raw_video_data',
                              '_iblrig_%sCamera.times.npy' % camera))

    # Align FPGA and DLC timestamps
    if len(timestamps) > len(dlc_data[list(dlc_data.keys())[0]]):
        timestamps = timestamps[0:len(dlc_data[list(dlc_data.keys())[0]])]
    elif len(timestamps) < len(dlc_data[list(dlc_data.keys())[0]]):
        for key in list(dlc_data.keys()):
            dlc_data[key] = dlc_data[key][0:len(timestamps)]
    dlc_data['timestamps'] = timestamps

    return dlc_data


def load_event_times(folder_path):
    """
    Load in DLC traces and timestamps from FPGA and align them

    Parameters
    ----------
    folder_path: path to top-level folder of recording
    camera:      which camera to use

    """
    stim_on_times = np.load(join(folder_path, 'alf', '_ibl_trials.stimOn_times.npy'))
    feedback_type = np.load(join(folder_path, 'alf', '_ibl_trials.feedbackType.npy'))
    feedback_times = np.load(join(folder_path, 'alf', '_ibl_trials.feedback_times.npy'))
    return stim_on_times, feedback_type, feedback_times


def transform_px_to_mm(dlc_data, width_mm=66, height_mm=54):
    """
    Transform pixel values to millimeter

    Parameters
    ----------
    width_mm:  the width of the video feed in mm
    height_mm: the height of the video feed in mm
    """

    # Set pixel dimensions for different cameras
    if dlc_data['camera'] == 'left':
        px_dim = [1280, 1024]
    elif dlc_data['camera'] == 'right' or dlc_data['camera'] == 'body':
        px_dim = [640, 512]

    # Transform pixels into mm
    for key in list(dlc_data.keys()):
        if key[-1] == 'x':
            dlc_data[key] = dlc_data[key] * (width_mm/px_dim[0])
        if key[-1] == 'y':
            dlc_data[key] = dlc_data[key] * (height_mm/px_dim[1])
    dlc_data['units'] = 'mm'

    return dlc_data





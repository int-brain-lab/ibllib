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

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:48:15 2019

@author: guido
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dlc_basis_functions import load_dlc_training, load_event_times, load_events, px_to_mm
from dlc_analysis_functions import pupil_features
from dlc_plotting_functions import peri_plot
from oneibl.one import ONE
from pathlib import Path
import cv2
from time import sleep


def get_video_frame(video_path, frame_number):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :param frame_number: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (1024, 1280, 3)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_number)  # 0-based index of the frame to be decoded/captured next.
    ret, frame_image = cap.read()
    cv2.imshow('Frame', frame_image)
    sleep(5)
    while(cap.isOpened()):
        ret, frame_image = cap.read()
        if ret is True:
            cv2.imshow('Frame', frame_image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    return frame_image


# Load in ONE
one = ONE()
dtypes = ['_ibl_leftCamera.dlc', '_iblrig_taskData.raw', 'trials.feedback_times',
          'trials.feedbackType', 'trials.stimOn_times', 'trials.choice', '_iblrig_leftCamera.raw']
eids = one.search(dataset_types=dtypes)
eid = eids[1]
d = one.load(eid, dataset_types=dtypes, download_only=True, dclass_output=True)
folder_path = str(Path(d.local_path[0]).parent.parent)

# Load in data
dlc_dict = load_dlc_training(folder_path)
dlc_dict = px_to_mm(dlc_dict)
stim_on_times, feedback_times = load_event_times(folder_path)
choice, feedback_type = load_events(folder_path)

# Get video frames during last reward
frame_number = np.argmin(np.abs(dlc_dict['timestamps']-feedback_times[feedback_type == 1][-1]))
frame = get_video_frame(d[7].local_path, frame_number)
"""
# Get pupil
pupil_x, pupil_y, diameter = pupil_features(dlc_dict)

# Plot pupil
sns.set(style="ticks", context="paper", font_scale=2, rc={"lines.linewidth": 2.5})
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
peri_plot(diameter, dlc_dict['timestamps'], stim_on_times, ax1, [-2, 4], 'baseline')
ax1.plot([0, 0], ax1.get_ylim(), 'r')
ax1.set(ylabel='Baseline subtracted pupil diameter (mm)', xlabel='Time (s)',
        title='Stimulus onset')
peri_plot(diameter, dlc_dict['timestamps'], feedback_times[feedback_type == 1],
          ax2, [-2, 4], 'baseline')
ax2.plot([0, 0], ax2.get_ylim(), 'r')
ax2.set(ylabel='Baseline subtracted pupil diameter (mm)', xlabel='Time (s)',
        title='Reward delivery')
plt.tight_layout(pad=2)
"""
# Plot paws
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
peri_plot(dlc_dict['middle_finger_r_x'], dlc_dict['timestamps'],
          feedback_times[(choice == -1) & (feedback_type == 1)], ax1, [-1, 10], 'baseline')
ax1.plot([0, 0], ax1.get_ylim(), 'r', label='Reward delivery')
ax1.set(ylabel='Baseline subtracted paw position (mm)', xlabel='Time (s)',
        title='Leftward choices')
ax1.legend(frameon=False)

peri_plot(dlc_dict['middle_finger_r_x'], dlc_dict['timestamps'],
          feedback_times[(choice == 1) & (feedback_type == 1)], ax2, [-1, 10], 'baseline')
ax2.plot([0, 0], ax2.get_ylim(), 'r', label='Reward delivery')
ax2.set(ylabel='Baseline subtracted paw position (mm)', xlabel='Time (s)',
        title='Rightward choices')
ax2.legend(frameon=False)
plt.tight_layout(pad=2)

# Plot tongue
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
peri_plot(dlc_dict['tongue_end_l_x'], dlc_dict['timestamps'],
          feedback_times[feedback_type == 1], ax1, [-1, 10], 'none')
ax1.plot([0, 0], ax1.get_ylim(), 'r', label='Reward delivery')
ax1.set(ylabel='Baseline subtracted paw position (mm)', xlabel='Time (s)',
        title='Correct trials')
ax1.legend(frameon=False)

peri_plot(dlc_dict['tongue_end_r_x'], dlc_dict['timestamps'],
          feedback_times[feedback_type == -1], ax2, [-1, 10], 'none')
ax2.plot([0, 0], ax2.get_ylim(), 'r', label='Reward delivery')
ax2.set(ylabel='Baseline subtracted paw position (mm)', xlabel='Time (s)',
        title='Incorrect trials')
ax2.legend(frameon=False)


# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:48:15 2019

@author: guido
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import zscore
from dlc_basis_functions import load_dlc, load_event_times
from dlc_pupil_functions import pupil_features

folder_path = 'C:\\Users\\guido\\Data\\Flatiron\\Subjects\\ZM_1736\\2019-08-09\\004'
#folder_path = 'C:\\Users\\guido\\Data\\Flatiron\\Subjects\\ZM_1735\\2019-08-01\\001'

# Load in data
dlc_data, timestamps = load_dlc(folder_path)
stim_on_times, feedback_type, feedback_times = load_event_times(folder_path)

# Get pupil
pupil_x, pupil_y, diameter = pupil_features(dlc_data)

# Plot data
#plot_window = [228, 238]
plot_window = [218, 230]

sns.set(style="ticks", context="talk", font_scale=1.2)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
ax1.plot(timestamps, zscore(pupil_x), label='x')
ax1.plot(timestamps, zscore(pupil_y), label='y')
for i in range(len(stim_on_times)):
    if i == 1:
        ax1.plot([stim_on_times[i], stim_on_times[i]], [-3, 3], 'k', label='stim onset')
    else:
        ax1.plot([stim_on_times[i], stim_on_times[i]], [-3, 3], 'k')
for i in range(len(feedback_times)):
    if i == 1:
        ax1.plot([feedback_times[i], feedback_times[i]], [-3, 3], 'g', label='reward delivery')
    else:
        ax1.plot([feedback_times[i], feedback_times[i]], [-3, 3], 'g')
ax1.set(ylabel='Z-scored position', xlabel='Time (s)', title='Pupil center',
        xlim=(plot_window[0], plot_window[-1]), ylim=(-3, 3))
ax1.legend()

ax2.plot(timestamps, zscore(diameter))
for i in range(len(stim_on_times)):
    if i == 1:
        ax2.plot([stim_on_times[i], stim_on_times[i]], [-3, 3], 'k', label='stim onset')
    else:
        ax2.plot([stim_on_times[i], stim_on_times[i]], [-3, 3], 'k')
for i in range(len(feedback_times)):
    if i == 1:
        ax2.plot([feedback_times[i], feedback_times[i]], [-3, 3], 'g', label='reward delivery')
    else:
        ax2.plot([feedback_times[i], feedback_times[i]], [-3, 3], 'g')
ax2.set(ylabel='Z-scored diameter', xlabel='Time (s)', title='Pupil diameter',
        xlim=(plot_window[0], plot_window[-1]), ylim=(-3, 3))
ax2.legend()





f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 5), sharex=True)
ax1.plot(timestamps, zscore(dlc_data['pupil_left_r_x']), label='x')
ax1.plot(timestamps, zscore(dlc_data['pupil_left_r_y']), label='y')
ax1.set(ylabel='Z-scored position', title='pupil left',
        xlim=(plot_window[0], plot_window[-1]))

ax2.plot(timestamps, zscore(dlc_data['pupil_top_r_x']), label='x')
ax2.plot(timestamps, zscore(dlc_data['pupil_top_r_y']), label='y')
ax2.set(ylabel='Z-scored position', title='pupil top',
        xlim=(plot_window[0], plot_window[-1]))

ax3.plot(timestamps, zscore(dlc_data['pupil_right_r_x']), label='x')
ax3.plot(timestamps, zscore(dlc_data['pupil_right_r_y']), label='y')
ax3.set(ylabel='Z-scored position', title='pupil right', xlabel='Time (s)',
        xlim=(plot_window[0], plot_window[-1]))

ax4.plot(timestamps, zscore(dlc_data['pupil_bottom_r_x']), label='x')
ax4.plot(timestamps, zscore(dlc_data['pupil_bottom_r_y']), label='y')
ax4.set(ylabel='Z-scored position', title='pupil bottom', xlabel='Time (s)',
        xlim=(plot_window[0], plot_window[-1]))

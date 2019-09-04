# -*- coding: utf-8 -*-
"""
Created on Tue Sep  3 15:48:15 2019

@author: guido
"""

import matplotlib.pyplot as plt
import seaborn as sns
from dlc_basis_functions import load_dlc, load_event_times, load_events, px_to_mm
from dlc_pupil_functions import pupil_features
from dlc_plotting_functions import peri_plot

#folder_path = 'C:\\Users\\guido\\Data\\Flatiron\\Subjects\\KS005\\2019-08-29\\001'
#folder_path = 'C:\\Users\\guido\\Data\\Flatiron\\Subjects\\KS005\\2019-08-30\\001'
#folder_path = 'C:\\Users\\guido\\Data\\Flatiron\\Subjects\\ibl_witten_04\\2019-08-04\\002'
#folder_path = 'C:\\Users\\guido\\Data\\Flatiron\\Subjects\\ibl_witten_04\\2018-08-11\\001'
folder_path = 'C:\\Users\\guido\\Data\\Flatiron\\Subjects\\ZM_1735\\2019-08-01\\001'
#folder_path = 'C:\\Users\\guido\\Data\\Flatiron\\Subjects\\ZM_1736\\2019-08-09\\004'

# Load in data
dlc_dict = load_dlc(folder_path)
dlc_dict = px_to_mm(dlc_dict)
stim_on_times, feedback_times = load_event_times(folder_path)
choice, feedback_type = load_events(folder_path)

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

# Plot paws
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
peri_plot(dlc_dict['middle_finger_r_x'], dlc_dict['timestamps'],
          feedback_times[(choice == -1) & (feedback_type == 1)], ax1, [-1, 1], 'baseline')
ax1.plot([0, 0], ax1.get_ylim(), 'r', label='Reward delivery')
ax1.set(ylabel='Baseline subtracted paw position (mm)', xlabel='Time (s)',
        title='Leftward choices')
ax1.legend(frameon=False)

peri_plot(dlc_dict['middle_finger_r_x'], dlc_dict['timestamps'],
          feedback_times[(choice == 1) & (feedback_type == 1)], ax2, [-1, 1], 'baseline')
ax2.plot([0, 0], ax2.get_ylim(), 'r', label='Reward delivery')
ax2.set(ylabel='Baseline subtracted paw position (mm)', xlabel='Time (s)',
        title='Rightward choices')
ax2.legend(frameon=False)
plt.tight_layout(pad=2)

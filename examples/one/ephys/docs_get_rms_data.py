"""
Download and plot RMS of raw data
=================================

Downloads rms data for a given session and probe and plots a heatmap of rms in AP and LFP band
on the channels along probe for duration of ephys recording.
"""


# import modules
from oneibl.one import ONE
import matplotlib.pyplot as plt
import alf.io
import numpy as np

# instantiate ONE
one = ONE()

# Specify subject, date and probe we are interested in
subject = 'CSHL049'
date = '2020-01-08'
sess_no = 1
probe_label = 'probe00'
eid = one.search(subject=subject, date=date, number=sess_no)[0]

# Specify the dataset types of interest
dtypes = ['_iblqc_ephysTimeRms.rms',
          '_iblqc_ephysTimeRms.timestamps',
          'channels.rawInd',
          'channels.localCoordinates']

# Download the data and get paths to downloaded data
_ = one.load(eid, dataset_types=dtypes, download_only=True)
ephys_path = one.path_from_eid(eid).joinpath('raw_ephys_data', probe_label)
alf_path = one.path_from_eid(eid).joinpath('alf', probe_label)

# Index of good recording channels along probe
chn_inds = np.load(alf_path.joinpath('channels.rawInd.npy'))
# Position of each recording channel along probe
chn_pos = np.load(alf_path.joinpath('channels.localCoordinates.npy'))
# Get range for y-axis
depth_range = [np.min(chn_pos[:, 1]), np.max(chn_pos[:, 1])]

# RMS data associated with AP band of data
rms_ap = alf.io.load_object(ephys_path, 'ephysTimeRmsAP', namespace='iblqc')
rms_ap_data = rms_ap['rms'][:, chn_inds] * 1e6  # convert to uV

# Median subtract to clean up the data
median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_ap_data))
# Add back the median so that the actual values in uV remain correct
rms_ap_data_median = np.apply_along_axis(lambda x: x - np.median(x), 1, rms_ap_data) + median

# Get levels for colour bar and x-axis
ap_levels = np.quantile(rms_ap_data_median, [0.1, 0.9])
ap_time_range = [rms_ap['timestamps'][0], rms_ap['timestamps'][-1]]

# RMS data associated with LFP band of data
rms_lf = alf.io.load_object(ephys_path, 'ephysTimeRmsLF', namespace='iblqc')
rms_lf_data = rms_lf['rms'][:, chn_inds] * 1e6  # convert to uV
# Median subtract to clean up the data
median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_lf_data))
rms_lf_data_median = np.apply_along_axis(lambda x: x - np.median(x), 1, rms_lf_data) + median

lf_levels = np.quantile(rms_lf_data_median, [0.1, 0.9])
lf_time_range = [rms_lf['timestamps'][0], rms_lf['timestamps'][-1]]

# Create figure
fig, ax = plt.subplots(2, 1, figsize=(6, 8))
# Plot the AP rms data
ax0 = ax[0]
rms_ap_plot = ax0.imshow(rms_ap_data_median.T, extent=np.r_[ap_time_range, depth_range],
                         cmap='plasma', vmin=ap_levels[0], vmax=ap_levels[1], origin='lower')
cbar_ap = fig.colorbar(rms_ap_plot, ax=ax0)
cbar_ap.set_label('AP RMS (uV)')
ax0.set_xlabel('Time (s)')
ax0.set_ylabel('Depth along probe (um)')
ax0.set_title('RMS of AP band')

# Plot the LFP rms data
ax1 = ax[1]
rms_lf_plot = ax1.imshow(rms_lf_data_median.T, extent=np.r_[lf_time_range, depth_range],
                         cmap='inferno', vmin=lf_levels[0], vmax=lf_levels[1], origin='lower')
cbar_lf = fig.colorbar(rms_lf_plot, ax=ax1)
cbar_lf.set_label('LFP RMS (uV)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Depth along probe (um)')
ax1.set_title('RMS of LFP band')

# Make sure it plots
plt.show()

"""
Functions dealing with passive task
"""
import numpy as np
from brainbox.processing import bincount2D
from scipy.linalg import svd


def get_on_off_times_and_positions(rf_map):

    rf_map_times = rf_map['times']
    rf_map_frames = rf_map['frames'].astype('float')

    gray = np.median(rf_map_frames)

    x_bin = rf_map_frames.shape[1]
    y_bin = rf_map_frames.shape[2]

    stim_on_frames = np.zeros((x_bin * y_bin, 1), dtype=np.ndarray)
    stim_off_frames = np.zeros((x_bin * y_bin, 1), dtype=np.ndarray)
    rf_map_pos = np.zeros((x_bin * y_bin, 2), dtype=np.int)

    i = 0
    for x_pos in np.arange(x_bin):
        for y_pos in np.arange(y_bin):

            pixel_val = rf_map_frames[:, x_pos, y_pos] - gray
            pixel_non_grey = np.where(pixel_val != 0)[0]
            # Find cases where the frame before was gray (i.e when the stim came on)
            frame_change = np.where(rf_map_frames[pixel_non_grey - 1, x_pos, y_pos] == gray)[0]

            stim_pos = pixel_non_grey[frame_change]

            # On stimulus, white squares
            on_pix = np.where(pixel_val[stim_pos] > 0)[0]
            stim_on = stim_pos[on_pix]
            stim_on_frames[i, 0] = stim_on

            off_pix = np.where(pixel_val[stim_pos] < 0)[0]
            stim_off = stim_pos[off_pix]
            stim_off_frames[i, 0] = stim_off

            rf_map_pos[i, :] = [x_pos, y_pos]
            i += 1

    rf_stim_frames = {}
    rf_stim_frames['on'] = stim_on_frames
    rf_stim_frames['off'] = stim_off_frames

    return rf_map_times, rf_map_pos, rf_stim_frames


def get_rf_map_over_depth(rf_map_times, rf_map_pos, rf_stim_frames, spike_times, spike_depths,
                          T_BIN=0.01, D_BIN=80, pre_stim=0.05, post_stim=1.5, y_lim=[0, 3840],
                          x_lim=None):

    binned_array, times, depths = bincount2D(spike_times, spike_depths, T_BIN, D_BIN,
                                             ylim=y_lim, xlim=x_lim)

    x_bin = len(np.unique(rf_map_pos[:, 0]))
    y_bin = len(np.unique(rf_map_pos[:, 1]))
    n_bins = int((pre_stim + post_stim) / T_BIN)

    rf_map = {}

    for stim_type, stims in rf_stim_frames.items():
        _rf_map = np.zeros(shape=(depths.shape[0], x_bin, y_bin, n_bins))
        for pos, stim_frame in zip(rf_map_pos, stims):

            x_pos = pos[0]
            y_pos = pos[1]

            stim_on_times = rf_map_times[stim_frame[0]]
            stim_intervals = np.c_[stim_on_times - pre_stim, stim_on_times + post_stim]

            idx_intervals = np.searchsorted(times, stim_intervals)

            stim_trials = np.zeros((depths.shape[0], n_bins, idx_intervals.shape[0]))
            for i, on in enumerate(idx_intervals):
                stim_trials[:, :, i] = binned_array[:, on[0]:on[1]]
            avg_stim_trials = np.mean(stim_trials, axis=2)

            _rf_map[:, x_pos, y_pos, :] = avg_stim_trials

        rf_map[stim_type] = _rf_map

    return rf_map, depths


def get_svd_map(rf_map):

    rf_svd = {}
    for stim_type, stims in rf_map.items():
        svd_stim = []
        for dep in stims:
            x_pix, y_pix, n_bins = dep.shape
            sub_reshaped = np.reshape(dep, (y_pix * x_pix, n_bins))
            bsl = np.mean(sub_reshaped[:, 0])

            u, s, v = svd(sub_reshaped - bsl)
            sign = -1 if np.median(v[0, :]) < 0 else 1
            rfs = sign * np.reshape(u[:, 0], (y_pix, x_pix))
            rfs *= s[0]

            svd_stim.append(rfs)

        rf_svd[stim_type] = svd_stim

    return rf_svd


def get_stim_aligned_activity(stim_events, spike_times, spike_depths, z_score_flag=True, D_BIN=20,
                              T_BIN=0.01, pre_stim=0.4, post_stim=1, base_stim=1,
                              y_lim=[0, 3840], x_lim=None):

    binned_array, times, depths = bincount2D(spike_times, spike_depths, T_BIN, D_BIN,
                                             ylim=y_lim, xlim=x_lim)
    n_bins = int((pre_stim + post_stim) / T_BIN)
    n_bins_base = int(np.ceil((base_stim - pre_stim) / T_BIN))

    stim_activity = {}
    for stim_type, stim_times in stim_events.items():

        stim_intervals = np.c_[stim_times - pre_stim, stim_times + post_stim]
        base_intervals = np.c_[stim_times - base_stim, stim_times - pre_stim]
        idx_stim = np.searchsorted(times, stim_intervals)
        idx_base = np.searchsorted(times, base_intervals)

        stim_trials = np.zeros((depths.shape[0], n_bins, idx_stim.shape[0]))
        noise_trials = np.zeros((depths.shape[0], n_bins_base, idx_stim.shape[0]))
        for i, (st, ba) in enumerate(zip(idx_stim, idx_base)):
            stim_trials[:, :, i] = binned_array[:, st[0]:st[1]]
            noise_trials[:, :, i] = binned_array[:, ba[0]:ba[1]]

        # Average across trials
        avg_stim_trials = np.mean(stim_trials, axis=2)
        if z_score_flag:
            # Average across trials and time
            avg_base_trials = np.mean(np.mean(noise_trials, axis=2), axis=1)[:, np.newaxis]
            std_base_trials = np.std(np.mean(noise_trials, axis=2), axis=1)[:, np.newaxis]
            z_score = (avg_stim_trials - avg_base_trials) / std_base_trials
            z_score[np.isnan(z_score)] = 0
            avg_stim_trials = z_score

        stim_activity[stim_type] = avg_stim_trials

    return stim_activity





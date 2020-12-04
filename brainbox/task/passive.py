"""
Functions dealing with passive task
"""
import numpy as np


def get_on_off_times_and_positions(rf_map):

    rf_map_times = rf_map['times']
    rf_map_frames = rf_map['frames'].astype('float')

    gray = np.median(rf_map_frames)
    sub_types = ['on', 'off']

    rf_pos = {sub: np.array([[]]) for sub in sub_types}

    x_bin = rf_map_frames.shape[1]
    y_bin = rf_map_frames.shape[2]

    rf_pos = np.full([rf_map_times.shape[0], 2], np.nan)
    for x_pos in np.arange(x_bin):
        for y_pos in np.arange(y_bin):

            pixel_val = rf_map_frames[:, x_pos, y_pos] - gray
            pixel_non_grey = np.where(pixel_val != 0)[0]
            # Find cases where the frame before was gray (i.e when the stim came on)
            frame_change = np.where(rf_map_frames[pixel_non_grey - 1, x_pos, y_pos] == gray)[0]

            stim_pos = pixel_non_grey[frame_change]

            rf_pos[stim_pos, 0] = x_pos
            rf_pos[stim_pos, 1] = y_pos

            # On stimulus, white squares
            on_pix = np.where(pixel_val[stim_pos] > 0)[0]
            stim_on = stim_pos[on_pix]
            #stim_on_all = np.c_[stim_on_all, stim_on]

            off_pix = np.where(pixel_val[stim_pos] < 0)[0]
            stim_off = stim_pos[off_pix]




    on_times = rf_map_times[stim_on_all]




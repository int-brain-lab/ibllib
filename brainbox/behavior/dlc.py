"""
Set of functions to deal with dlc data
"""
import logging
import pandas as pd
import warnings
import string

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.interpolate as interpolate

from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound
from ibllib.io.video import get_video_frame, url_from_eid
from ibllib.dsp.smooth import smooth_interpolate_savgol
import brainbox.behavior.wheel as bbox_wheel
from brainbox.processing import bincount2D

logger = logging.getLogger('ibllib')

SAMPLING = {'left': 60,
            'right': 150,
            'body': 30}
RESOLUTION = {'left': 2,
              'right': 1,
              'body': 1}

T_BIN = 0.02 # sec
WINDOW_LEN = 2 # sec
WINDOW_LAG = -0.5 # sec


# For plotting we use a window around the event the data is aligned to WINDOW_LAG before and WINDOW_LEN after the event
def plt_window(x):
    return x+WINDOW_LAG, x+WINDOW_LEN


def insert_idx(array, values):
    idx = np.searchsorted(array, values, side="left")
    # Choose lower index if insertion would be after last index or if lower index is closer
    idx[idx == len(array)] -= 1
    idx[np.where(abs(values - array[idx - 1]) < abs(values - array[idx]))] -= 1
    # If 0 index was reduced, revert
    idx[idx == -1] = 0
    return idx


def likelihood_threshold(dlc, threshold=0.9):
    """
    Set dlc points with likelihood less than threshold to nan
    :param dlc: dlc pqt object
    :param threshold: likelihood threshold
    :return:
    """
    features = np.unique(['_'.join(x.split('_')[:-1]) for x in dlc.keys()])
    for feat in features:
        nan_fill = dlc[f'{feat}_likelihood'] < threshold
        dlc[f'{feat}_x'][nan_fill] = np.nan
        dlc[f'{feat}_y'][nan_fill] = np.nan

    return dlc


def get_speed(dlc, dlc_t, camera, feature='paw_r'):
    """

    :param dlc: dlc pqt table
    :param dlc_t: dlc time points
    :param camera: camera type e.g 'left', 'right', 'body'
    :param feature: dlc feature to compute speed over
    :return:
    """
    x = dlc[f'{feature}_x'] / RESOLUTION[camera]
    y = dlc[f'{feature}_y'] / RESOLUTION[camera]

    # get speed in px/sec [half res]
    s = ((np.diff(x) ** 2 + np.diff(y) ** 2) ** .5) * SAMPLING[camera]

    dt = np.diff(dlc_t)
    tv = dlc_t[:-1] + dt / 2

    # interpolate over original time scale
    if tv.size > 1:
        ifcn = interpolate.interp1d(tv, s, fill_value="extrapolate")
        return ifcn(dlc_t)


def get_speed_for_features(dlc, dlc_t, camera, features=['paw_r', 'paw_l', 'nose_tip']):
    """
    Wrapper to compute speed for a number of dlc features and add them to dlc table
    :param dlc: dlc pqt table
    :param dlc_t: dlc time points
    :param camera: camera type e.g 'left', 'right', 'body'
    :param features: dlc features to compute speed for
    :return:
    """
    for feat in features:
        dlc[f'{feat}_speed'] = get_speed(dlc, dlc_t, camera, feat)

    return dlc


def get_feature_event_times(dlc, dlc_t, features):
    """
    Detect events from the dlc traces. Based on the standard deviation between frames
    :param dlc: dlc pqt table
    :param dlc_t: dlc times
    :param features: features to consider
    :return:
    """

    for i, feat in enumerate(features):
        f = dlc[feat]
        threshold = np.nanstd(np.diff(f)) / 4
        if i == 0:
            events = np.where(np.abs(np.diff(f)) > threshold)[0]
        else:
            events = np.r_[events, np.where(np.abs(np.diff(f)) > threshold)[0]]

    return dlc_t[np.unique(events)]


def get_licks(dlc, dlc_t):
    """
    Compute lick times from the toungue dlc points
    :param dlc: dlc pqt table
    :param dlc_t: dlc times
    :return:
    """
    lick_times = get_feature_event_times(dlc, dlc_t, ['tongue_end_l_x', 'tongue_end_l_y',
                                                      'tongue_end_r_x', 'tongue_end_r_y'])
    return lick_times


def get_sniffs(dlc, dlc_t):
    """
    Compute sniff times from the nose tip
    :param dlc: dlc pqt table
    :param dlc_t: dlc times
    :return:
    """

    sniff_times = get_feature_event_times(dlc, dlc_t, ['nose_tip_y'])
    return sniff_times


def get_dlc_everything(dlc_cam, camera):
    """
    Get out features of interest for dlc
    :param dlc_cam: dlc object
    :param camera: camera type e.g 'left', 'right'
    :return:
    """

    aligned = True
    if dlc_cam.times.shape[0] != dlc_cam.dlc.shape[0]:
        # logger warning and print out status of the qc, specific serializer django!
        logger.warning('Dimension mismatch between dlc points and timestamps')
        min_samps = min(dlc_cam.times.shape[0], dlc_cam.dlc.shape[0])
        dlc_cam.times = dlc_cam.times[:min_samps]
        dlc_cam.dlc = dlc_cam.dlc[:min_samps]
        aligned = False

    dlc_cam.dlc = likelihood_threshold(dlc_cam.dlc)
    dlc_cam.dlc = get_speed_for_features(dlc_cam.dlc, dlc_cam.times, camera)
    dlc_cam['licks'] = get_licks(dlc_cam.dlc, dlc_cam.times)
    dlc_cam['sniffs'] = get_sniffs(dlc_cam.dlc, dlc_cam.times)
    dlc_cam['aligned'] = aligned

    return dlc_cam


def get_pupil_diameter(dlc):
    """
    Estimates pupil diameter by taking median of different computations.

    The two most straightforward estimates: d1 = top - bottom, d2 = left - right
    In addition, assume the pupil is a circle and estimate diameter from other pairs of points

    :param dlc: dlc pqt table with pupil estimates, should be likelihood thresholded (e.g. at 0.9)
    :return: np.array, pupil diameter estimate for each time point, shape (n_frames,)
    """
    diameters = []
    # Get the x,y coordinates of the four pupil points
    top, bottom, left, right = [np.vstack((dlc[f'pupil_{point}_r_x'], dlc[f'pupil_{point}_r_y']))
                                for point in ['top', 'bottom', 'left', 'right']]
    # First compute direct diameters
    diameters.append(np.linalg.norm(top - bottom, axis=0))
    diameters.append(np.linalg.norm(left - right, axis=0))

    # For non-crossing edges, estimate diameter via circle assumption
    for pair in [(top, left), (top, right), (bottom, left), (bottom, right)]:
        diameters.append(np.linalg.norm(pair[0] - pair[1], axis=0) * 2 ** 0.5)

    # Ignore all nan runtime warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(diameters, axis=0)


def get_smooth_pupil_diameter(diameter_raw, camera, std_thresh=5, nan_thresh=1):
    """
    :param diameter_raw: np.array, raw pupil diameters, calculated from (thresholded) dlc traces
    :param camera: str ('left', 'right'), which camera to run the smoothing for
    :param std_thresh: threshold (in standard deviations) beyond which a point is labeled as an outlier
    :param nan_thresh: threshold (in seconds) above which we will not interpolate nans, but keep them
                       (for long stretches interpolation may not be appropriate)
    :return:
    """
    # set framerate of camera
    if camera == 'left':
        fr = SAMPLING['left']  # set by hardware
        window = 31  # works well empirically
    elif camera == 'right':
        fr = SAMPLING['right']  # set by hardware
        window = 75  # works well empirically
    else:
        raise NotImplementedError("camera has to be 'left' or 'right")

    # run savitzy-golay filter on non-nan time points to denoise
    diameter_smoothed = smooth_interpolate_savgol(diameter_raw, window=window, order=3, interp_kind='linear')

    # find outliers and set them to nan
    difference = diameter_raw - diameter_smoothed
    outlier_thresh = std_thresh * np.nanstd(difference)
    without_outliers = np.copy(diameter_raw)
    without_outliers[(difference < -outlier_thresh) | (difference > outlier_thresh)] = np.nan
    # run savitzy-golay filter again on (possibly reduced) non-nan timepoints to denoise
    diameter_smoothed = smooth_interpolate_savgol(without_outliers, window=window, order=3, interp_kind='linear')

    # don't interpolate long strings of nans
    t = np.diff(np.isnan(without_outliers).astype(int))
    begs = np.where(t == 1)[0]
    ends = np.where(t == -1)[0]
    if begs.shape[0] > ends.shape[0]:
        begs = begs[:ends.shape[0]]
    for b, e in zip(begs, ends):
        if (e - b) > (fr * nan_thresh):
            diameter_smoothed[(b + 1):(e + 1)] = np.nan  # offset by 1 due to earlier diff

    return diameter_smoothed


def get_trial_info(trials):
    """
    Extract relevant information from trials and clean up a bit

    :param trials: dict, ALF trials object
    :returns: dataframe with relevant, digested trials info
    """
    trials_df = pd.DataFrame({k: trials[k] for k in ['stimOn_times', 'feedback_times', 'choice', 'feedbackType']})
    # Translate choice and feedback type
    trials_df.loc[trials_df['choice'] == 1, 'choice'] = 'left'
    trials_df.loc[trials_df['choice'] == -1, 'choice'] = 'right'
    trials_df.loc[trials_df['feedbackType'] == -1, 'feedbackType'] = 'incorrect'
    trials_df.loc[trials_df['feedbackType'] == 1, 'feedbackType'] = 'correct'
    # Discard nan events
    trials_df = trials_df.dropna()
    # Discard too long trials
    idcs = trials_df[(trials_df['feedback_times'] - trials_df['stimOn_times']) > 10].index
    trials_df = trials_df.drop(idcs)
    return trials_df


def plot_trace_on_frame(frame, dlc_df, cam):
    """
    Plots dlc traces as scatter plots on a frame of the video.
    For left and right video also plots whisker pad and eye and tongue zoom.

    :param frame: frame to plot on
    :param dlc_df: thresholded dlc dataframe
    :param cam: string, which camera to process ('left', 'right', 'body')
    :returns: matplolib axis
    """
    # Define colors
    colors = {'tail_start': '#636EFA',
              'nose_tip': '#636EFA',
              'paw_l': '#EF553B',
              'paw_r': '#00CC96',
              'pupil_bottom_r': '#AB63FA',
              'pupil_left_r': '#FFA15A',
              'pupil_right_r': '#19D3F3',
              'pupil_top_r': '#FF6692',
              'tongue_end_l': '#B6E880',
              'tongue_end_r': '#FF97FF'}

    # Features without tube
    features = np.unique(['_'.join(x.split('_')[:-1]) for x in dlc_df.keys() if 'tube' not in x])
    # Normalize the number of points across cameras
    dlc_df_norm = pd.DataFrame()
    for feat in features:
        dlc_df_norm[f'{feat}_x'] = dlc_df[f'{feat}_x'][0::int(SAMPLING[cam] / 10)]
        dlc_df_norm[f'{feat}_y'] = dlc_df[f'{feat}_y'][0::int(SAMPLING[cam] / 10)]
        # Scatter
        plt.scatter(dlc_df_norm[f'{feat}_x'], dlc_df_norm[f'{feat}_y'], alpha=0.05, s=2, label=feat, c=colors[feat])

    plt.axis('off')
    plt.imshow(frame, cmap='gray')
    plt.tight_layout()

    ax = plt.gca()
    if cam == 'body':
        plt.title(f'{cam.capitalize()} camera')
        return ax
    # For left and right cam plot whisker pad rectangle
    # heuristic: square with side length half the distance between nose and pupil and anchored on midpoint
    p_nose = np.array(dlc_df[['nose_tip_x', 'nose_tip_y']].mean())
    p_pupil = np.array(dlc_df[['pupil_top_r_x', 'pupil_top_r_y']].mean())
    p_anchor = np.mean([p_nose, p_pupil], axis=0)
    dist = np.linalg.norm(p_nose - p_pupil)
    rect = matplotlib.patches.Rectangle((int(p_anchor[0] - dist/4), int(p_anchor[1])), int(dist/2), int(dist/3),
                                        linewidth=1, edgecolor='lime', facecolor='none')
    ax.add_patch(rect)
    # Plot eye region zoom
    inset_anchor = 0 if cam == 'right' else 0.5
    ax_ins = ax.inset_axes([inset_anchor, -0.5, 0.5, 0.5])
    ax_ins.imshow(frame, cmap='gray', origin="lower")
    for feat in features:
        ax_ins.scatter(dlc_df_norm[f'{feat}_x'], dlc_df_norm[f'{feat}_y'], alpha=1, s=0.001, label=feat, c=colors[feat])
    ax_ins.set_xlim(int(p_pupil[0] - 33 * RESOLUTION[cam] / 2), int(p_pupil[0] + 33 * RESOLUTION[cam] / 2))
    ax_ins.set_ylim(int(p_pupil[1] + 38 * RESOLUTION[cam] / 2), int(p_pupil[1] - 28 * RESOLUTION[cam] / 2))
    ax_ins.axis('off')
    # Plot tongue region zoom
    p1 = np.array(dlc_df[['tube_top_x', 'tube_top_y']].mean())
    p2 = np.array(dlc_df[['tube_bottom_x', 'tube_bottom_y']].mean())
    p_tongue = np.nanmean([p1, p2], axis=0)
    inset_anchor = 0 if cam == 'left' else 0.5
    ax_ins = ax.inset_axes([inset_anchor, -0.5, 0.5, 0.5])
    ax_ins.imshow(frame, cmap='gray', origin="upper")
    for feat in features:
        ax_ins.scatter(dlc_df_norm[f'{feat}_x'], dlc_df_norm[f'{feat}_y'], alpha=1, s=0.001, label=feat, c=colors[feat])
    ax_ins.set_xlim(int(p_tongue[0] - 60 * RESOLUTION[cam] / 2), int(p_tongue[0] + 100 * RESOLUTION[cam] / 2))
    ax_ins.set_ylim(int(p_tongue[1] + 60 * RESOLUTION[cam] / 2), int(p_tongue[1] - 100 * RESOLUTION[cam] / 2))
    ax_ins.axis('off')

    plt.title(f'{cam.capitalize()} camera')
    return ax


def plot_wheel_position(wheel_position, wheel_time, trials_df):
    """
    Plots wheel position across trials, color by which side was chosen

    :param wheel_position: Interpolated wheel position
    :param wheel_time: Interpolated wheel timestamps
    :param trials_df: Dataframe with trials info
    :return: matplotlib axis
    """
    # Create a window around the stimulus onset
    start_window, end_window = plt_window(trials_df['stimOn_times'])
    # Translating the time window into an index window
    start_idx = insert_idx(wheel_time, start_window)
    end_idx = np.array(start_idx + int(WINDOW_LEN / T_BIN), dtype='int64')
    # Getting the wheel position for each window, normalize to first value of each window
    trials_df['wheel_position'] = [wheel_position[start_idx[w] : end_idx[w]] - wheel_position[start_idx[w]]
                                   for w in range(len(start_idx))]
    # Plotting
    times = np.arange(len(trials_df['wheel_position'][0])) * T_BIN + WINDOW_LAG
    for side, color in zip(['left', 'right'], ['#1f77b4', 'darkred']):
        side_df = trials_df[trials_df['choice'] == side]
        for idx in side_df.index:
            plt.plot(times, side_df.loc[idx, 'wheel_position'], c=color, alpha=0.5, linewidth=0.05)
        plt.plot(times, side_df['wheel_position'].mean(), c=color, linewidth=2, label=side)

    plt.axhline(y=-0.26, linestyle='--', c='k', label='reward boundary')
    plt.axvline(x=0, linestyle='--', c='g', label='stimOn')
    plt.ylim([-0.27, 0.27])
    plt.xlabel('time [sec]')
    plt.ylabel('wheel position [rad]')
    plt.legend(loc='lower right')
    plt.title('Wheel position by choice')
    plt.tight_layout()

    return plt.gca()


def _bin_window_licks(lick_times, trials_df):
    """
    Helper function to bin and window the lick times and get them into trials df for plotting

    :param lick_times: licks.times loaded into numpy array
    :param trials_df: dataframe with info about trials
    :returns: dataframe with binned, windowed lick times for plotting
    """
    # Bin the licks
    lick_bins, bin_times, _ = bincount2D(lick_times, np.ones(len(lick_times)), T_BIN)
    lick_bins = np.squeeze(lick_bins)
    start_window, end_window = plt_window(trials_df['feedback_times'])
    # Translating the time window into an index window
    start_idx = insert_idx(bin_times, start_window)
    end_idx = np.array(start_idx + int(WINDOW_LEN / T_BIN), dtype='int64')
    # Get the binned licks for each window
    trials_df['lick_bins'] = [lick_bins[start_idx[l]:end_idx[l]] for l in range(len(start_idx))]
    # Remove windows that the exceed bins
    trials_df['end_idx'] = end_idx
    trials_df = trials_df[trials_df['end_idx'] <= len(lick_bins)]
    return trials_df


def plot_lick_psth(lick_times, trials_df):
    """
    Peristimulus histogram and licks raster of licks as estimated from dlc traces

    :param lick_times: licks.times loaded into numpy array
    :param trials_df: dataframe with info about trials
    :returns: matplotlib axis
    """
    licks_df = _bin_window_licks(lick_times, trials_df)
    # Plot
    times = np.arange(len(licks_df['lick_bins'][0])) * T_BIN + WINDOW_LAG
    plt.plot(times, licks_df[licks_df['feedbackType']=='correct']['lick_bins'].mean(), c='k', label='correct trial')
    plt.plot(times, licks_df[licks_df['feedbackType']=='incorrect']['lick_bins'].mean(), c='gray', label='incorrect trial')
    plt.axvline(x=0, label='feedback time', linestyle='--', c='r')
    plt.title('licks')
    plt.xlabel('time [sec]')
    plt.ylabel('lick events \n [a.u.]')
    plt.legend(loc='lower right')
    return plt.gca()


def plot_lick_raster(lick_times, trials_df):
    """
    Lick raster for correct licks

    :param lick_times: licks.times loaded into numpy array
    :param trials_df: dataframe with info about trials
    :returns: matplotlib axis
    """
    licks_df = _bin_window_licks(lick_times, trials_df)
    plt.imshow(list(licks_df[licks_df['feedbackType']=='correct']['lick_bins']), aspect='auto',
               extent=[-0.5, 1.5, len(licks_df['lick_bins'][0]), 0], cmap='gray_r')
    plt.xticks([-0.5, 0, 0.5, 1, 1.5])
    plt.ylabel('trials')
    plt.xlabel('time [sec]')
    plt.axvline(x=0, label='feedback time', linestyle='--', c='r')
    plt.title('lick events per correct trial')
    plt.tight_layout()
    return plt.gca()


# def plot_speed_psth(eid, feature='paw_r', align_to='goCue_times', cam='left', one=None):
#     """
#     Peristimulus histogramm of different dlc features
#     :param eid:
#     :param one:
#     """
#
#     one = one or ONE()
#     aligned_trials = align_trials_to_event(eid, align_to=align_to)
#     dlc = one.load_dataset(eid, f'_ibl_{cam}Camera.dlc.pqt')
#     dlc = likelihood_threshold(dlc)
#     dlc_t = one.load_dataset(eid, f'_ibl_{cam}Camera.times.npy')
#     speeds = get_speed(dlc, dlc_t, camera=cam, feature=feature)
#
#     speeds_sorted = {'correct': [], 'incorrect': []}
#     for trial in aligned_trials:
#         start_idx = _find_idx(dlc_t, aligned_trials[trial][0] + WINDOW_LAG)
#         end_idx = _find_idx(dlc_t, aligned_trials[trial][0] + WINDOW_LAG + WINDOW_LEN)
#         if aligned_trials[trial][3] == 1:
#             speeds_sorted['correct'].append(speeds[start_idx:end_idx])
#         elif aligned_trials[trial][3] == -1:
#             speeds_sorted['incorrect'].append(speeds[start_idx:end_idx])
#     # trim on e frame if necessary
#     for color, choice in zip(['k', 'gray'], ['correct', 'incorrect']):
#         m = min([len(x) for x in speeds_sorted[choice]])
#         q = [x[:m] for x in speeds_sorted[choice]]
#         xs = np.arange(m) / SAMPLING[cam]
#         xs = np.concatenate([-np.array(list(reversed(xs[:int(abs(WINDOW_LAG) * SAMPLING[cam])]))),
#                              np.array(xs[:int((WINDOW_LEN - abs(WINDOW_LAG)) * SAMPLING[cam])])])
#         m = min(len(xs), m)
#         qm = np.nanmean(q, axis=0)
#         plt.plot(xs[:m], qm[:m], c=color, linestyle='-', linewidth=1, label=choice)
#     plt.axvline(x=0, label=f'{align_to}', linestyle='--', c='g')
#     plt.title(f'{feature.split("_")[0].capitalize()} speed PSTH')
#     plt.xlabel('time [sec]')
#     plt.ylabel('speed [px/sec]')
#     plt.legend()  # bbox_to_anchor=(1.05, 1), loc='upper left')


def dlc_qc_plot(eid, one=None, cams=('left', 'right', 'body')):

    one = one or ONE()

    ''' Data loading '''
    dlc_traces = dict()
    video_frames = dict()
    for cam in cams:
        # Load and threshold the dlc traces
        dlc_traces[cam] = likelihood_threshold(one.load_dataset(eid, f'_ibl_{cam}Camera.dlc.pqt'))
        # Load a single frame for each video, first check if data is local, otherwise stream
        video_path = one.eid2path(eid).joinpath('raw_video_data', f'_iblrig_{cam}Camera.raw.mp4')
        if not video_path.exists():
            video_path = url_from_eid(eid, one=one)[cam]
        try:
            video_frames[cam] = get_video_frame(video_path, frame_number=5 * 60 * SAMPLING[cam])[:, :, 0]
        except TypeError:
            logger.warning(f"Could not load video frame for {cam} camera, some DLC QC plots have to be skipped.")
            video_frames[cam] = None
    # Load and extract trial info
    try:
        trials_df = get_trial_info(one.load_object(eid, 'trials'))
    except ALFObjectNotFound:
        logger.warning(f"Could not load trials object for session {eid}, some DLC QC plots have to be skipped.")
        trials_df = None
    # Load wheel data
    try:
        wheel_obj = one.load_object(eid, 'wheel')
        wheel_position, wheel_time = bbox_wheel.interpolate_position(wheel_obj.timestamps, wheel_obj.position,
                                                                     freq=1 / T_BIN)
    except ALFObjectNotFound:
        logger.warning(f"Could not load wheel object for session {eid}, some DLC QC plots have to be skipped.")
        wheel_position, wheel_time = None, None
    # Load lick data
    try:
        lick_times = one.load_dataset(eid, 'licks.times.npy')
    except ALFObjectNotFound:
        logger.warning(f"Could not load lick times for session {eid}, some DLC QC plots have to be skipped.")
        lick_times = None

    '''Create the list of panels'''
    panels = []
    for cam in cams:
        panels.append((plot_trace_on_frame, {'frame': video_frames[cam], 'dlc_df': dlc_traces[cam], 'cam': cam},
                       f'Traces on {cam} video'))
    panels.append((plot_wheel_position, {'wheel_position': wheel_position, 'wheel_time': wheel_time, 'trials_df': trials_df},
                   'Wheel position'))
    # Motion energy
    panels.append((plot_lick_psth, {'lick_times': lick_times, 'trials_df': trials_df}, 'Lick histogram'))
    panels.append((plot_lick_raster, {'lick_times': lick_times, 'trials_df': trials_df}, 'Lick raster'))

    ''' Plotting'''
    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(17, 10))
    for i, panel in enumerate(panels):
        plt.subplot(2, 5, i + 1)
        ax = plt.gca()
        ax.text(-0.1, 1.15, string.ascii_uppercase[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='right')
        # Check if any of the inputs is None
        if any([v is None for v in panel[1].values()]):
            ax.text(.5, .5, f"Data incomplete\nfor panel\n'{panel[2]}'", color='r', fontweight='bold', fontsize=12,
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            plt.axis('off')
        else:
            # Run the function to plot
            try:
                panel[0](**panel[1])
            except BaseException:
                ax.text(.5, .5, f'Error in \n{panel[0]}', color='r', fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.5), fontsize=10, transform=ax.transAxes)
    plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig

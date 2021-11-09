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
import ibllib.io.video as vidio
from ibllib.dsp.smooth import smooth_interpolate_savgol
import brainbox.behavior.wheel as bbox_wheel

logger = logging.getLogger('ibllib')

SAMPLING = {'left': 60,
            'right': 150,
            'body': 30}
RESOLUTION = {'left': 2,
              'right': 1,
              'body': 1}

T_BIN = 0.02
WINDOW_LEN = 2 # rt
WINDOW_LAG = -0.5 # st


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


def trial_event_times(eid, event_type='goCue_times', one=None):
    """
    For each trial get timing of event_type, contrast, side of contrast, choice and feedbackType

    :param eid: string, session ID
    :param event_type: string, type of event to extract timing for ('goCue_times', 'feedback_times', 'wheelMoves')
    :param one: ONE instance, if none is passed, defaults to instantiating ONE()
    """
    one = one or ONE()
    trials = one.load_object(eid, 'trials')
    if event_type == 'wheelMoves':
        wheel_moves = one.load_object(eid, event_type)['intervals'][:, 0]
    # dictionary, trial number and still interval
    d = dict()
    events = ['goCue_times', 'feedback_times', 'choice', 'feedbackType']
    for trial in range(len(trials['intervals'])):
        # discard nan events
        if any(np.isnan([trials[k][trial] for k in events])):
            continue
        # discard too long trials
        if (trials['feedback_times'][trial] - trials['goCue_times'][trial]) > 10:
            continue

        contrast, side = trials['contrastLeft'][trial], 1
        if np.isnan(contrast):
            contrast, side = trials['contrastRight'][trial], 0

        if event_type == 'wheelMoves':
            # make sure the motion onset time is in a coupled interval
            coupled = (wheel_moves > trials['goCue_times'][trial]) & (wheel_moves < trials['feedback_times'][trial])
            if not any(coupled):
                continue
            d[trial] = [wheel_moves[coupled][0], contrast, side, trials['choice'][trial], trials['feedbackType'][trial]]
        else:
            d[trial] = [trials[event_type][trial], contrast, side, trials['choice'][trial], trials['feedbackType'][trial]]
    return d


def plot_trace_on_frame(eid, cam, one=None):
    """
    Plots dlc traces as scatter plots on a frame of the video.
    For left and right video also plots whisker pad and eye and tongue zoom.

    :param eid: string, session ID
    :param cam: string, which camera to process ('left', 'right', 'body')
    :param one: ONE instance, if none is passed, defaults to instantiating ONE()
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

    one = one or ONE()
    # Load a single image to plot traces on
    url = vidio.url_from_eid(eid, one=one)[cam]
    frame_idx = [5 * 60 * SAMPLING[cam]]
    frame = np.squeeze(vidio.get_video_frames_preload(url, frame_idx, mask=np.s_[:, :, 0], quiet=True))
    # Load dlc trace
    dlc_df = one.load_dataset(eid, f'_ibl_{cam}Camera.dlc.pqt')
    # Threshold traces
    dlc_df = likelihood_threshold(dlc_df)
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


def _find_idx(array, value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or abs(value - array[idx-1]) < abs(value - array[idx])):
        return idx-1
    return idx


def plot_wheel_position(eid, one=None):
    """
    Plot wheel position across trials, color by choice
    :param eid: string, session ID
    """
    one = one or ONE()

    go_cues = trial_event_times(eid)
    wheel = one.load_object(eid, 'wheel')
    position, time = bbox_wheel.interpolate_position(wheel.timestamps, wheel.position, freq=1/T_BIN)

    plot_dict = {'wheel_left': [], 'wheel_right': []}
    for trial in go_cues.keys():
        start_idx = _find_idx(time, go_cues[trial][0] + WINDOW_LAG)
        wheel_pos = position[start_idx:(start_idx + int(WINDOW_LEN / T_BIN))]
        wheel_pos -= wheel_pos[0]

        if go_cues[trial][3] == -1:
            plot_dict['wheel_left'].append(wheel_pos)
        elif go_cues[trial][3] == 1:
            plot_dict['wheel_right'].append(wheel_pos)

    xs = np.arange(len(plot_dict['wheel_left'][0])) * T_BIN
    times = np.concatenate([-np.array(list(reversed(xs[:int(len(xs) * abs(WINDOW_LAG / WINDOW_LEN))]))),
                            np.array(xs[:int(len(xs) * (1 - abs(WINDOW_LAG / WINDOW_LEN)))])])

    for side, color in zip(['left', 'right'], ['#1f77b4', 'darkred']):
        for trajectory in plot_dict[f'wheel_{side}']:
            plt.plot(times, trajectory, c=color, alpha=0.5, linewidth=0.05)
        plt.plot(times, np.mean(plot_dict[f'wheel_{side}'], axis=0), c=color, linewidth=2, label=side)

    plt.axhline(y=-0.26, linestyle='--', c='k', label='reward boundary')
    plt.axvline(x=0, linestyle='--', c='g', label='stimOn')
    plt.ylim([-0.27, 0.27])
    plt.xlabel('time [sec]')
    plt.ylabel('wheel position [rad]')
    plt.legend(loc='lower right')
    plt.title('Wheel position by choice')
    plt.tight_layout()

    return plt.gca()


def plot_paw_speed(eid, one=None):

    one = one or ONE()
    # fs = {'right': 150, 'left': 60}
    line_specs = {'left': {'1': ['darkred', '--'], '-1': ['#1f77b4', '--']},
                  'right': {'1': ['darkred', '-'], '-1': ['#1f77b4', '-']}}

    go_cues = trial_event_times(eid)

    sc = {'left': {'1': [], '-1': []}, 'right': {'1': [], '-1': []}}
    for cam in ['right', 'left']:
        dlc = one.load_dataset(eid, f'_ibl_{cam}Camera.dlc.pqt')
        dlc = likelihood_threshold(dlc)
        dlc_t = one.load_dataset(eid, f'_ibl_{cam}Camera.times.npy')
        # take speed from right paw only, i.e. closer to cam
        speeds = get_speed(dlc, dlc_t, feature='paw_r')
        for trial in go_cues:
            start_idx = _find_idx(dlc_t, go_cues[trial][0] + WINDOW_LAG)
            end_idx = _find_idx(dlc_t, go_cues[trial][0] + WINDOW_LAG + WINDOW_LEN)
            sc[cam][str(go_cues[trial][3])].append(speeds[start_idx:end_idx])
        # trim on e frame if necessary
        for choice in ['-1', '1']:
            m = min([len(x) for x in sc[cam][choice]])
            q = [x[:m] for x in sc[cam][choice]]
            xs = np.arange(m) / SAMPLING[cam]
            xs = np.concatenate([
                -1 * np.array(list(reversed(xs[:int(abs(WINDOW_LAG) * SAMPLING[cam])]))),
                np.array(xs[:int((WINDOW_LEN - abs(WINDOW_LAG)) * SAMPLING[cam])])])
            m = min(len(xs), m)

            c = line_specs[cam][choice][0]
            ls = line_specs[cam][choice][1]

            qm = np.nanmean(q, axis=0)
            plt.plot(xs[:m], qm[:m], c=c, linestyle=ls,
                     linewidth=1,
                     label=f'paw {cam[0]},' + ' choice ' + choice)

    ax = plt.gca()
    ax.axvline(x=0, label='stimOn', linestyle='--', c='g')
    plt.title('paw speed PSTH')
    plt.xlabel('time [sec]')
    plt.ylabel('speed [px/sec]')
    plt.legend()  # bbox_to_anchor=(1.05, 1), loc='upper left')


def dlc_qc_plot(eid, one=None, dlc_df=None):

    one = one or ONE()
    panels = [(plot_trace_on_frame, {'eid': eid, 'cam': 'left', 'one': one}),
              (plot_trace_on_frame, {'eid': eid, 'cam': 'right', 'one': one}),
              (plot_trace_on_frame, {'eid': eid, 'cam': 'body', 'one': one}),
              (plot_wheel_position, {'eid': eid, 'one': one})]

    matplotlib.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(17, 10))

    for i, panel in enumerate(panels):
        plt.subplot(2, 5, i+1)
        ax = plt.gca()
        ax.text(-0.1, 1.15, string.ascii_uppercase[i], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='top', ha='right')
        # try:
        panel[0](**panel[1])
            # continue
        # except BaseException:
        #     plt.text(.5, .5, f'error in \n {panel}', color='r', fontweight='bold',
        #              bbox=dict(facecolor='white', alpha=0.5), fontsize=10, transform=ax.transAxes)
    plt.tight_layout()
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig

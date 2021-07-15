"""
Set of functions to deal with dlc data
"""
import numpy as np
import scipy.interpolate as interpolate
import logging
from one.api import ONE

logger = logging.getLogger('ibllib')

one = ONE()

SAMPLING = {'left': 60,
            'right': 150,
            'body': 30}
RESOLUTION = {'left': 2,
              'right': 1,
              'body': 1}


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

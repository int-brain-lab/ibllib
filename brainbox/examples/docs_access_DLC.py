"""
Accessing DeepLabCut (DLC) traces
=====================
This script illustrates how to access DLC results for a range of trials,
filter them by likelihood and save as a dictionary of numpy arrays,
with the keys being the tracked points and the entries being x,y
coordinates. This can be done for each camera
('left' only for training sessions, 'left',
'right' and 'body' for ephys sessions).

See also
https://github.com/int-brain-lab/iblapps/blob/develop/dlc/DLC_labeled_video.py
to make a labeled video, for checking specific trials.
"""

# Author: Michael

import numpy as np
import pandas as pd
import alf.io
from oneibl.one import ONE
from pathlib import Path


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def GetXYs(eid, video_type, trial_range):
    '''
    INPUT:
        eid: session id, e.g. '3663d82b-f197-4e8b-b299-7b803a155b84'
        video_type: one of 'left', 'right', 'body'
        trial_range: first and last trial number
            of range to be accessed, e.g. [5,7]
    OUTPUT:
        XYs: dictionary with DLC-tracked points as keys,
            x,y coordinates as entries, set to nan for low
            likelihood
        Times: corresponding timestamps
    '''

    one = ONE()
    dataset_types = ['camera.times',
                     'trials.intervals',
                     'camera.dlc']

    a = one.list(eid, 'dataset-types')

    assert all([i in a for i in dataset_types]
               ), 'For this eid, not all data available'

    D = one.load(eid, dataset_types=dataset_types, dclass_output=True)
    alf_path = Path(D.local_path[0]).parent.parent / 'alf'
    cam = alf.io.load_object(
        alf_path,
        '%sCamera' %
        video_type,
        namespace='ibl')

    # pick trial range
    trials = alf.io.load_object(alf_path, 'trials', namespace='ibl')
    num_trials = len(trials['intervals'])
    if trial_range[-1] > num_trials - 1:
        print('There are only %s trials' % num_trials)

    frame_start = find_nearest(cam['times'],
                               [trials['intervals'][trial_range[0]][0]])
    frame_stop = find_nearest(cam['times'],
                              [trials['intervals'][trial_range[-1]][1]])

    last_time_stamp = trials['intervals'][-1][-1]
    last_stamp_idx = find_nearest(cam['times'], last_time_stamp)

    print('Last trial ends at time %s, which is stamp index %s' %
          (last_time_stamp, last_stamp_idx))

    Times = cam['times'][frame_start:frame_stop]
    n_stamps = len(cam['times'])  #
    del cam['times']

    # some exceptions for inconsisitent data formats
    try:
        dlc_name = '_ibl_%sCamera.dlc.pqt' % video_type
        dlc_path = alf_path / dlc_name
        cam = pd.read_parquet(dlc_path, engine="fastparquet")
    except BaseException:
        raw_vid_path = alf_path.parent / 'raw_video_data'
        cam = alf.io.load_object(
            raw_vid_path,
            '%sCamera' %
            video_type,
            namespace='ibl')

    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])

    if video_type != 'body':
        d = list(points)
        d.remove('tube_top')
        d.remove('tube_bottom')
        points = np.array(d)

    # Set values to nan if likelyhood is too low # for pqt: .to_numpy()
    XYs = {}
    for point in points:
        print(point, len(cam[point + '_x']), n_stamps)
        assert len(cam[point + '_x']) <= n_stamps, 'n_stamps > dlc'
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array(
            [x[frame_start:frame_stop], y[frame_start:frame_stop]])

    return XYs, Times

"""
Accessing DeepLabCut (DLC) traces
=================================
This script illustrates how to access DLC results for a session
and video type, filter them by likelihood and save as a
dictionary of numpy arrays, with the keys being the tracked points
and the entries being x,y coordinates. This can be done for
each camera ('left' only for training sessions, 'left',
'right' and 'body' for ephys sessions).
See also

https://github.com/int-brain-lab/iblapps/blob/
develop/dlc/DLC_labeled_video.py to

make a labeled video, and

https://github.com/int-brain-lab/ibllib/blob/
camera_extractor/ibllib/qc/stream_dlc_labeled_frames.py

to stream some frames and paint dlc labels on top.
"""

# Author: Michael


import alf.io
import numpy as np
from oneibl.one import ONE


def get_DLC(eid, video_type):
    '''load dlc traces
    load dlc traces for a given session and
    video type.

    :param eid: A session eid
    :param video_type: string in 'left', 'right', body'
    :return: array of times and dict with dlc points
             as keys and x,y coordinates as values,
             for each frame id

    Example:

    eid = '6c6983ef-7383-4989-9183-32b1a300d17a'
    video_type = 'right'

    Times, XYs = get_DLC(eid, video_type)

    # get for frame 500 the x coordinate of the nose
    # and the time stamp:

    x_frame_500 = XYs['nose_tip'][0][500]
    t_frame_500 = Times[500]
    '''

    one = ONE()
    alf_path = one.path_from_eid(eid) / 'alf'
    cam0 = alf.io.load_object(
        alf_path,
        '%sCamera' %
        video_type,
        namespace='ibl')
    Times = cam0['times']
    cam = cam0['dlc']
    points = np.unique(['_'.join(x.split('_')[:-1]) for x in cam.keys()])
    XYs = {}
    for point in points:
        x = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_x'])
        x = x.filled(np.nan)
        y = np.ma.masked_where(
            cam[point + '_likelihood'] < 0.9, cam[point + '_y'])
        y = y.filled(np.nan)
        XYs[point] = np.array([x, y])

    return Times, XYs

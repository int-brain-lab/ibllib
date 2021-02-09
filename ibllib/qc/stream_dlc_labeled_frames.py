from oneibl.one import ONE
from ibllib.io.video import get_video_meta, get_video_frames_preload
import numpy as np
from pathlib import Path
import cv2
import alf.io
import matplotlib
from random import sample
import time


def stream_save_labeled_frames(eid, video_type):

    startTime = time.time()
    '''
    For a given eid and camera type, stream
    sample frames, print DLC labels on them
    and save
    '''

    # eid = '5522ac4b-0e41-4c53-836a-aaa17e82b9eb'
    # video_type = 'left'

    n_frames = 5  # sample 5 random frames

    save_images_folder = '/home/mic/DLC_QC/example_frames/'
    one = ONE()
    info = '_'.join(np.array(str(one.path_from_eid(eid)).split('/'))[[5,7,8]])
    print(info, video_type)

    r = one.list(eid, 'dataset_types')

    dtypes_DLC = ['_ibl_rightCamera.times.npy',
                  '_ibl_leftCamera.times.npy',
                  '_ibl_bodyCamera.times.npy',
                  '_iblrig_leftCamera.raw.mp4',
                  '_iblrig_rightCamera.raw.mp4',
                  '_iblrig_bodyCamera.raw.mp4',
                  '_ibl_leftCamera.dlc.pqt',
                  '_ibl_rightCamera.dlc.pqt',
                  '_ibl_bodyCamera.dlc.pqt']

    dtype_names = [x['name'] for x in r]

    assert all([i in dtype_names for i in dtypes_DLC]
               ), 'For this eid, not all data available'


    D = one.load(eid,
                 dataset_types=['camera.times', 'camera.dlc'],
                 dclass_output=True)
    alf_path = Path(D.local_path[0]).parent.parent / 'alf'

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
        XYs[point] = np.array(
            [x, y])


 
    if video_type != 'body':
        d = list(points)
        d.remove('tube_top')
        d.remove('tube_bottom')
        points = np.array(d)    
    
    # stream frames
    recs = [x for x in r if f'{video_type}Camera.raw.mp4' in x['name']][0]['file_records']
    video_path = [x['data_url'] for x in recs if x['data_url'] is not None][0]
    vid_meta = get_video_meta(video_path)

    frame_idx = sample(range(vid_meta['length']), n_frames)
    print('frame indices:', frame_idx) 
    frames = get_video_frames_preload(video_path,
                                      frame_idx,
                                      mask=np.s_[:, :, 0])
    size = [vid_meta['width'], vid_meta['height']]
    #return XYs, frames    
    
    x0 = 0
    x1 = size[0]
    y0 = 0
    y1 = size[1]
    if video_type == 'left':
        dot_s = 10  # [px] for painting DLC dots
    else:
        dot_s = 5

    # writing stuff on frames
    font = cv2.FONT_HERSHEY_SIMPLEX

    if video_type == 'left':
        bottomLeftCornerOfText = (20, 1000)
        fontScale = 4
    else:
        bottomLeftCornerOfText = (10, 500)
        fontScale = 2

    lineType = 2

    # assign a color to each DLC point (now: all points red)
    cmap = matplotlib.cm.get_cmap('Set1')
    CR = np.arange(len(points)) / len(points)

    block = np.ones((2 * dot_s, 2 * dot_s, 3))

    k = 0
    for frame in frames:

        gray = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)


        # print session info
        fontColor = (255, 255, 255)
        cv2.putText(gray,
                    info,
                    bottomLeftCornerOfText,
                    font,
                    fontScale / 4,
                    fontColor,
                    lineType)
                    
        # print time                    
        Time = round(Times[frame_idx[k]], 3)
        a, b = bottomLeftCornerOfText
        bottomLeftCornerOfText0 = (int(a * 10 + b / 2), b)



        a, b = bottomLeftCornerOfText
        bottomLeftCornerOfText0 = (int(a * 10 + b / 2), b)
        cv2.putText(gray,
                    '  time: ' + str(Time),
                    bottomLeftCornerOfText0,
                    font,
                    fontScale / 2,
                    fontColor,
                    lineType)



        # print DLC dots
        ll = 0
        for point in points:

            # Put point color legend
            fontColor = (np.array([cmap(CR[ll])]) * 255)[0][:3]
            a, b = bottomLeftCornerOfText
            if video_type == 'right':
                bottomLeftCornerOfText2 = (a, a * 2 * (1 + ll))
            else:
                bottomLeftCornerOfText2 = (b, a * 2 * (1 + ll))
            fontScale2 = fontScale / 4
            cv2.putText(gray, point,
                        bottomLeftCornerOfText2,
                        font,
                        fontScale2,
                        fontColor,
                        lineType)

            X0 = XYs[point][0][frame_idx[k]]
            Y0 = XYs[point][1][frame_idx[k]]

            X = Y0
            Y = X0

            #print(point,X,Y)
            if not np.isnan(X) and not np.isnan(Y):
                try:
                    col = (np.array([cmap(CR[ll])]) * 255)[0][:3]
                    # col = np.array([0, 0, 255]) # all points red
                    X = X.astype(int)
                    Y = Y.astype(int)

                    uu = block * col
                    gray[X - dot_s:X + dot_s, Y - dot_s:Y + dot_s] = uu

                except Exception as e:
                    print('frame', frame_idx[k])
                    print(e)
            ll += 1

        gray = gray[y0:y1, x0:x1]
        # cv2.imshow('frame', gray)
        cv2.imwrite(f'{save_images_folder}{eid}_frame_{frame_idx[k]}.png',
                    gray)
        cv2.waitKey(1)
        k += 1

    print(f'{n_frames} frames done in', np.round(time.time() - startTime))

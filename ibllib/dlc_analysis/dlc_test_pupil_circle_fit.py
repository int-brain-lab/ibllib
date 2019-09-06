# Test Guido's circle fit code by overlaying circle on top of image
from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import json

from dlc_overlay_features import get_video_frame, create_frame_array
from dlc_basis_functions import load_dlc
from dlc_analysis_functions import pupil_features

def create_pupil_array(dlc_data, frame_number):
    """
    Create x_vec, y_vec with paw coordinates from dlc dictionary.  x_vec and y_vec will each have
    four components and will be ordered pinky, ring, middle, pointer
    :param dlc_data: dictionary of DLC features
    :param frame_number: frame number
    :return: vec_x and vec_y
    """
    # Filter out non-position keys
    filtered_dict = {k: dlc_data[k] for k in dlc_data.keys() if k.endswith(('x', 'y'))}
    pupil_dict = {k: filtered_dict[k] for k in filtered_dict.keys() if k.startswith('pupil')}
    keys = set(map(lambda k: k[0:-2], pupil_dict.keys()))
    xy = [(filtered_dict[k + '_x'][frame_number], filtered_dict[k + '_y'][frame_number])
          for k in keys]
    return xy, filtered_dict.keys()

if __name__ == '__main__':
    # Load DLC data
    np.random.seed(0)
    main_path = Path('../../../code_camp_data/')
    SES = {
        'A': main_path.joinpath(Path('ZM_1735/2019-08-01/001')),
        'B': main_path.joinpath(Path('ibl_witten_04/2019-08-04/002')),
        'C': main_path.joinpath(Path('ZM_1736/2019-08-09/004')),
        'D': main_path.joinpath(Path('ibl_witten_04/2018-08-11/001')),
        'E': main_path.joinpath(Path('KS005/2019-08-29/001')),
        'F': main_path.joinpath(Path('KS005/2019-08-30/001')),
    }

    # select a session
    sid = 'E'
    ses_path = Path(SES[sid])

    # read in the alf objects
    left_camera = load_dlc(ses_path, camera='left')
    meta_path = ses_path / 'alf' / '_ibl_leftCamera.dlc.metadata.json'
    with open(meta_path) as json_data:
        dlc_meta = json.load(json_data)

    # Load DLC data
    dlc_dict = load_dlc(ses_path)

    # Load raw video data:
    # Path to raw video data
    raw_vid_meta = ses_path / 'raw_video_data' / '_iblrig_leftCamera.raw.mp4'

    # Try out Guido's circle fitting function:
    x_circ, y_circ, diameter_circ = pupil_features(dlc_dict)

    # Make directory for session data:
    dir_for_plotting = str(ses_path) + '/overlay_images/'
    if not os.path.exists(dir_for_plotting):
        os.makedirs(dir_for_plotting)

    #Fit the DLC pupil data:
    # Loop through frames and produce png:
    # Randomly sample frames:
    n_samps = 10
    frames_to_plot = np.random.choice(range(left_camera.timestamps.size), n_samps, replace=False)
    for frame in frames_to_plot:
        # Extract a single video frame and overlay location of top of pupil onto it
        video_frame = get_video_frame(str(raw_vid_meta), frame)

        # Overlay a circle corresponding to top of pupil onto image:
        pupil_xy = create_pupil_array(left_camera, frame)[0]

        # Extract Guido's circle coordinates and rad:
        this_x_circ = x_circ[frame]
        this_y_circ = y_circ[frame]
        this_circ_rad = diameter_circ[frame]/2

        # Show the image
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        plt.subplots_adjust(left=0.1, bottom=0.05, right=0.95, top=0.95, wspace=0.3, hspace=0.2)
        ax1.set_aspect('equal')
        ax1.imshow(video_frame, vmin=0, vmax=255)
        if sid == "D":
            ax1.set_xlim((450, 650))
            ax1.set_ylim((175, 275))
        else:
            ax1.set_xlim((250, 550))
            ax1.set_ylim((50, 250))
        ax1.set_title("Frame " + str(frame))

        ax2.set_aspect('equal')
        ax2.imshow(video_frame, vmin=0, vmax=255)
        for j in range(len(pupil_xy)):
            ax2.scatter(pupil_xy[j][0], pupil_xy[j][1], marker='+', s=100)
        circle1 = plt.Circle((this_x_circ, this_y_circ), this_circ_rad, color='r', fill=False)
        ax2.add_artist(circle1)
        if sid == "D":
            ax2.set_xlim((450, 650))
            ax2.set_ylim((175, 275))
        else:
            ax2.set_xlim((250, 550))
            ax2.set_ylim((50, 250))
        plt.title("Frame " + str(frame))
        #plt.show()
        fig.savefig(dir_for_plotting + 'frame_' + str(frame) + '_with_pupil_circle.png')


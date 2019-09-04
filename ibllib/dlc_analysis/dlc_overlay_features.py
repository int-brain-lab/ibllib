# Overlay DLC features on top of a few video frames as a sanity check. Written by ZCA, August 2019
# with code segments taken from script by Michael Schartner
# (https://github.com/int-brain-lab/iblvideo/blob/master/IBL_video_tile_plot.py)

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from dlc_basis_functions import load_dlc


def get_video_frame(video_path, frame_number):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :param frame_number: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (1024, 1280, 3)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_number)  # 0-based index of the frame to be decoded/captured next.
    ret, frame_image = cap.read()
    cap.release()
    return frame_image


def create_frame_array(dlc_data, frame_number):
    """
    Create x_vec, y_vec with paw coordinates from dlc dictionary.  x_vec and y_vec will each have
    four components and will be ordered pinky, ring, middle, pointer
    :param dlc_data: dictionary of DLC features
    :param frame_number: frame number
    :return: vec_x and vec_y
    """
    # Filter out non-position keys
    filtered_dict = {k: dlc_data[k] for k in dlc_data.keys() if k.endswith(('x', 'y'))}
    keys = set(map(lambda k: k[0:-2], filtered_dict.keys()))
    xy = [(filtered_dict[k + '_x'][frame_number], filtered_dict[k + '_y'][frame_number])
          for k in keys]
    return xy, filtered_dict.keys()


main_path = Path(r'C:\Users\User\Documents\Work\usb1\Subjects')
SES = {
    'A': main_path.joinpath(Path('ZM_1735/2019-08-01/001')),
    'B': main_path.joinpath(Path('ibl_witten_04/2019-08-04/002')),
    'C': main_path.joinpath(Path('ZM_1736/2019-08-09/004')),
    'D': main_path.joinpath(Path('ibl_witten_04/2018-08-11/001')),
    'E': main_path.joinpath(Path('KS005/2019-08-29/001')),
    'F': main_path.joinpath(Path('KS005/2019-08-30/001')),
}

# select a session from the bunch
sid = 'B'
ses_path = main_path / Path(SES[sid])

# read in the alf objects
left_camera = load_dlc(ses_path, camera='left')
meta_path = ses_path / 'alf' / '_ibl_leftCamera.dlc.metadata.json'
with open(meta_path) as json_data:
    dlc_meta = json.load(json_data)

raw_vid_meta = ses_path / 'raw_video_data' / '_iblrig_leftCamera.raw.mp4'
dlc_columns = dlc_meta['columns']

# Loop through frames and produce png:
# Randomly sample frames:
n_samps = 5
frames_to_plot = np.random.choice(range(left_camera.timestamps.size), n_samps, replace=False)
for frame in frames_to_plot:
    # Extract a single video frame and overlay location of top of pupil onto it
    video_frame = get_video_frame(str(raw_vid_meta), frame)

    # Overlay a circle corresponding to top of pupil onto image:
    xy = create_frame_array(left_camera, frame)[0]

    # Show the image
    fig, ax = plt.subplots(1)
    ax.set_aspect('equal')
    ax.imshow(video_frame)

    # Create circle at x, y pair extracted by Guido
    # circ = plt.Circle((top_pupil_x, top_pupil_y), 10, color="r", alpha=0.5)
    # ax.add_patch(circ)

    for j in range(len(xy)):
        ax.scatter(xy[j][0], xy[j][1], marker='+', s=100, alpha=0.9)

    plt.title("Overlay DLC on video frame - pupil_top_r \n Session: "
              + str(SES[sid]) + "; Frame " + str(frame))
    #  plt.show()
    # Save image:
    fig.savefig(SES[sid] / ('frame_' + str(frame) + '.png'))

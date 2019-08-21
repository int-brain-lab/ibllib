# Overlay DLC features on top of a few video frames as a sanity check. Written by ZCA, August 2019 with code segments
# taken from script by Michael Schartner (https://github.com/int-brain-lab/iblvideo/blob/master/IBL_video_tile_plot.py)

from numpy.core._multiarray_umath import ndarray
from oneibl.one import ONE
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def get_video_frame(video_path, frame_number):
    """
    Obtain numpy array corresponding to a particular video frame in video_path


    :param video_path: local path to mp4 file
    :param frame_number: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (1024, 1280, 3)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(1, frame_number)  # 0-based index of the frame to be decoded/captured next.
    ret, frame = cap.read()
    cap.release()
    return frame


def clean_timestamp_dta(timestamps):
    """
    Clean up timestamps data: when read in from ONE, the header for the pandas dataframe is a row of data. Append this to dataframe and convert dataframe to numpy array

    :param timestamps: camera timestamps obtained via ONE
    :return: numpy array of cleaned up camera timestamps data with one more row of data compared to input pandas df
    """
    timestamps_np: ndarray = np.array(timestamps.data)
    timestamps_new = np.vstack([np.array(timestamps.data.columns), timestamps_np])
    timestamps_new[:, 0] = timestamps_new[:, 0].astype('int')
    return timestamps_new


def create_pupil_array(dlc_frame_dict):
    """
    Create x_vec, y_vec with pupil coordinates from dlc dictionary.  x_vec and y_vec will each have four components and will be ordered top, right, left, bottom
    :param dlc_frame_dict: dictionary of DLC features
    :return: vec_x and vec_y
    """
    vec_x = [dlc_frame_dict['pupil_top_r_x'], dlc_frame_dict['pupil_right_r_x'], dlc_frame_dict['pupil_left_r_x'], dlc_frame_dict['pupil_bottom_r_x']]
    vec_y = [dlc_frame_dict['pupil_top_r_y'], dlc_frame_dict['pupil_right_r_y'], dlc_frame_dict['pupil_left_r_y'], dlc_frame_dict['pupil_bottom_r_y']]
    return vec_x, vec_y


if __name__ == '__main__':
    one = ONE()
    dtypes = ['_ibl_leftCamera.dlc', '_iblrig_leftCamera.raw', '_iblrig_leftCamera.timestamps']
    eids = one.search(dataset_types=dtypes)
    print(eids)
    eid = eids[0]
    # Make directory for session data:
    dir_id = eid
    if not os.path.exists(dir_id):
        os.makedirs(dir_id)
    # Load and download data to local computer
    dlc_meta, dlc_segments, timestamps, raw_vid_meta = one.load(eid, dataset_types=dtypes,
                                                                dclass_output=True)
    # Clean up data
    clean_timestamps = clean_timestamp_dta(timestamps)
    segments = dlc_segments.data
    dlc_columns = dlc_meta.data['columns']
    # Loop through frames and produce png:
    # Randomly sample 100 frames:
    frames_to_plot = np.random.choice(range(segments.shape[0]), 100, replace=False)
    for frame_number in frames_to_plot:
        # Extract a single video frame and overlay location of top of pupil onto it
        video_frame = get_video_frame(raw_vid_meta.local_path, frame_number)
        dlc_frame_dict = {k: v for k, v in zip(dlc_columns, segments[frame_number])}
        # Overlay a circle corresponding to top of pupil onto image:
        pupil_vec_x, pupil_vec_y = create_pupil_array(dlc_frame_dict)
        # Show the image
        fig, ax = plt.subplots(1)
        ax.set_aspect('equal')
        ax.imshow(video_frame)
        # Create circle at x, y pair extracted by Guido
        #circ = plt.Circle((top_pupil_x, top_pupil_y), 10, color="r", alpha=0.5)
        #ax.add_patch(circ)
        for j in range(len(pupil_vec_y)):
            ax.scatter(pupil_vec_x[j], pupil_vec_y[j], marker ='+', s=5, alpha = 0.9)
        plt.title("Overlay DLC on video frame - pupil_top_r \n Session: " + str(eid) + "; Frame " + str(frame_number))
        # Save image:
        fig.savefig(dir_id +'/frame_' + str(frame_number)+ '.png')

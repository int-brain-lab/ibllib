from pathlib import Path
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from dlc_basis_functions import load_dlc
from dlc_overlay_features import get_video_frame

def get_frame_at_specific_ms_time(video_path, time_in_ms):
    """
        Obtain numpy array corresponding to a frame at a particular time in video (where video is the one in video_path)
        :param video_path: local path to mp4 file
        :param time_in_ms: time in ms.  E.g. 20s is 20000
        :return: number of frames in video
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(0, time_in_ms)
    success, image = cap.read()
    frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES)
    cap.release()
    return image, frame_number


def get_video_length(video_path):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :return: number of frames in video
    """
    cap = cv2.VideoCapture(video_path)
    # Get number of frames in video
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return video_length

if __name__ == '__main__':
    np.random.seed()
    main_path = Path('../../../code_camp_data/')
    SES = {
        'A': main_path.joinpath(Path('ZM_1735/2019-08-01/001')),
        'B': main_path.joinpath(Path('ibl_witten_04/2019-08-04/002')),
        'C': main_path.joinpath(Path('ZM_1736/2019-08-09/004')),
        'D': main_path.joinpath(Path('ibl_witten_04/2018-08-11/001')),
        'E': main_path.joinpath(Path('KS005/2019-08-29/001')),
        'F': main_path.joinpath(Path('KS005/2019-08-30/001')),
    }

    # select a session from the bunch
    sid = 'A'
    ses_path = Path(SES[sid])

    # Load DLC data
    dlc_dict = load_dlc(ses_path)

    # Load raw video data:
    # Path to raw video data
    raw_video_path = ses_path / 'raw_video_data/_iblrig_leftCamera.raw.mp4'

    # Make directory for session data:
    dir_for_plotting = str(ses_path) + '/overlay_images/'
    if not os.path.exists(dir_for_plotting):
        os.makedirs(dir_for_plotting)

    # Get frame at a particular time in the video, as well as surrounding frames
    time = 51000
    video_frame, master_frame_number = get_frame_at_specific_ms_time(str(raw_video_path), time)
    earliest_frame = int(master_frame_number) - 18
    latest_frame = int(master_frame_number) + 15

    # DLC trace for these surrounding frames:
    vec_top_pupil_x = dlc_dict['pupil_top_r_x'][range(earliest_frame-20, latest_frame+20)]
    vec_top_pupil_y = dlc_dict['pupil_top_r_y'][range(earliest_frame-20, latest_frame+20)]
    vec_bottom_pupil_x = dlc_dict['pupil_bottom_r_x'][range(earliest_frame-20, latest_frame+20)]
    vec_bottom_pupil_y = dlc_dict['pupil_bottom_r_y'][range(earliest_frame-20, latest_frame+20)]
    vec_right_pupil_x = dlc_dict['pupil_right_r_x'][range(earliest_frame-20, latest_frame+20)]
    vec_right_pupil_y = dlc_dict['pupil_right_r_y'][range(earliest_frame-20, latest_frame+20)]
    vec_left_pupil_x = dlc_dict['pupil_left_r_x'][range(earliest_frame-20, latest_frame+20)]
    vec_left_pupil_y = dlc_dict['pupil_left_r_y'][range(earliest_frame-20, latest_frame+20)]
    vec_top_likelihood = dlc_dict['pupil_top_r_likelihood'][range(earliest_frame-20, latest_frame+20)]
    vec_bottom_likelihood = dlc_dict['pupil_bottom_r_likelihood'][range(earliest_frame - 20, latest_frame + 20)]
    vec_right_likelihood = dlc_dict['pupil_right_r_likelihood'][range(earliest_frame - 20, latest_frame + 20)]
    vec_left_likelihood = dlc_dict['pupil_left_r_likelihood'][range(earliest_frame - 20, latest_frame + 20)]

    # Plot surrounding frames
    for frame_no in range(earliest_frame, latest_frame):
    # Save the image with DLC trace overlaid
        frame_to_plot = get_video_frame(str(raw_video_path), frame_no)
        fig = plt.figure(figsize=(10, 18), dpi=80, facecolor='w', edgecolor='k')
        plt.subplots_adjust(left=0.1, bottom=0.07, right=0.95, top=0.93, wspace=0.3, hspace=0.2)

        plt.subplot(4, 1, 1)
        plt.imshow(frame_to_plot)
        plt.title("Blink Detector \n Frame " + str(int(frame_no)), fontsize = 28)
        plt.scatter(dlc_dict['pupil_top_r_x'][frame_no], dlc_dict['pupil_top_r_y'][frame_no], marker='+', s=100, alpha=0.9)
        plt.scatter(dlc_dict['pupil_bottom_r_x'][frame_no], dlc_dict['pupil_bottom_r_y'][frame_no], marker='+', s=100, alpha=0.9)
        plt.scatter(dlc_dict['pupil_right_r_x'][frame_no], dlc_dict['pupil_right_r_y'][frame_no], marker='+', s=100,
                   alpha=0.9)
        plt.scatter(dlc_dict['pupil_left_r_x'][frame_no], dlc_dict['pupil_left_r_y'][frame_no], marker='+', s=100,
                   alpha=0.9)
        plt.xlim((150, 550))
        plt.ylim((0, 300))

        plt.subplot(4, 1, 2)
        plt.plot(range(earliest_frame-20, latest_frame+20), vec_top_pupil_x, label='top pupil')
        plt.plot(range(earliest_frame-20, latest_frame+20), vec_bottom_pupil_x, label='bottom pupil')
        plt.plot(range(earliest_frame-20, latest_frame+20), vec_right_pupil_x, label='right pupil')
        plt.plot(range(earliest_frame-20, latest_frame+20), vec_left_pupil_x, label='left pupil')
        plt.axvline(x=frame_no, color='r', alpha=0.5)
        plt.ylabel("x pixel loc", fontsize = 20)
        plt.xlabel("Frame number", fontsize = 20)
        plt.axvspan(3053, 3062, alpha=0.5, color='grey', label='blink')
        plt.xticks(range(earliest_frame-20, latest_frame+20, 3), range(len(range(earliest_frame-20, latest_frame+20)), 3))
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(range(earliest_frame-20, latest_frame+20), vec_top_pupil_y, label='top pupil')
        plt.plot(range(earliest_frame-20, latest_frame+20), vec_bottom_pupil_y, label='bottom pupil')
        plt.plot(range(earliest_frame-20, latest_frame+20), vec_right_pupil_y, label='right pupil')
        plt.plot(range(earliest_frame-20, latest_frame+20), vec_left_pupil_y, label='left pupil')
        plt.axvline(x=frame_no, color='r', alpha=0.5)
        plt.axvspan(3053, 3062, alpha=0.5, color='grey', label = 'blink')
        plt.ylabel("y pixel loc", fontsize = 20)
        plt.xlabel("Frame number ", fontsize = 20)
        plt.xticks(range(earliest_frame-20, latest_frame+20, 3), range(len(range(earliest_frame-20, latest_frame+20)), 3))
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(range(earliest_frame - 20, latest_frame + 20), vec_top_likelihood, label='top pupil')
        plt.plot(range(earliest_frame - 20, latest_frame + 20), vec_bottom_likelihood, label='bottom pupil')
        plt.plot(range(earliest_frame - 20, latest_frame + 20), vec_right_likelihood, label='right pupil')
        plt.plot(range(earliest_frame - 20, latest_frame + 20), vec_left_likelihood, label='left pupil')
        plt.axvline(x=frame_no, color='r', alpha=0.5)
        plt.axvspan(3053, 3062, alpha=0.5, color='grey', label='blink')
        plt.ylabel("DLC likelihood", fontsize=20)
        plt.xlabel("Frame number ", fontsize=20)
        plt.xticks(range(earliest_frame - 20, latest_frame + 20, 3),
                   range(len(range(earliest_frame - 20, latest_frame + 20)), 3))
        plt.legend()

    # Save image:
        fig.savefig(dir_for_plotting + '/frame_' + str(frame_no) +'.png')# + str(frame_number) + '.png')

    #for frame_number in frames_to_plot:
        # Extract a single video frame and overlay location of top of pupil onto it
        #video_frame = get_video_frame(str(raw_video_path), frame_number)


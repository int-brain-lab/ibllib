"""Camera QC
This module runs a list of quality control metrics on the camera and extracted video data.

TODO Add checks from check_cam scratch
Question:
What is the order of things? I am assuming the frames, pin state, FPGA times all occur outside of the task
What are the units of the ssv timestamps?  Can we use these as a measure of Bpod timestamp accuracy (these are interpolated)
What to do if the pin_state file exists but not the count file?
We're not extracting the audio based on TTL length.  Is this a problem?
"""
import sys
import logging
from inspect import getmembers, isfunction

import cv2
import numpy as np
import matplotlib.pyplot as plt

from ibllib.io.extractors.camera import (
    load_embedded_frame_data, extract_camera_sync, PIN_STATE_THRESHOLD, extract_all
)
from ibllib.exceptions import ALFObjectNotFound
from ibllib.io.extractors import ephys_fpga
from ibllib.io import raw_data_loaders as raw
import alf.io as alfio
from brainbox.core import Bunch
from . import base

_log = logging.getLogger('ibllib')


def get_video_frame(video_path, frame_number):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :param frame_number: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (1024, 1280, 3)
    """
    cap = cv2.VideoCapture(str(video_path))
    #  fps = cap.get(cv2.CAP_PROP_FPS)
    #  print("Frame rate = " + str(fps))
    cap.set(1, frame_number)  # 0-based index of the frame to be decoded/captured next.
    ret, frame_image = cap.read()
    cap.release()
    return frame_image


def get_video_frames_preload(video_path, frame_numbers):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :param frame_numbers: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (1024, 1280,
    3).  Also returns the frame rate and total number of frames
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if len(frame_numbers) == 0:
        return None, fps, total_frames
    elif 0 < frame_numbers[-1] >= total_frames:
        raise IndexError('frame numbers must be between 0 and ' + str(total_frames))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_numbers[0])
    frame_images = []
    for i in frame_numbers:
        sys.stdout.write(f'\rloading frame {i}/{frame_numbers[-1]}')
        sys.stdout.flush()
        ret, frame = cap.read()
        frame_images.append(frame)
    cap.release()
    sys.stdout.write('\x1b[2K\r')  # Erase current line in stdout
    return np.array(frame_images), fps, total_frames


class CameraQC(base.QC):
    """A class for computing camera QC metrics"""
    dstypes = [
        '_iblrig_Camera.frame_counter',
        '_iblrig_Camera.GPIO',
        '_iblrig_Camera.timestamps',
        '_iblrig_taskData.raw',
        '_iblrig_taskSettings.raw',
        '_iblrig_Camera.raw'
    ]
    dstypes_fpga = [
        '_spikeglx_sync.channels',
        '_spikeglx_sync.polarities',
        '_spikeglx_sync.times',
        'ephysData.raw.meta',
        'ephysData.raw.wiring'
    ]
    """Recall that for the training rig there is only one side camera at 30 Hz and 1280 x 1024 px. 
    For the recording rig there are two side cameras (left: 60 Hz, 1280 x 1024 px; 
    right: 150 Hz, 640 x 512 px) and one body camera (30 Hz, 640 x 512 px). """
    video_meta = {
        'training': {
            'left': {
                'fps': 30,
                'width': 1280,
                'height': 1024
            }
        },
        'ephys': {
            'left': {
                'fps': 60,
                'width': 1280,
                'height': 1024
            },
            'right': {
                'fps': 150,
                'width': 640,
                'height': 512
            },
            'body': {
                'fps': 30,
                'width': 640,
                'height': 512
            },
        }
    }

    def __init__(self, session_path_or_eid, side, **kwargs):
        """
        :param session_path_or_eid: A session eid or path
        :param log: A logging.Logger instance, if None the 'ibllib' logger is used
        :param one: An ONE instance for fetching and setting the QC on Alyx
        :param cameras: The comeras to run QC on, if None QC is run for all three cameras
        """
        # When an eid is provided, we will download the required data by default (if necessary)
        self.download_data = not alfio.is_session_path(session_path_or_eid)
        super().__init__(session_path_or_eid, **kwargs)

        # Data
        self.side = side
        filename = f'_iblrig_{self.side}Camera.raw.mp4'
        self.video_path = self.session_path / 'raw_video_data' / filename
        self.type = raw.get_session_extractor_type(self.session_path)
        self.data = Bunch()

        # QC outcomes map
        self.metrics = None
        self.outcome = 'NOT_SET'

    def load_data(self, download_data=True):
        """Extract the data from raw data files
        Extracts all the required task data from the raw data files.

        :param download_data: if True, any missing raw data is downloaded via ONE.
        """
        if download_data:
            self._ensure_required_data()

        # Get frame count and pin state
        self.data['count'], self.data['pin_state'] = \
            load_embedded_frame_data(self.session_path, self.side, raw=True)

        # Get audio data
        if self.type == 'ephys':
            sync, chmap = ephys_fpga.get_main_probe_sync(self.session_path)
            audio_ttls = ephys_fpga._get_sync_fronts(sync, chmap['audio'])
            self.data['audio'] = audio_ttls['times'][::2]  # Get rises
            # Load raw FPGA times
            cam_ts = extract_camera_sync(sync, chmap)
            self.data['fpga_times'] = cam_ts[f'{self.side}_camera']
        else:
            _, audio_ttls = raw.load_bpod_fronts(self.session_path)
            self.data['audio'] = audio_ttls['times'][::2]

        # Load extracted frame times
        alf_path = self.session_path / 'alf'
        try:
            self.data['frame_times'] = alfio.load_object(alf_path, f'{self.side}Camera')['times']
        except ALFObjectNotFound:  # Re-extract
            # TODO Flag for extracting single camera's data
            outputs, _ = extract_all(self.session_path, self.type, save=False)
            self.data['frame_times'] = outputs[f'{self.side}_camera_timestamps']

        # Gather information from video file
        _log.info('inspecting video file...')
        cap = cv2.VideoCapture(str(self.video_path))

        # Get basic properties of video
        info = Bunch()
        info.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info.fps = int(cap.get(cv2.CAP_PROP_FPS))
        info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.data['video'] = info
        self.data['brightness'] = vid_to_brightness(self.video_path)

        # Load Bonsai frame timestamps
        file = self.session_path / 'raw_video_data' / f'_iblrig_{self.side}Camera.timestamps.ssv'
        # df = pd.read_csv(file, delim_whitespace=True)
        ssv_params = dict(names=('date', 'timestamp'), dtype='<M8[ns],<u4', delimiter=' ')
        self.data['bpod_frame_times'] = np.genfromtxt(file, **ssv_params)  # np.loadtxt is slower

    def _ensure_required_data(self):
        files = self.one.load(self.eid,
                              dataset_types=self.dstypes + self.dstypes_fpga, download_only=True)
        assert not any(file is None for file in files)

    def run(self, update: bool = False, download_data: bool = False) -> (str, dict):
        """
        Run video QC checks and return outcome
        :param update: if True, updates the session QC fields on Alyx
        :param download_data: if True, downloads any missing data if required
        :returns:
        """
        _log.info('Computing QC outcome')
        if not self.data:
            self.load_data(download_data)

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(CameraQC, is_metric)
        namespace = f'video{self.side.capitalize()}'
        self.metrics = {f'_{namespace}_' + k[6:]: fn(self) for k, fn in checks}
        all_pass = all(x is None or x == 'PASS' for x in self.metrics.values())
        self.outcome = 'PASS' if all_pass else 'FAIL'
        if update:
            bool_map = {k: None if v is None else v == 'PASS' for k, v in self.metrics.items()}
            self.update_extended_qc(bool_map)
            self.update(self.outcome, namespace)
        return self.outcome, self.metrics

    def check_contrast(self):
        """Check that the video contrast range"""
        MIN_MAX = [20, 80]  # TODO Normalize
        MIN_STD = 20
        brightness = self.data['brightness']

        within_range = np.logical_and(brightness > MIN_MAX[0],
                                      brightness < MIN_MAX[1])
        passed = within_range.all() and np.std(brightness) < MIN_STD
        return 'PASS' if passed else 'FAIL'

    def check_file_headers(self):
        """Check reported frame rate matches FPGA frame rate"""
        expected = self.video_meta[self.type][self.side]
        return 'PASS' if self.data['video']['fps'] == expected['fps'] else 'FAIL'

    def check_framerate(self):
        """Check camera times match specified frame rate for camera"""
        THRESH = 1.  # NB: Does not take into account dropped frames
        fps = self.video_meta[self.type][self.side]['fps']
        Fs = 1 / np.median(np.diff(self.data['frame_times']))  # Approx. frequency of camera
        return 'PASS' if Fs - fps < THRESH else 'FAIL'

    def check_pin_state(self, display=False):
        """Check the pin state reflects Bpod TTLs"""
        if self.data['pin_state'] is None:
            return 'NOT_SET'
        size_matches = self.data['video']['length'] == self.data['pin_state'].size
        # There should be only one value below our threshold
        correct_threshold = sum(np.unique(self.data['pin_state']) < PIN_STATE_THRESHOLD) == 1
        state = self.data['pin_state'] > PIN_STATE_THRESHOLD
        # NB: The pin state to be high for 2 consecutive frames
        low2high = np.insert(np.diff(state.astype(int)) == 1, 0, False)
        state_ttl_matches = sum(low2high) == self.data['audio'].size
        # Check within ms of audio times
        if display:
            plt.Figure()
            plt.plot(self.data['frame_times'][low2high], np.zeros(sum(low2high)), 'o',
                     label='GPIO Low -> High')
            plt.plot(self.data['audio'], np.zeros(self.data['audio'].size), 'rx',
                     label='Audio TTL High')
            plt.xlabel('FPGA frame times / s')
            plt.gca().set(yticklabels=[])
            plt.gca().tick_params(left=False)
        # idx = [i for i, x in enumerate(self.data['audio'])
        #        if np.abs(x - self.data['frame_times'][low2high]).min() > 0.01]
        # mins = [np.abs(x - self.data['frame_times'][low2high]).min() for x in self.data['audio']]

        return 'PASS' if size_matches and correct_threshold and state_ttl_matches else 'FAIL'

    def check_dropped_frames(self):
        """Check how many frames were reported missing"""
        THRESH = .1  # Percent
        size_matches = self.data['video']['length'] == self.data['count'].size
        assert np.all(np.diff(self.data['count']) > 0), 'frame count not strictly increasing'
        dropped = np.diff(self.data['count']).astype(int) - 1
        return 'PASS' if size_matches and (sum(dropped) / len(dropped) * 100) < THRESH else 'FAIL'

    def check_timestamps(self):
        """Check that the camera.times array is reasonable"""
        # Check frame rate matches what we expect
        expected = 1 / self.video_meta[self.type][self.side]['fps']
        # TODO Remove dropped frames from test
        frame_delta = np.diff(self.data['frame_times'])
        fps_matches = np.isclose(frame_delta.mean(), expected, atol=0.001)
        # Check number of timestamps matches video
        length_matches = self.data['frame_times'].size == self.data['video'].length
        # Check times are strictly increasing
        increasing = all(np.diff(self.data['frame_times']) > 0)
        return 'PASS' if increasing and fps_matches and length_matches else 'FAIL'

    def check_resolution(self):
        """Check that the timestamps and video file resolution match what we expect"""
        actual = self.data['video']
        expected = self.video_meta[self.type][self.side]
        match = actual['width'] == expected['width'] and actual['height'] == expected['height']
        return 'PASS' if match else 'FAIL'

    def check_wheel_alignment(self):
        """Check wheel motion in video correlates with the rotary encoder signal"""
        pass

    def check_position(self):
        """Check camera is positioned correctly"""
        pass

    def check_focus(self, display=False):
        """Check video is in focus"""
        # Could blur a frame and check difference
        n = 50
        img = get_video_frame(self.video_path, n)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Smoothing without removing edges.
        filtered = cv2.bilateralFilter(img, 7, 50, 50)

        # Applying the canny filter
        edges = cv2.Canny(filtered, 60, 120)

        if display:
            # Display the resulting frame
            cv2.imshow(f'Frame #{n}', edges)
        return  # 'NOT_SET'


def vid_to_brightness(camera_path, n_frames=5):
    """
    :param camera_path: The full path for the camera
    :param n_frames: Number of frames to sample for brightness
    :return:
    """
    # TODO Use video loaders from iblapps
    cap = cv2.VideoCapture(str(camera_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness = []
    # for n sampled frames, save brightness in array
    idx = np.linspace(100, total_frames - 100, n_frames).astype(int)
    for ii, i in enumerate(idx):
        sys.stdout.write(f'\rloading frame {ii}/{n_frames}')
        sys.stdout.flush()
        cap.set(1, i)
        ret, frame = cap.read()
        if ret:
            brightness.append(np.mean(frame))
        else:
            _log.warning(f'failed to read frame {ii}')
    cap.release()
    sys.stdout.write('\x1b[2K\r')  # Erase current line in stdout
    return np.array(brightness)

# def plot_brightness(D):
#     plt.ion()
#     plt.figure()
#     ax0 = plt.subplot(1, 3, 1)
#     for eid in D:
#         for vid in D[eid]:
#             if 'left' in vid:
#                 plt.plot(D[eid][vid], label=eid)
#     ax0.set_ylabel('brightness (mean pixel)')
#     ax0.set_xlabel('uniformly sampled frames')
#     ax0.set_title('left')
#     ax1 = plt.subplot(1, 3, 2, sharex=ax0, sharey=ax0)
#     for eid in D:
#         for vid in D[eid]:
#             if 'right' in vid:
#                 plt.plot(D[eid][vid], label=eid)
#     ax1.set_title('right')
#     ax2 = plt.subplot(1, 3, 3, sharex=ax0, sharey=ax0)
#     for eid in D:
#         for vid in D[eid]:
#             if 'body' in vid:
#                 plt.plot(D[eid][vid], label='_'.join(vid.split('_')[:-1]))
#     ax2.set_title('body')
#     ax2.legend().set_draggable(True)


def run_all_qc(session, update=False):
    """Run QC for all cameras
    Run the camera QC for left, right and body cameras.
    :param session: A session path or eid.
    :param update: If True, QC fields are updated on Alyx.
    :return:
    """
    qc = {}
    for camera in ['left', 'right', 'body']:
        qc[camera] = CameraQC(session, side=camera)
        qc[camera].run(update=update)
    return qc

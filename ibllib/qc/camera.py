"""Camera QC
This module runs a list of quality control metrics on the camera and extracted video data.

TODO Add checks from check_cam scratch
Question:
What is the order of things? I am assuming the frames, pin state, FPGA times all occur outside of the task
What are the units of the ssv timestamps?  Can we use these as a measure of Bpod timestamp accuracy (these are interpolated)
What to do if the pin_state file exists but not the count file?
"""

import logging
import sys
from datetime import datetime, timedelta
from inspect import getmembers, isfunction
from functools import reduce
from collections.abc import Sized

import cv2
import numpy as np
from scipy.stats import chisquare

from ibllib.io.extractors.camera import (
    load_embedded_frame_data, extract_camera_sync, PIN_STATE_THRESHOLD
)
from ibllib.io.extractors import training_trials
from ibllib.io.extractors import ephys_fpga
from ibllib.io import raw_data_loaders as raw
import alf.io as alfio
from brainbox.core import Bunch
from . import base

_log = logging.getLogger('ibllib')


class CameraQC(base.QC):
    """A class for computing camera QC metrics"""
    criteria = {"PASS": 0.99,
                "WARNING": 0.95,
                "FAIL": 0}
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
        self.type = raw.get_session_extractor_type(self.session_path)
        self.data = Bunch()

        # Metrics and passed trials
        self.metrics = None
        self.passed = None

    def load_data(self, partial=False, download_data=True):
        """Extract the data from raw data files
        Extracts all the required task data from the raw data files.

        :param partial: if True, assumes trials and camera times already assigned to data
        :param download_data: if True, any missing raw data is downloaded via ONE.
        """
        # if raw.get_session_extractor_type(self.session_path) == 'ephys':
        #     pass
        if download_data:
            self._ensure_required_data()

        # Get frame count and pin state
        self.data['count'], self.data['pin_state'] = load_embedded_frame_data(self.session_path, self.side)
        assert self.data['count'].size == self.data['pin_state'].size
        assert all(np.diff(self.data['count']) > 0), "frame count doesn't make sense"


        # Get audio data
        # TODO Load from trials ALF if possible
        if alfio.exists(self.session_path / 'alf', 'trials'):
            trials = alfio.load_object(self.session_path / 'alf', 'trials')
        else:
            # Extract the trial data
            # TODO Bpod only extraction
            if self.type == 'ephys':
                trials, _ = ephys_fpga.extract_all(self.session_path, save=False, bin_exists=False)
            else:
                trials, _ = training_trials.extract_all(self.session_path, save=False)

        incorrect_trial = trials['feedbackType'] < 0
        self.data['audioTTLs'] = np.sort(
            np.append(trials['goCue_times'], trials['feedback_times'][incorrect_trial]))

        # Extract camera times
        sync, chmap = ephys_fpga._get_main_probe_sync(self.session_path)
        ts = extract_camera_sync(sync, chmap)
        self.data['frame_times'] = ts[f'{self.side}_camera']

        # Gather information from video file
        self.log('inspecting video file...')
        cam_path = self.session_path / 'raw_video_data' / f'_iblrig_{self.side}Camera.raw.mp4'
        cap = cv2.VideoCapture(str(cam_path))

        # Get basic properties of video
        info = Bunch()
        info.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        info.fps = int(cap.get(cv2.CAP_PROP_FPS))
        info.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        info.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        self.data['video'] = info
        # TODO Sample frames for brightness range check

        assert self.data['count'].size == info.length  # FIXME Should be QC check

        # Load Bonsai frame timestamps
        file = self.session_path / 'raw_video_data' / f'_iblrig_{self.side}Camera.timestamps.ssv'
        # df = pd.read_csv(file, delim_whitespace=True)
        ssv_params = dict(names=('date', 'timestamp'), dtype='<M8[ns],<u4', delimiter=' ')
        self.data['bpod_frame_times'] = np.genfromtxt(file, **ssv_params)  # np.loadtxt is slower

        """Here we will ensure that the FPGA camera times match the number of video frames in 
        length.  We will make the following assumptions: 
        
        1. The number of FPGA camera times is equal to or greater than the number of video frames.
        2. No TTLs were missed between the camera and FPGA.
        3. No pin states were missed by Bonsai.
        4  No pixel count data was missed by Bonsai.

        In other words the count and pin state arrays accurately reflect the number of frames 
        sent by the camera and should therefore be the same length.
        
        The missing frame timestamps are removed in three stages:
        
        1. Remove any timestamps that occurred before video frame acquisition in Bonsai.
        2. Remove any timestamps where the frame counter reported missing frames, i.e. remove the
        dropped frames which occurred throughout the session.
        3. Remove the trailing timestamps at the end of the session if the camera was turned off
        in the wrong order.
        
        TODO Confirm this last point 
        """

        # Align on first pin state change
        # TODO It may be possible for the pin state to be high for 2 consecutive frames
        first_uptick = (self.data['pin_state'] > 0).argmax()
        first_ttl = np.searchsorted(self.data['frame_times'], self.data['audioTTLs'][0])  # - 1
        # Minus any frames that were dropped between the start of frame acquisition and the
        # first TTL
        """Here we find up to which index in the FPGA times we discard by taking the difference 
        between the index of the first pin state change (when the audio TTL was reported by the 
        camera) and the index of the first audio TTL in FPGA time.  We subtract the difference 
        between the frame count at the first pin state change and the index to account for any 
        video frames that were not saved during this period (we will remove those from the 
        camera FPGA times later).
        """
        start = first_ttl - first_uptick - (self.data['count'][first_uptick] - first_uptick)
        assert start >= 0

        # Remove the extraneous timestamps from the beginning and end
        # TODO Add case for missing FPGA timestamps
        end = self.data['count'][-1] + 1 + start
        ts = self.data['frame_times'][start:end]

        assert np.searchsorted(ts, self.data['audioTTLs'][0]) == first_uptick
        assert ts.size >= self.data['count'].size
        assert ts.size == self.data['count'][-1] + 1

        # Remove the rest of the dropped frames
        ts = ts[self.data['count']]

        # Double check everything looks okay TODO Remove this
        last_uptick = np.where(self.data['pin_state'] > 0)[0][-1]
        last_ttl = np.searchsorted(ts, self.data['audioTTLs'][-1])

        # Plot to check
        import matplotlib.pyplot as plt
        from ibllib.plots import squares, vertical_lines
        help(squares)
        y = self.data['pin_state'] > 0
        y[y == 1] = 0.0005
        y += 0.0002
        plt.plot(ts, y, marker='d', color='blue', drawstyle='steps-pre')
        plt.plot(ts, np.zeros_like(ts), 'kx')
        vertical_lines(self.data['audioTTLs'], ymin=0, ymax=1, color='r', linestyle=':')
        # vertical_lines(self.data['audioTTLs'], ymin=0, ymax=1, color='r', linestyle=':')

        """Two ways to do this: first way assumes there are always more timestamps than frames.
        This way does not make that assumption, and if the FPGA is missing timestamps at the end of
        the session they are filled with nan values.
        
        
        """
        # ts2 = self.data['frame_times'][start:]
        # incl = np.ones_like(ts2, dtype=bool)
        # dropped = np.setdiff1d(np.arange(self.data['count'][-1]), self.data['count'],
        #                        assume_unique=True)
        # incl[dropped] = False
        # ts2 = ts2[incl]
        #
        # plt.plot(ts2, y, marker='d', color='blue', drawstyle='steps-pre')
        # plt.plot(ts2, np.zeros_like(ts2), 'kx')
        # vertical_lines(self.data['audioTTLs'], ymin=0, ymax=1, color='r', linestyle=':')

        # count_diff = np.diff(self.data['count'])
        # for i, in np.where(count_diff > 1):
        #     d = int(np.diff(self.data['count'][[i, i + 1]]) - 1)
        #     ts[i:i+d]
        # assert first_ttl < first_uptick
        # aln = first_uptick - first_ttl

    def _ensure_required_data(self):
        files = self.one.load(self.eid,
                              dataset_types=self.dstypes + self.dstypes_fpga, download_only=True)
        assert not any(file is None for file in files)

    def check_video_contrast(self):
        """Check that the video contrast range"""
        pass

    def check_video_headers(self):
        """Check reported frame rate matches FPGA frame rate"""
        pass

    def check_video_framerate(self):
        """Check camera times match specified frame rate for camera"""
        pass

    def check_pin_state(self):
        """Check the pin state reflects Bpod TTLs"""
        # FIXME It may be possible for the pin state to be high for 2 consecutive frames
        if self.data['pin_state'] is None:
            return  # NOT_SET
        # There should be only one value below our threshold
        correct_threshold = sum(np.unique(self.data['pin_state']) < PIN_STATE_THRESHOLD) == 1
        state = self.data['pin_state'] > PIN_STATE_THRESHOLD
        # Trailing one or two pin states permitted as last trial dropped
        # TODO Trim pin state based on bonsai frames (if in Bpod time we can remove the last trial)
        state_ttl_matches = sum(state) != self.data['audioTTLs'].size
        return correct_threshold and state_ttl_matches

    def check_dropped_frames(self):
        """Check how many frames were reported missing"""
        assert np.all(np.diff(self.data['count']) > 0), 'frame count not strictly increasing'
        dropped = np.diff(self.data['count']).astype(int) - 1

    def check_camera_times(self):
        """Check that the camera.times array is reasonable"""
        # Check frame rate matches what we expect
        expected = 1 / self.video_meta[self.type][self.side]['fps']
        # TODO Remove dropped frames from test
        fps_matches = np.allclose(np.diff(self.data['frame_times']), expected)
        # Check number of timestamps matches video
        length_matches = self.data['frame_times'].size == self.data['video'].length

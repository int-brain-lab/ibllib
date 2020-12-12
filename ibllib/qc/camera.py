"""Camera QC
This module runs a list of quality control metrics on the camera and extracted video data.

Question:
    We're not extracting the audio based on TTL length.  Is this a problem?
"""
import sys
import logging
from inspect import getmembers, isfunction
from datetime import timedelta
from pathlib import Path
import re

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ibllib.io.extractors.camera import (
    load_embedded_frame_data, extract_camera_sync, PIN_STATE_THRESHOLD, extract_all
)
from ibllib.exceptions import ALFObjectNotFound
from ibllib.io.extractors import ephys_fpga
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io import raw_data_loaders as raw
from oneibl.stream import VideoStreamer
import alf.io as alfio
from brainbox.core import Bunch
from brainbox.io.one import path_to_url
from . import base

_log = logging.getLogger('ibllib')


def get_video_frame(video_path, frame_number):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: local path to mp4 file
    :param frame_number: video frame to be returned
    :return: numpy array corresponding to frame of interest.  Dimensions are (w, h, 3)
    """
    is_url = isinstance(video_path, str) and video_path.startswith('http')
    cap = VideoStreamer(video_path).cap if is_url else cv2.VideoCapture(str(video_path))
    # 0-based index of the frame to be decoded/captured next.
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame_image = cap.read()
    cap.release()
    return frame_image


def get_video_frames_preload(video_path, frame_numbers, mask=None, as_list=False):
    """
    Obtain numpy array corresponding to a particular video frame in video_path
    :param video_path: URL or local path to mp4 file
    :param frame_numbers: video frames to be returned
    :param mask: a logical mask or slice to apply to frames
    :param as_list: if true the frames are returned as a list, this is faster but less memory
    efficient
    :return: numpy array corresponding to frame of interest.  Default dimensions are (n, w, h, 3)
    where n = len(frame_numbers)

    Example - Load first 1000 frames, keeping only the first colour channel:
        frames = get_video_frames_preload(video_path, range(1000), mask=np.s_[:, :, 0])
    """
    is_url = isinstance(video_path, str) and video_path.startswith('http')
    cap = VideoStreamer(video_path).cap if is_url else cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), 'Failed to open video'

    if as_list:
        frame_images = [None] * len(frame_numbers)
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        frame_images = np.empty((len(frame_numbers), *frame[mask or ...].shape), np.uint8)

    for ii, i in enumerate(frame_numbers):
        sys.stdout.write(f'\rloading frame {ii}/{len(frame_numbers)}')
        sys.stdout.flush()
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_images[ii] = frame[mask]
        else:
            print(f'failed to read frame #{i}')
            if not as_list:
                frame_images[ii] = np.nan
    cap.release()
    sys.stdout.write('\x1b[2K\r')  # Erase current line in stdout
    return frame_images


def get_video_meta(video_path, one=None):
    """
    Return a bunch of video information with the fields ('length', 'fps', 'width', 'height',
    'duration', 'size')
    :param video_path: A path to the video
    :param one:
    :return:
    """
    is_url = isinstance(video_path, str) and video_path.startswith('http')
    cap = VideoStreamer(video_path).cap if is_url else cv2.VideoCapture(str(video_path))
    assert cap.isOpened(), f'Failed to open video file {video_path}'

    # Get basic properties of video
    meta = Bunch()
    meta.length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta.fps = int(cap.get(cv2.CAP_PROP_FPS))
    meta.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    meta.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    meta.duration = timedelta(seconds=meta.length / meta.fps)
    if is_url and one:
        eid = one.eid_from_path(video_path)
        name = re.match(r'.*(_iblrig_[a-z]+Camera\.raw\.)(?:[\w-]{36}\.)?(mp4)$', video_path)
        det, = one.alyx.rest('datasets', 'list', session=eid, name=''.join(name.groups()))
        meta.size = det['file_size']
    elif is_url and not one:
        meta.size = None
    else:
        meta.size = Path(video_path).stat().st_size
    cap.release()
    return meta


class CameraQC(base.QC):
    """A class for computing camera QC metrics"""
    dstypes = [
        '_iblrig_Camera.frame_counter',
        '_iblrig_Camera.GPIO',
        '_iblrig_Camera.timestamps',
        '_iblrig_taskData.raw',
        '_iblrig_taskSettings.raw',
        '_iblrig_Camera.raw',
        'camera.times'
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
        :param camera: The camera to run QC on, if None QC is run for all three cameras
        :param stream: If true and local video files not available, the data are streamed from
        the remote source.
        :param n_samples: The number of frames to sample for the position and brightness QC
        """
        # When an eid is provided, we will download the required data by default (if necessary)
        download_data = not alfio.is_session_path(session_path_or_eid)
        self.download_data = kwargs.pop('download_data', download_data)
        self.stream = kwargs.pop('stream', True)
        self.n_samples = kwargs.pop('n_samples', 2)
        super().__init__(session_path_or_eid, **kwargs)

        # Data
        self.side = side
        filename = f'_iblrig_{self.side}Camera.raw.mp4'
        self.video_path = self.session_path / 'raw_video_data' / filename
        # If local video doesn't exist, change video path to URL
        if not self.video_path.exists() and self.stream and self.one is not None:
            self.video_path = path_to_url(self.video_path, self.one)

        self.type = get_session_extractor_type(self.session_path) or None
        self.data = Bunch()
        self.frame_sampels = None
        self.frame_samples_idx = None

        # QC outcomes map
        self.metrics = None
        self.outcome = 'NOT_SET'

    def load_data(self, download_data: bool = None, extract_times: bool = False) -> None:
        """Extract the data from raw data files
        Extracts all the required task data from the raw data files.

        :param download_data: if True, any missing raw data is downloaded via ONE.
        :param extract_times: if True, the camera.times are re-extracted from the raw data
        """
        if download_data is not None:
            self.download_data = download_data
        if self.download_data:
            self._ensure_required_data()
        _log.info('Gathering data for QC')

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
            assert not extract_times
            self.data['frame_times'] = alfio.load_object(alf_path, f'{self.side}Camera')['times']
        except (ALFObjectNotFound, AssertionError):  # Re-extract
            # TODO Flag for extracting single camera's data
            kwargs = dict(sync=sync, chmap=chmap) if self.type == 'ephys' else {}
            outputs, _ = extract_all(self.session_path, self.type, save=False, **kwargs)
            self.data['frame_times'] = outputs[f'{self.side}_camera_timestamps']

        # Gather information from video file
        _log.info('inspecting video file...')
        # Get basic properties of video
        self.data['video'] = get_video_meta(self.video_path, one=self.one)
        # Sample some frames from the video file
        indices = np.linspace(100, self.data['video'].length - 100, self.n_samples).astype(int)
        self.frame_samples_idx = indices
        self.data['frame_samples'] = get_video_frames_preload(self.video_path, indices,
                                                              mask=np.s_[:, :, 0])
        # Load Bonsai frame timestamps
        ssv_times = raw.load_camera_ssv_times(self.session_path, self.side)
        self.data['bonsai_times'], self.data['camera_times'] = ssv_times

    def _ensure_required_data(self):
        """
        TODO make static method with side as optional arg; download cameras individually,
        remove assert
        :return:
        """
        assert self.one is not None, 'ONE required to download data'
        # Get extractor type
        is_ephys = 'ephys' in (self.type or self.one.get_details(self.eid)['task_protocol'])
        dtypes = self.dstypes + self.dstypes_fpga if is_ephys else self.dstypes
        files = self.one.load(self.eid,
                              dataset_types=dtypes, download_only=True)
        assert not any(file is None for file in files)
        self.type = get_session_extractor_type(self.session_path)

    def run(self, update: bool = False, **kwargs) -> (str, dict):
        """
        Run video QC checks and return outcome
        :param update: if True, updates the session QC fields on Alyx
        :param download_data: if True, downloads any missing data if required
        :param extract_times: if True, re-extracts the camera timestamps from the raw data
        :returns:
        """
        _log.info('Computing QC outcome')
        if not self.data:
            self.load_data(**kwargs)

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

    def check_contrast(self, display=False):
        """Check that the video contrast range
        Assumes that the frame samples are 2D (no colour channels)
        """
        MIN_MAX = [20, 80]
        MIN_STD = 20
        brightness = self.data['frame_samples'].mean(axis=(1, 2))
        # dims = self.data['frame_samples'].shape
        # brightness /= np.array((*dims[1:], 255)).prod()  # Normalize

        within_range = np.logical_and(brightness > MIN_MAX[0],
                                      brightness < MIN_MAX[1])
        passed = within_range.all() and np.std(brightness) < MIN_STD
        if display:
            plt.figure()
            plt.plot(brightness, label='brightness')
            ax = plt.gca()
            ax.set(
                xlabel='brightness (mean pixel)',
                ylabel='uniformly sampled frames',
                title='Brightness')
            ax.hlines(MIN_MAX, 0, self.n_samples, colors='r', linestyles=':', label='bounds')
            ax.legend()

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
            plt.legend()
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
        # Check times do not contain nans
        nanless = not np.isnan(self.data['frame_times']).any()
        return 'PASS' if increasing and fps_matches and length_matches and nanless else 'FAIL'

    def check_resolution(self):
        """Check that the timestamps and video file resolution match what we expect"""
        actual = self.data['video']
        expected = self.video_meta[self.type][self.side]
        match = actual['width'] == expected['width'] and actual['height'] == expected['height']
        return 'PASS' if match else 'FAIL'

    def check_wheel_alignment(self):
        """Check wheel motion in video correlates with the rotary encoder signal"""
        pass

    def check_position(self, hist_thresh=0.7, metric=cv2.TM_CCOEFF_NORMED,
                       display=False, test=False):
        """Check camera is positioned correctly
        For the template matching zero-normalized cross-correlation (default) should be more
        robust to exposer (which we're not checking here).  The L2 norm (TM_SQDIFF) should
        also work.
        """
        if self.side in ('right', 'body'):
            return 'NOT_SET'
        # TODO Save only histogram
        refs = self.load_reference_frames('left')

        # Method 1: compareHist
        ref_h = cv2.calcHist([refs[0]], [0], None, [256], [0, 256])
        frames = refs if test else self.data['frame_samples']
        hists = [cv2.calcHist([x], [0], None, [256], [0, 256]) for x in frames]
        corr = [cv2.compareHist(test_h, ref_h, cv2.HISTCMP_CORREL) for test_h in hists]
        hist_correlates = all(x > hist_thresh for x in corr)

        # Method 2:
        # roi = (
        #     np.arange(138, 501, dtype=int),  # col
        #     np.arange(45, 346, dtype=int)    # row
        # )
        roi = {
            'left': ((45, 346), (138, 501))
        }
        template = refs[0][tuple(slice(*r) for r in roi['left'])]
        (y1, y2), (x1, x2) = roi['left']
        top_left = []
        for frame in frames:
            res = cv2.matchTemplate(frame, template, metric)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            top_left.append(min_loc if metric < 2 else max_loc)
            # bottom_right = (top_left[0] + w, top_left[1] + h)
        err = (x1, y1) - np.median(np.array(top_left), axis=0)

        if display:
            plt.figure()
            # Plot frame with template overlay
            img = frames[0]
            ax0 = plt.subplot(221)
            ax0.imshow(img, cmap='gray', vmin=0, vmax=255)
            bounds = (x1 - err[0], x2 - err[0], y2 - err[1], y1 - err[1])
            ax0.imshow(template, cmap='gray', alpha=0.5, extent=bounds)
            xy = (x1 - 60, y1 - 60)
            ax0.add_patch(Rectangle(xy, x2-x1+120, y2-y1+120,
                                    fill=True, facecolor='green', lw=0, alpha=0.2))
            xy = (x1 - err[0], y1 - err[1])
            ax0.add_patch(Rectangle(xy, x2-x1, y2-y1,
                                    edgecolor='pink', fill=False, hatch='//', lw=1))
            ax0.set(xlim=(0, img.shape[1]), ylim=(img.shape[0], 0))
            ax0.set_axis_off()
            # Plot the image histograms
            ax1 = plt.subplot(212)
            ax1.plot(ref_h[5:-1], label='reference frame')
            ax1.plot(np.array(hists).mean(axis=0)[5:-1], label='mean frame')
            ax1.set_xlim([0, 256])
            plt.legend()
            # Plot the correlations for each sample frame
            ax2 = plt.subplot(222)
            ax2.plot(corr, label='hist correlation')
            ax2.axhline(hist_thresh, 0, self.n_samples,
                        linestyle=':', color='r', label='pass threshold')
            ax2.set(xlabel='Sample Frame #', ylabel='Hist correlation')
            plt.legend()
            plt.suptitle('Check position')
            plt.show()
        face_aligned = all(np.abs(err) < 60)

        return 'PASS' if face_aligned and hist_correlates else 'FAIL'

    def check_focus(self, n=5, threshold=100, display=False, test=False):
        """Check video is in focus
        Two methods are used here: Looking at the high frequencies with a DFT and
        applying a Laplacian HPF and looking at the variance.

        Note:
            - Both methods are sensitive to noise (Laplacian is 2nd order filter).
            - The thresholds for the fft may need to be different for the left/right vs body as
              the distribution of frequencies in the image is different (e.g. the holder
              comprises mostly very high frequencies).
            - The image may be overall in focus but the places we care about can still be out of
              focus (namely the face).  For this we'll take 2 ROIs - face and wheel/paws.
        """
        if self.side in ('right', 'body'):
            return 'NOT_SET'
        if test:
            """In test mode load a reference frame and run it through a normalized box filter with
            increasing kernel size.
            """
            idx = 0
            ref = self.load_reference_frames(self.side)[idx]
            img = np.empty((n, *ref.shape), dtype=np.uint8)
            kernal_sz = np.unique(np.linspace(0, 15, n, dtype=int))
            for i, k in enumerate(kernal_sz):
                img[i] = ref if k == 0 else cv2.blur(ref, (k, k))
            if display:
                # Plot blurred images
                f, axes = plt.subplots(1, len(img))
                for ax, ig, k in zip(axes, img, kernal_sz):
                    self.imshow(ig, ax=ax, title='Kernal ({0}, {0})'.format(k or 'None'))
                f.suptitle('Reference frame with box filter')
        else:
            # Sub-sample the frame samples
            idx = np.unique(np.linspace(0, len(self.data['frame_samples']) - 1, n, dtype=int))
            img = self.data['frame_samples'][idx]

        # A measure of the sharpness effectively taking the second derivative of the image
        roi = {
            'left': (np.s_[:400, :561], np.s_[500:, 100:800])  # (face, wheel)
        }

        lpc_var = np.empty((min(n, len(img)), len(roi['left'])))
        for i, frame in enumerate(img[::-1]):
            lpc = cv2.Laplacian(frame, cv2.CV_16S, ksize=1)
            lpc_var[i] = [lpc[mask].var() for mask in roi[self.side]]

        if display:
            # Plot the first sample image
            f = plt.figure()
            gs = f.add_gridspec(len(roi[self.side]), 4)
            f.add_subplot(gs[:, 0])
            self.imshow(img[0], title=f'Frame #{self.frame_samples_idx[idx[0]]}')
            # Plot the ROIs with and without filter
            lpc = cv2.Laplacian(img[0], cv2.CV_16S, ksize=1)
            abs_lpc = cv2.convertScaleAbs(lpc)
            for i, r in enumerate(roi[self.side]):
                f.add_subplot(gs[i, 1])
                self.imshow(img[0][r], title=f'ROI #{i + 1}')
                f.add_subplot(gs[i, 2])
                self.imshow(abs_lpc[r], title=f'ROI #{i + 1} - Lapacian filter')
            f.suptitle('Laplacian blur detection')
            # TODO Add variance over frames
            ax = f.add_subplot(gs[1, 0])
            ln = plt.plot(lpc_var)
            [l.set_label(f'ROI #{i + 1}') for i, l in enumerate(ln)]
            ax.axhline(threshold, 0, n, linestyle=':', color='r', label='lower threshold')
            ax.set(xlabel='Frame sample', ylabel='Variance of the Laplacian')
            plt.legend()

        # Second test is to highpass with dft
        h, w = img.shape[1:]
        cX, cY = w // 2, h // 2
        sz = 60  # Seems to be the magic number for high pass
        mask = np.ones((h, w, 2), bool)
        mask[cY - sz:cY + sz, cX - sz:cX + sz] = False
        filt_mean = np.empty(len(img))
        for i, frame in enumerate(img[::-1]):
            dft = cv2.dft(np.float32(frame), flags=cv2.DFT_COMPLEX_OUTPUT)
            f_shift = np.fft.fftshift(dft) * mask  # Shift & remove low frequencies
            f_ishift = np.fft.ifftshift(f_shift)  # Shift back
            filt_frame = cv2.idft(f_ishift)  # Reconstruct
            filt_frame = cv2.magnitude(filt_frame[..., 0], filt_frame[..., 1])
            # Re-normalize to 8-bits to make threshold simpler
            img_back = cv2.normalize(filt_frame, None, alpha=0, beta=256,
                                     norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            filt_mean[i] = np.mean(img_back)
            if i == len(img) - 1 and display:
                # Plot Fourier transforms
                f, axes = plt.subplots(1, 3)
                self.imshow(img[0], ax=axes[0], title='Original frame')
                dft_shift = np.fft.fftshift(dft)
                magnitude = 20 * np.log(cv2.magnitude(dft_shift[..., 0], dft_shift[..., 1]))
                self.imshow(magnitude, ax=axes[1], title='Magnitude spectrum')
                self.imshow(img_back, ax=axes[2], title='Filtered frame')
                f.suptitle('Discrete Fourier Transform')
                plt.show()
        return 'PASS' if np.all(lpc_var > threshold) and np.all(filt_mean > threshold) else 'FAIL'

    @staticmethod
    def load_reference_frames(side):
        refs = [np.load(str(x)) for x in Path(__file__).parent.glob('ref*.npy')]
        refs = np.c_[refs]
        return refs

    @staticmethod
    def imshow(frame, ax=None, title=None, **kwargs):
        """plt.imshow with some convenient defaults for greyscale frames"""
        h = ax or plt.gca()
        defaults = {
            'cmap': kwargs.pop('cmap', 'gray'),
            'vmin': kwargs.pop('vmin', 0),
            'vmax': kwargs.pop('vmax', 255)
        }
        h.imshow(frame, **defaults, **kwargs)
        h.set(title=title)
        h.set_axis_off()
        return ax


def run_all_qc(session, update=False, cameras=('left', 'right', 'body'), **kwargs):
    """Run QC for all cameras
    Run the camera QC for left, right and body cameras.
    :param session: A session path or eid.
    :param update: If True, QC fields are updated on Alyx.
    :param cameras: A list of camera names to perform QC on.
    :return: dict of CameraCQ objects
    """
    qc = {}
    for camera in cameras:
        qc[camera] = CameraQC(session, side=camera, one=kwargs.pop('one', None))
        qc[camera].run(update=update, **kwargs)
    return qc

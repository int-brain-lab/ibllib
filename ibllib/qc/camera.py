"""Camera QC
This module runs a list of quality control metrics on the camera and extracted video data.

Example - Run right camera QC, downloading all but video file
    qc = CameraQC(eid, 'right', download_data=True, stream=True)
    qc.run()

Example - Run left camera QC with session path, update QC field in Alyx
    qc = CameraQC(session_path, 'left')
    outcome, extended = qc.run(update=True)  # Returns outcome of videoQC only
    print(f'video QC = {outcome}; overall session QC = {qc.outcome}')  # NB difference outcomes

Example - Run only video QC (no timestamp/alignment checks) on 20 frames for the body camera
    qc = CameraQC(eid, 'body', n_samples=20)
    qc.load_video_data()  # Quicker than loading all data
    qc.run()

Example - Run specific video QC check and display the plots
    qc = CameraQC(eid, 'left;)
    qc.load_data(download_data=True)
    qc.check_position(display=True)  # NB: Not all checks make plots

Example - Run the QC for all cameras
    qcs = run_all_qc(eid)
    qcs['left'].metrics  # Dict of checks and outcomes for left camera
"""
import logging
from inspect import getmembers, isfunction
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import one.alf.io as alfio
from one.util import filter_datasets
from one.alf.spec import is_session_path
from one.alf.exceptions import ALFObjectNotFound
from iblutil.util import Bunch
from iblutil.numerical import within_ranges

from ibllib.io.extractors.camera import extract_camera_sync, extract_all
from ibllib.io.extractors import ephys_fpga, training_wheel
from ibllib.io.extractors.video_motion import MotionAlignment
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io import raw_data_loaders as raw
import brainbox.behavior.wheel as wh
from ibllib.io.video import get_video_meta, get_video_frames_preload, assert_valid_label
from . import base

_log = logging.getLogger('ibllib')


class CameraQC(base.QC):
    """A class for computing camera QC metrics"""
    dstypes = [
        '_iblrig_Camera.frame_counter',
        '_iblrig_Camera.GPIO',
        '_iblrig_Camera.timestamps',
        '_iblrig_taskData.raw',
        '_iblrig_taskSettings.raw',
        '_iblrig_Camera.raw',
        'camera.times',
        'wheel.position',
        'wheel.timestamps'
    ]
    dstypes_fpga = [
        '_spikeglx_sync.channels',
        '_spikeglx_sync.polarities',
        '_spikeglx_sync.times',
        'ephysData.raw.meta'
    ]
    """Recall that for the training rig there is only one side camera at 30 Hz and 1280 x 1024 px.
    For the recording rig there are two label cameras (left: 60 Hz, 1280 x 1024 px;
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

    def __init__(self, session_path_or_eid, camera, **kwargs):
        """
        :param session_path_or_eid: A session id or path
        :param camera: The camera to run QC on, if None QC is run for all three cameras
        :param n_samples: The number of frames to sample for the position and brightness QC
        :param stream: If true and local video files not available, the data are streamed from
        the remote source.
        :param log: A logging.Logger instance, if None the 'ibllib' logger is used
        :param one: An ONE instance for fetching and setting the QC on Alyx
        """
        # When an eid is provided, we will download the required data by default (if necessary)
        download_data = not is_session_path(session_path_or_eid)
        self.download_data = kwargs.pop('download_data', download_data)
        self.stream = kwargs.pop('stream', None)
        self.n_samples = kwargs.pop('n_samples', 100)
        super().__init__(session_path_or_eid, **kwargs)

        # Data
        self.label = assert_valid_label(camera)
        filename = f'_iblrig_{self.label}Camera.raw*.mp4'
        raw_video_path = self.session_path.joinpath('raw_video_data')
        self.video_path = next(raw_video_path.glob(filename), None)

        # If local video doesn't exist, change video path to URL
        if not self.video_path and self.stream is not False and self.one is not None:
            try:
                self.stream = True
                self.video_path = self.one.path2url(raw_video_path / filename.replace('*', ''))
            except (StopIteration, ALFObjectNotFound):
                _log.error('No remote or local video file found')
                self.video_path = None

        logging.disable(logging.CRITICAL)
        self._type = get_session_extractor_type(self.session_path) or None
        logging.disable(logging.NOTSET)
        keys = ('count', 'pin_state', 'audio', 'fpga_times', 'wheel', 'video',
                'frame_samples', 'timestamps', 'camera_times', 'bonsai_times')
        self.data = Bunch.fromkeys(keys)
        self.frame_samples_idx = None

        # QC outcomes map
        self.metrics = None
        self.outcome = 'NOT_SET'

    @property
    def type(self):
        """
        Returns the camera type based on the protocol.
        :return: Returns either None, 'ephys' or 'training'
        """
        if not self._type:
            return
        else:
            return 'ephys' if 'ephys' in self._type else 'training'

    def load_data(self, download_data: bool = None,
                  extract_times: bool = False, load_video: bool = True) -> None:
        """Extract the data from raw data files
        Extracts all the required task data from the raw data files.

        Data keys:
            - count (int array): the sequential frame number (n, n+1, n+2...)
            - pin_state (): the camera GPIO pin; records the audio TTLs; should be one per frame
            - audio (float array): timestamps of audio TTL fronts
            - fpga_times (float array): timestamps of camera TTLs recorded by FPGA
            - timestamps (float array): extracted video timestamps (the camera.times ALF)
            - bonsai_times (datetime array): system timestamps of video PC; should be one per frame
            - camera_times (float array): camera frame timestamps extracted from frame headers
            - wheel (Bunch): rotary encoder timestamps, position and period used for wheel motion
            - video (Bunch): video meta data, including dimensions and FPS
            - frame_samples (h x w x n array): array of evenly sampled frames (1 colour channel)

        :param download_data: if True, any missing raw data is downloaded via ONE.
        Missing data will raise an AssertionError
        :param extract_times: if True, the camera.times are re-extracted from the raw data
        :param load_video: if True, calls the load_video_data method
        """
        assert self.session_path, 'no session path set'
        if download_data is not None:
            self.download_data = download_data
        if self.download_data and self.eid and self.one and not self.one.offline:
            self.ensure_required_data()
        _log.info('Gathering data for QC')

        # Get frame count and pin state
        self.data['count'], self.data['pin_state'] = \
            raw.load_embedded_frame_data(self.session_path, self.label, raw=True)

        # Load the audio and raw FPGA times
        if self.type == 'ephys':
            sync, chmap = ephys_fpga.get_main_probe_sync(self.session_path)
            audio_ttls = ephys_fpga.get_sync_fronts(sync, chmap['audio'])
            self.data['audio'] = audio_ttls['times']  # Get rises
            # Load raw FPGA times
            cam_ts = extract_camera_sync(sync, chmap)
            self.data['fpga_times'] = cam_ts[self.label]
        else:
            bpod_data = raw.load_data(self.session_path)
            _, audio_ttls = raw.load_bpod_fronts(self.session_path, bpod_data)
            self.data['audio'] = audio_ttls['times']

        # Load extracted frame times
        alf_path = self.session_path / 'alf'
        try:
            assert not extract_times
            self.data['timestamps'] = alfio.load_object(
                alf_path, f'{self.label}Camera', short_keys=True)['times']
        except AssertionError:  # Re-extract
            kwargs = dict(video_path=self.video_path, labels=self.label)
            if self.type == 'ephys':
                kwargs = {**kwargs, 'sync': sync, 'chmap': chmap}  # noqa
            outputs, _ = extract_all(self.session_path, self.type, save=False, **kwargs)
            self.data['timestamps'] = outputs[f'{self.label}_camera_timestamps']
        except ALFObjectNotFound:
            _log.warning('no camera.times ALF found for session')

        # Get audio and wheel data
        wheel_keys = ('timestamps', 'position')
        try:
            self.data['wheel'] = alfio.load_object(alf_path, 'wheel', short_keys=True)
        except ALFObjectNotFound:
            # Extract from raw data
            if self.type == 'ephys':
                wheel_data = ephys_fpga.extract_wheel_sync(sync, chmap)
            else:
                wheel_data = training_wheel.get_wheel_position(self.session_path)
            self.data['wheel'] = Bunch(zip(wheel_keys, wheel_data))

        # Find short period of wheel motion for motion correlation.
        if data_for_keys(wheel_keys, self.data['wheel']) and self.data['timestamps'] is not None:
            self.data['wheel'].period = self.get_active_wheel_period(self.data['wheel'])

        # Load Bonsai frame timestamps
        try:
            ssv_times = raw.load_camera_ssv_times(self.session_path, self.label)
            self.data['bonsai_times'], self.data['camera_times'] = ssv_times
        except AssertionError:
            _log.warning('No Bonsai video timestamps file found')

        # Gather information from video file
        if load_video:
            _log.info('Inspecting video file...')
            self.load_video_data()

    def load_video_data(self):
        # Get basic properties of video
        try:
            self.data['video'] = get_video_meta(self.video_path, one=self.one)
            # Sample some frames from the video file
            indices = np.linspace(100, self.data['video'].length - 100, self.n_samples).astype(int)
            self.frame_samples_idx = indices
            self.data['frame_samples'] = get_video_frames_preload(self.video_path, indices,
                                                                  mask=np.s_[:, :, 0])
        except AssertionError:
            _log.error('Failed to read video file; setting outcome to CRITICAL')
            self._outcome = 'CRITICAL'

    @staticmethod
    def get_active_wheel_period(wheel, duration_range=(3., 20.), display=False):
        """
        Attempts to find a period of movement where the wheel accelerates and decelerates for
        the wheel motion alignment QC.
        :param wheel: A Bunch of wheel timestamps and position data
        :param duration_range: The candidates must be within min/max duration range
        :param display: If true, plot the selected wheel movement
        :return: 2-element array comprising the start and end times of the active period
        """
        pos, ts = wh.interpolate_position(wheel.timestamps, wheel.position)
        v, acc = wh.velocity_smoothed(pos, 1000)
        on, off, *_ = wh.movements(ts, acc, pos_thresh=.1, make_plots=False)
        edges = np.c_[on, off]
        indices, _ = np.where(np.logical_and(
            np.diff(edges) > duration_range[0], np.diff(edges) < duration_range[1]))
        if len(indices) == 0:
            _log.warning('No period of wheel movement found for motion alignment.')
            return None
        # Pick movement somewhere in the middle
        i = indices[int(indices.size / 2)]
        if display:
            _, (ax0, ax1) = plt.subplots(2, 1, sharex='all')
            mask = np.logical_and(ts > edges[i][0], ts < edges[i][1])
            ax0.plot(ts[mask], pos[mask])
            ax1.plot(ts[mask], acc[mask])
        return edges[i]

    def ensure_required_data(self):
        """
        Ensures the datasets required for QC are local.  If the download_data attribute is True,
        any missing data are downloaded.  If all the data are not present locally at the end of
        it an exception is raised.  If the stream attribute is True, the video file is not
        required to be local, however it must be remotely accessible.
        NB: Requires a valid instance of ONE and a valid session eid.
        :return:
        """
        assert self.one is not None, 'ONE required to download data'
        # dataset collections outside this list are ignored (e.g. probe00, raw_passive_data)
        collections = ('alf', 'raw_ephys_data', 'raw_behavior_data', 'raw_video_data')
        # Get extractor type
        is_ephys = 'ephys' in (self.type or self.one.get_details(self.eid)['task_protocol'])
        dtypes = self.dstypes + self.dstypes_fpga if is_ephys else self.dstypes
        assert_unique = True
        # Check we have raw ephys data for session
        if is_ephys and len(self.one.list_datasets(self.eid, collection='raw_ephys_data')) == 0:
            # Assert 3A probe model; if so download all probe data
            det = self.one.get_details(self.eid, full=True)
            probe_model = next(x['model'] for x in det['probe_insertion'])
            assert probe_model == '3A', 'raw ephys data missing'
            collections += ('raw_ephys_data/probe00', 'raw_ephys_data/probe01')
            assert_unique = False
        for dstype in dtypes:
            datasets = self.one.type2datasets(self.eid, dstype, details=True)
            if 'camera' in dstype.lower():  # Download individual camera file
                datasets = filter_datasets(datasets, filename=f'.*{self.label}.*')
            else:  # Ignore probe datasets, etc.
                datasets = filter_datasets(datasets, collection=collections,
                                           assert_unique=assert_unique)
            if any(x.endswith('.mp4') for x in datasets.rel_path) and self.stream:
                names = [x.split('/')[-1] for x in self.one.list_datasets(self.eid, details=False)]
                assert f'_iblrig_{self.label}Camera.raw.mp4' in names, 'No remote video file found'
                continue
            optional = ('camera.times', '_iblrig_Camera.raw', 'wheel.position',
                        'wheel.timestamps', '_iblrig_Camera.frame_counter', '_iblrig_Camera.GPIO')
            present = (
                self.one._download_datasets(datasets)
                if self.download_data
                else (next(self.session_path.rglob(d), None) for d in datasets['rel_path'])
            )

            required = (dstype not in optional)
            all_present = not datasets.empty and all(present)
            assert all_present or not required, f'Dataset {dstype} not found'
        self._type = get_session_extractor_type(self.session_path)

    def run(self, update: bool = False, **kwargs) -> (str, dict):
        """
        Run video QC checks and return outcome
        :param update: if True, updates the session QC fields on Alyx
        :param download_data: if True, downloads any missing data if required
        :param extract_times: if True, re-extracts the camera timestamps from the raw data
        :returns: overall outcome as a str, a dict of checks and their outcomes
        """
        _log.info(f'Computing QC outcome for {self.label} camera, session {self.eid}')
        namespace = f'video{self.label.capitalize()}'
        if all(x is None for x in self.data.values()):
            self.load_data(**kwargs)
        if self.data['frame_samples'] is None or self.data['timestamps'] is None:
            return 'NOT_SET', {}
        if self.data['timestamps'].shape[0] == 0:
            _log.error(f'No timestamps for {self.label} camera; setting outcome to CRITICAL')
            return 'CRITICAL', {}

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(CameraQC, is_metric)
        self.metrics = {f'_{namespace}_' + k[6:]: fn(self) for k, fn in checks}

        values = [x if isinstance(x, str) else x[0] for x in self.metrics.values()]
        code = max(base.CRITERIA[x] for x in values)
        outcome = next(k for k, v in base.CRITERIA.items() if v == code)

        if update:
            extended = {
                k: None if v is None or v == 'NOT_SET'
                else base.CRITERIA[v] < 3 if isinstance(v, str)
                else (base.CRITERIA[v[0]] < 3, *v[1:])  # Convert first value to bool if array
                for k, v in self.metrics.items()
            }
            self.update_extended_qc(extended)
            self.update(outcome, namespace)
        return outcome, self.metrics

    def check_brightness(self, bounds=(40, 200), max_std=20, roi=True, display=False):
        """Check that the video brightness is within a given range
        The mean brightness of each frame must be with the bounds provided, and the standard
        deviation across samples frames should be less then the given value.  Assumes that the
        frame samples are 2D (no colour channels).

        :param bounds: For each frame, check that: bounds[0] < M < bounds[1],
        where M = mean(frame). If less than 75% of sample frames outside of these bounds, the
        outcome is WARNING. If <75% of frames within twice the bounds, the outcome is FAIL.
        :param max_std: The standard deviation of the frame luminance means must be less than this
        :param roi: If True, check brightness on ROI of frame
        :param display: When True the mean frame luminance is plotted against sample frames.
        The sample frames with the lowest and highest mean luminance are shown.
        """
        if self.data['frame_samples'] is None:
            return 'NOT_SET'
        if roi is True:
            _, h, w = self.data['frame_samples'].shape
            if self.label == 'body':  # Latter half
                roi = (slice(None), slice(None), slice(int(w / 2), None, None))
            elif self.label == 'left':  # Top left quadrant (~2/3, 1/2 height)
                roi = (slice(None), slice(None, int(h / 2), None), slice(None, int(w * .66), None))
            else:  # Top right quadrant (~2/3 width, 1/2 height)
                roi = (slice(None), slice(None, int(h / 2), None), slice(int(w * .66), None, None))
        else:
            roi = (slice(None), slice(None), slice(None))
        brightness = self.data['frame_samples'][roi].mean(axis=(1, 2))
        # dims = self.data['frame_samples'].shape
        # brightness /= np.array((*dims[1:], 255)).prod()  # Normalize

        if display:
            f = plt.figure()
            gs = f.add_gridspec(2, 3)
            indices = self.frame_samples_idx
            # Plot mean frame luminance
            ax = f.add_subplot(gs[:2, :2])
            plt.plot(indices, brightness, label='brightness')
            ax.set(
                xlabel='frame #',
                ylabel='brightness (mean pixel)',
                title='Brightness')
            ax.hlines(bounds, 0, indices[-1],
                      colors='tab:orange', linestyles=':', label='warning bounds')
            ax.hlines((bounds[0] / 2, bounds[1] * 2), 0, indices[-1],
                      colors='r', linestyles=':', label='failure bounds')
            ax.legend()
            # Plot min-max frames
            for i, idx in enumerate((np.argmax(brightness), np.argmin(brightness))):
                a = f.add_subplot(gs[i, 2])
                ax.annotate('*', (indices[idx], brightness[idx]),  # this is the point to label
                            textcoords='offset points', xytext=(0, 1), ha='center')
                frame = self.data['frame_samples'][idx][roi[1:]]
                title = ('min' if i else 'max') + ' mean luminance = %.2f' % brightness[idx]
                self.imshow(frame, ax=a, title=title)

        PCT_PASS = .75  # Proportion of sample frames that must pass
        # Warning if brightness not within range (3/4 of frames must be between bounds)
        warn_range = np.logical_and(brightness > bounds[0], brightness < bounds[1])
        warn_range = 'PASS' if sum(warn_range) / self.n_samples > PCT_PASS else 'WARNING'
        # Fail if brightness not within twice the range or std less than threshold
        fail_range = np.logical_and(brightness > bounds[0] / 2, brightness < bounds[1] * 2)
        within_range = sum(fail_range) / self.n_samples > PCT_PASS
        fail_range = 'PASS' if within_range and np.std(brightness) < max_std else 'FAIL'
        return self.overall_outcome([warn_range, fail_range])

    def check_file_headers(self):
        """Check reported frame rate matches FPGA frame rate"""
        if None in (self.data['video'], self.video_meta):
            return 'NOT_SET'
        expected = self.video_meta[self.type][self.label]
        return 'PASS' if self.data['video']['fps'] == expected['fps'] else 'FAIL'

    def check_framerate(self, threshold=1.):
        """Check camera times match specified frame rate for camera

        :param threshold: The maximum absolute difference between timestamp sample rate and video
        frame rate.  NB: Does not take into account dropped frames.
        """
        if any(x is None for x in (self.data['timestamps'], self.video_meta)):
            return 'NOT_SET'
        fps = self.video_meta[self.type][self.label]['fps']
        Fs = 1 / np.median(np.diff(self.data['timestamps']))  # Approx. frequency of camera
        return 'PASS' if abs(Fs - fps) < threshold else 'FAIL', float(round(Fs, 3))

    def check_pin_state(self, display=False):
        """Check the pin state reflects Bpod TTLs
        """
        if not data_for_keys(('video', 'pin_state', 'audio'), self.data):
            return 'NOT_SET'
        size_diff = int(self.data['pin_state'].shape[0] - self.data['video']['length'])
        # NB: The pin state to be high for 2 consecutive frames
        low2high = np.insert(np.diff(self.data['pin_state'][:, -1].astype(int)) == 1, 0, False)
        # NB: Time between two consecutive TTLs can be sub-frame, so this will fail
        ndiff_low2high = int(self.data['audio'][::2].size - sum(low2high))
        state_ttl_matches = ndiff_low2high == 0
        # Check within ms of audio times
        if display:
            plt.Figure()
            plt.plot(self.data['timestamps'][low2high], np.zeros(sum(low2high)), 'o',
                     label='GPIO Low -> High')
            plt.plot(self.data['audio'], np.zeros(self.data['audio'].size), 'rx',
                     label='Audio TTL High')
            plt.xlabel('FPGA frame times / s')
            plt.gca().set(yticklabels=[])
            plt.gca().tick_params(left=False)
            plt.legend()

        outcome = self.overall_outcome(
            ('PASS' if size_diff == 0 else 'WARNING' if np.abs(size_diff) < 5 else 'FAIL',
             'PASS' if state_ttl_matches else 'WARNING')
        )
        return outcome, ndiff_low2high, size_diff

    def check_dropped_frames(self, threshold=.1):
        """Check how many frames were reported missing

        :param threshold: The maximum allowable percentage of dropped frames
        """
        if not data_for_keys(('video', 'count'), self.data):
            return 'NOT_SET'
        size_diff = int(self.data['count'].size - self.data['video']['length'])
        strict_increase = np.all(np.diff(self.data['count']) > 0)
        if not strict_increase:
            n_effected = np.sum(np.invert(strict_increase))
            _log.info(f'frame count not strictly increasing: '
                      f'{n_effected} frames effected ({n_effected / strict_increase.size:.2%})')
            return 'CRITICAL'
        dropped = np.diff(self.data['count']).astype(int) - 1
        pct_dropped = (sum(dropped) / len(dropped) * 100)
        # Calculate overall outcome for this check
        outcome = self.overall_outcome(
            ('PASS' if size_diff == 0 else 'WARNING' if np.abs(size_diff) < 5 else 'FAIL',
             'PASS' if pct_dropped < threshold else 'FAIL')
        )
        return outcome, int(sum(dropped)), size_diff

    def check_timestamps(self):
        """Check that the camera.times array is reasonable"""
        if not data_for_keys(('timestamps', 'video'), self.data):
            return 'NOT_SET'
        # Check frame rate matches what we expect
        expected = 1 / self.video_meta[self.type][self.label]['fps']
        # TODO Remove dropped frames from test
        frame_delta = np.diff(self.data['timestamps'])
        fps_matches = np.isclose(np.median(frame_delta), expected, atol=0.01)
        # Check number of timestamps matches video
        length_matches = self.data['timestamps'].size == self.data['video'].length
        # Check times are strictly increasing
        increasing = all(np.diff(self.data['timestamps']) > 0)
        # Check times do not contain nans
        nanless = not np.isnan(self.data['timestamps']).any()
        return 'PASS' if increasing and fps_matches and length_matches and nanless else 'FAIL'

    def check_camera_times(self):
        """Check that the number of raw camera timestamps matches the number of video frames"""
        if not data_for_keys(('bonsai_times', 'video'), self.data):
            return 'NOT_SET'
        length_match = len(self.data['camera_times']) == self.data['video'].length
        outcome = 'PASS' if length_match else 'WARNING'
        # 1 / np.median(np.diff(self.data.camera_times))
        return outcome, len(self.data['camera_times']) - self.data['video'].length

    def check_resolution(self):
        """Check that the timestamps and video file resolution match what we expect"""
        if self.data['video'] is None:
            return 'NOT_SET'
        actual = self.data['video']
        expected = self.video_meta[self.type][self.label]
        match = actual['width'] == expected['width'] and actual['height'] == expected['height']
        return 'PASS' if match else 'FAIL'

    def check_wheel_alignment(self, tolerance=(1, 2), display=False):
        """Check wheel motion in video correlates with the rotary encoder signal

        Check is skipped for body camera videos as the wheel is often obstructed

        :param tolerance: maximum absolute offset in frames.  If two values, the maximum value
        is taken as the warning threshold
        :param display: if true, the wheel motion energy is plotted against the rotary encoder
        :returns: outcome string, frame offset
        """
        wheel_present = data_for_keys(('position', 'timestamps', 'period'), self.data['wheel'])
        if not wheel_present or self.label == 'body':
            return 'NOT_SET'

        # Check the selected wheel movement period occurred within camera timestamp time
        camera_times = self.data['timestamps']
        in_range = within_ranges(camera_times, self.data['wheel']['period'].reshape(-1, 2))
        if not in_range.any():
            # Check if any camera timestamps overlap with the wheel times
            if np.any(np.logical_and(
                camera_times > self.data['wheel']['timestamps'][0],
                camera_times < self.data['wheel']['timestamps'][-1])
            ):
                _log.warning('Unable to check wheel alignment: '
                             'chosen movement is not during video')
                return 'NOT_SET'
            else:
                # No overlap, return fail
                return 'FAIL'
        aln = MotionAlignment(self.eid, self.one, self.log, session_path=self.session_path)
        aln.data = self.data.copy()
        aln.data['camera_times'] = {self.label: camera_times}
        aln.video_paths = {self.label: self.video_path}
        offset, *_ = aln.align_motion(period=self.data['wheel'].period,
                                      display=display, side=self.label)
        if offset is None:
            return 'NOT_SET'
        if display:
            aln.plot_alignment()

        # Determine the outcome.  If there are two values for the tolerance, one is taken to be
        # a warning threshold, the other a failure threshold.
        out_map = {0: 'FAIL', 1: 'WARNING', 2: 'PASS'}
        passed = np.abs(offset) <= np.sort(np.array(tolerance))
        return out_map[sum(passed)], int(offset)

    def check_position(self, hist_thresh=(75, 80), pos_thresh=(10, 15),
                       metric=cv2.TM_CCOEFF_NORMED,
                       display=False, test=False, roi=None, pct_thresh=True):
        """Check camera is positioned correctly
        For the template matching zero-normalized cross-correlation (default) should be more
        robust to exposure (which we're not checking here).  The L2 norm (TM_SQDIFF) should
        also work.

        If display is True, the template ROI (pick hashed) is plotted over a video frame,
        along with the threshold regions (green solid).  The histogram correlations are plotted
        and the full histogram is plotted for one of the sample frames and the reference frame.

        :param hist_thresh: The minimum histogram cross-correlation threshold to pass (0-1).
        :param pos_thresh: The maximum number of pixels off that the template matcher may be off
         by. If two values are provided, the lower threshold is treated as a warning boundary.
        :param metric: The metric to use for template matching.
        :param display: If true, the results are plotted
        :param test: If true a reference frame instead of the frames in frame_samples.
        :param roi: A tuple of indices for the face template in the for ((y1, y2), (x1, x2))
        :param pct_thresh: If true, the thresholds are treated as percentages
        """
        if not test and self.data['frame_samples'] is None:
            return 'NOT_SET'
        refs = self.load_reference_frames(self.label)
        # ensure iterable
        pos_thresh = np.sort(np.array(pos_thresh))
        hist_thresh = np.sort(np.array(hist_thresh))

        # Method 1: compareHist
        #### Mean hist comparison
        # ref_h = [cv2.calcHist([x], [0], None, [256], [0, 256]) for x in refs]
        # ref_h = np.array(ref_h).mean(axis=0)
        # frames = refs if test else self.data['frame_samples']
        # hists = [cv2.calcHist([x], [0], None, [256], [0, 256]) for x in frames]
        # test_h = np.array(hists).mean(axis=0)
        # corr = cv2.compareHist(test_h, ref_h, cv2.HISTCMP_CORREL)
        # if pct_thresh:
        #     corr *= 100
        # hist_passed = corr > hist_thresh
        ####
        ref_h = cv2.calcHist([refs[0]], [0], None, [256], [0, 256])
        frames = refs if test else self.data['frame_samples']
        hists = [cv2.calcHist([x], [0], None, [256], [0, 256]) for x in frames]
        corr = np.array([cv2.compareHist(test_h, ref_h, cv2.HISTCMP_CORREL) for test_h in hists])
        if pct_thresh:
            corr *= 100
        hist_passed = [np.all(corr > x) for x in hist_thresh]

        # Method 2:
        top_left, roi, template = self.find_face(roi=roi, test=test, metric=metric, refs=refs)
        (y1, y2), (x1, x2) = roi
        err = (x1, y1) - np.median(np.array(top_left), axis=0)
        h, w = frames[0].shape[:2]

        if pct_thresh:  # Threshold as percent
            # t_x, t_y = pct_thresh
            err_pct = [(abs(x) / y) * 100 for x, y in zip(err, (h, w))]
            face_passed = [all(err_pct < x) for x in pos_thresh]
        else:
            face_passed = [np.all(np.abs(err) < x) for x in pos_thresh]

        if display:
            plt.figure()
            # Plot frame with template overlay
            img = frames[0]
            ax0 = plt.subplot(221)
            ax0.imshow(img, cmap='gray', vmin=0, vmax=255)
            bounds = (x1 - err[0], x2 - err[0], y2 - err[1], y1 - err[1])
            ax0.imshow(template, cmap='gray', alpha=0.5, extent=bounds)
            if pct_thresh:
                for c, thresh in zip(('green', 'yellow'), pos_thresh):
                    t_y = (h / 100) * thresh
                    t_x = (w / 100) * thresh
                    xy = (x1 - t_x, y1 - t_y)
                    ax0.add_patch(Rectangle(xy, x2 - x1 + (t_x * 2), y2 - y1 + (t_y * 2),
                                            fill=True, facecolor=c, lw=0, alpha=0.05))
            else:
                for c, thresh in zip(('green', 'yellow'), pos_thresh):
                    xy = (x1 - thresh, y1 - thresh)
                    ax0.add_patch(Rectangle(xy, x2 - x1 + (thresh * 2), y2 - y1 + (thresh * 2),
                                            fill=True, facecolor=c, lw=0, alpha=0.05))
            xy = (x1 - err[0], y1 - err[1])
            ax0.add_patch(Rectangle(xy, x2 - x1, y2 - y1,
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
            ax2.axhline(hist_thresh[0], 0, self.n_samples,
                        linestyle=':', color='r', label='fail threshold')
            ax2.axhline(hist_thresh[1], 0, self.n_samples,
                        linestyle=':', color='g', label='pass threshold')
            ax2.set(xlabel='Sample Frame #', ylabel='Hist correlation')
            plt.legend()
            plt.suptitle('Check position')
            plt.show()

        pass_map = {i: s for i, s in enumerate(('FAIL', 'WARNING', 'PASS'))}
        face_aligned = pass_map[sum(face_passed)]
        hist_correlates = pass_map[sum(hist_passed)]

        return self.overall_outcome([face_aligned, hist_correlates], agg=min)

    def check_focus(self, n=20, threshold=(100, 6),
                    roi=False, display=False, test=False, equalize=True):
        """Check video is in focus
        Two methods are used here: Looking at the high frequencies with a DFT and
        applying a Laplacian HPF and looking at the variance.

        Note:
            - Both methods are sensitive to noise (Laplacian is 2nd order filter).
            - The thresholds for the fft may need to be different for the left/right vs body as
              the distribution of frequencies in the image is different (e.g. the holder
              comprises mostly very high frequencies).
            - The image may be overall in focus but the places we care about can still be out of
              focus (namely the face).  For this we'll take an ROI around the face.
            - Focus check thrown off by brightness.  This may be fixed by equalizing the histogram
              (set equalize=True)

        :param n: number of frames from frame_samples data to use in check.
        :param threshold: the lower boundary for Laplacian variance and mean FFT filtered
         brightness, respectively
        :param roi: if False, the roi is determined via template matching for the face or body.
        If None, some set ROIs for face and paws are used.  A list of slices may also be passed.
        :param display: if true, the results are displayed
        :param test: if true, a set of artificially blurred reference frames are used as the
        input.  This can be used to selecting reasonable thresholds.
        :param equalize: if true, the histograms of the frames are equalized, resulting in an
        increased the global contrast and linear CDF.  This makes check robust to low light
        conditions.
        """
        no_frames = self.data['frame_samples'] is None or len(self.data['frame_samples']) == 0
        if not test and no_frames:
            return 'NOT_SET'

        if roi is False:
            top_left, roi, _ = self.find_face(test=test)  # (y1, y2), (x1, x2)
            h, w = map(lambda x: np.diff(x).item(), roi)
            x, y = np.median(np.array(top_left), axis=0).round().astype(int)
            roi = (np.s_[y: y + h, x: x + w],)
        else:
            ROI = {
                'left': (np.s_[:400, :561], np.s_[500:, 100:800]),  # (face, wheel)
                'right': (np.s_[:196, 397:], np.s_[221:, 255:]),
                'body': (np.s_[143:274, 84:433],)  # body holder
            }
            roi = roi or ROI[self.label]

        if test:
            """In test mode load a reference frame and run it through a normalized box filter with
            increasing kernel size.
            """
            idx = (0,)
            ref = self.load_reference_frames(self.label)[idx]
            kernal_sz = np.unique(np.linspace(0, 15, n, dtype=int))
            n = kernal_sz.size  # Size excluding repeated kernels
            img = np.empty((n, *ref.shape), dtype=np.uint8)
            for i, k in enumerate(kernal_sz):
                img[i] = ref.copy() if k == 0 else cv2.blur(ref, (k, k))
            if equalize:
                [cv2.equalizeHist(x, x) for x in img]
            if display:
                # Plot blurred images
                f, axes = plt.subplots(1, len(kernal_sz))
                for ax, ig, k in zip(axes, img, kernal_sz):
                    self.imshow(ig, ax=ax, title='Kernal ({0}, {0})'.format(k or 'None'))
                f.suptitle('Reference frame with box filter')
        else:
            # Sub-sample the frame samples
            idx = np.unique(np.linspace(0, len(self.data['frame_samples']) - 1, n, dtype=int))
            img = self.data['frame_samples'][idx]
            if equalize:
                [cv2.equalizeHist(x, x) for x in img]

        # A measure of the sharpness effectively taking the second derivative of the image
        lpc_var = np.empty((min(n, len(img)), len(roi)))
        for i, frame in enumerate(img[::-1]):
            lpc = cv2.Laplacian(frame, cv2.CV_16S, ksize=1)
            lpc_var[i] = [lpc[mask].var() for mask in roi]

        if display:
            # Plot the first sample image
            f = plt.figure()
            gs = f.add_gridspec(len(roi) + 1, 3)
            f.add_subplot(gs[0:len(roi), 0])
            frame = img[0]
            self.imshow(frame, title=f'Frame #{self.frame_samples_idx[idx[0]]}')
            # Plot the ROIs with and without filter
            lpc = cv2.Laplacian(frame, cv2.CV_16S, ksize=1)
            abs_lpc = cv2.convertScaleAbs(lpc)
            for i, r in enumerate(roi):
                f.add_subplot(gs[i, 1])
                self.imshow(frame[r], title=f'ROI #{i + 1}')
                f.add_subplot(gs[i, 2])
                self.imshow(abs_lpc[r], title=f'ROI #{i + 1} - Lapacian filter')
            f.suptitle('Laplacian blur detection')
            # Plot variance over frames
            ax = f.add_subplot(gs[len(roi), :])
            ln = plt.plot(lpc_var)
            [l.set_label(f'ROI #{i + 1}') for i, l in enumerate(ln)]
            ax.axhline(threshold[0], 0, n, linestyle=':', color='r', label='lower threshold')
            ax.set(xlabel='Frame sample', ylabel='Variance of the Laplacian')
            plt.tight_layout()
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
                f = plt.figure()
                gs = f.add_gridspec(2, 3)
                self.imshow(img[0], ax=f.add_subplot(gs[0, 0]), title='Original frame')
                dft_shift = np.fft.fftshift(dft)
                magnitude = 20 * np.log(cv2.magnitude(dft_shift[..., 0], dft_shift[..., 1]))
                self.imshow(magnitude, ax=f.add_subplot(gs[0, 1]), title='Magnitude spectrum')
                self.imshow(img_back, ax=f.add_subplot(gs[0, 2]), title='Filtered frame')
                ax = f.add_subplot(gs[1, :])
                ax.plot(filt_mean)
                ax.axhline(threshold[1], 0, n, linestyle=':', color='r', label='lower threshold')
                ax.set(xlabel='Frame sample', ylabel='Mean of filtered frame')
                f.suptitle('Discrete Fourier Transform')
                plt.show()
        passes = np.all(lpc_var > threshold[0]) or np.all(filt_mean > threshold[1])
        return 'PASS' if passes else 'FAIL'

    def find_face(self, roi=None, test=False, metric=cv2.TM_CCOEFF_NORMED, refs=None):
        """Use template matching to find face location in frame
        For the template matching zero-normalized cross-correlation (default) should be more
        robust to exposure (which we're not checking here).  The L2 norm (TM_SQDIFF) should
        also work.  That said, normalizing the histograms works best.

        :param roi: A tuple of indices for the face template in the for ((y1, y2), (x1, x2))
        :param test: If True the template is matched against frames that come from the same session
        :param metric: The metric to use for template matching
        :param refs: An array of frames to match the template to

        :returns: (y1, y2), (x1, x2)
        """
        ROI = {
            'left': ((45, 346), (138, 501)),
            'right': ((14, 174), (430, 618)),
            'body': ((141, 272), (90, 339))
        }
        roi = roi or ROI[self.label]
        refs = self.load_reference_frames(self.label) if refs is None else refs

        frames = refs if test else self.data['frame_samples']
        template = refs[0][tuple(slice(*r) for r in roi)]
        top_left = []  # [(x1, y1), ...]
        for frame in frames:
            res = cv2.matchTemplate(frame, template, metric)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            top_left.append(min_loc if metric < 2 else max_loc)
            # bottom_right = (top_left[0] + w, top_left[1] + h)
        return top_left, roi, template

    @staticmethod
    def load_reference_frames(side):
        """Load some reference frames for a given video

        The reference frames are from sessions where the camera was well positioned. The
        frames are in qc/reference, one file per camera, only one channel per frame.  The
        session eids can be found in qc/reference/frame_src.json

        :param side: Video label, e.g. 'left'
        :return: numpy array of frames with the shape (n, h, w)
        """
        file = next(Path(__file__).parent.joinpath('reference').glob(f'frames_{side}.npy'))
        refs = np.load(file)
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


def data_for_keys(keys, data):
    """Check keys exist in 'data' dict and contain values other than None"""
    return data is not None and all(k in data and data.get(k, None) is not None for k in keys)


def run_all_qc(session, cameras=('left', 'right', 'body'), **kwargs):
    """Run QC for all cameras
    Run the camera QC for left, right and body cameras.
    :param session: A session path or eid.
    :param update: If True, QC fields are updated on Alyx.
    :param cameras: A list of camera names to perform QC on.
    :param stream: If true and local video files not available, the data are streamed from
    the remote source.
    :return: dict of CameraCQ objects
    """
    qc = {}
    run_args = {k: kwargs.pop(k) for k in ('download_data', 'extract_times', 'update')
                if k in kwargs.keys()}
    for camera in cameras:
        qc[camera] = CameraQC(session, camera, **kwargs)
        qc[camera].run(**run_args)
    return qc

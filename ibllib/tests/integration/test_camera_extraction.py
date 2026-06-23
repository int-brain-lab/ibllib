"""Integration tests for the camera extraction and QC.

Folders used:
    - camera/
    - ephys/ephys_choice_world_task/ibl_witten_27/2021-01-21/001/
    - ephys/choice_world_init/KS022/2019-12-10/001
    - Subjects_init/ZM_1098/2019-01-25/001/
    - training/CSHL_003/2019-04-05/001/


NB: These tests hit the main Alyx database.  This is required for full coverage of the QC (in
particular the CameraQC._ensure_required_data method and MotionAlign using eid2ref).
"""
import unittest
from unittest import mock
from pathlib import Path
from collections import OrderedDict
import logging
import tempfile
import shutil

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from iblutil.util import Bunch
import one.alf.io as alfio
from one.alf.path import get_session_path
from one.alf import spec
import one.params
from one.api import ONE

from ibllib.io.extractors.video_motion import MotionAlignment, MotionAlignmentFullSession
from ibllib.io.extractors.ephys_fpga import get_main_probe_sync
import ibllib.io.extractors.camera as camio
import ibllib.io.raw_data_loaders as raw
from ibllib.io import session_params
from ibllib.pipes.dynamic_pipeline import acquisition_description_legacy_session
import ibllib.qc.camera as camQC
from ibllib.qc.camera import CameraQC
import ibllib.io.video as vidio

from ci.tests import base


def _get_video_lengths(eid):
    urls = vidio.url_from_eid(eid)
    return {k: camio.get_video_length(v) for k, v in urls.items()}


def _save_qc_frames(qc, **kwargs):
    """
    Given a QC object, save the frames required for wheel alignment, etc.
    This may then be used as a test fixture.
    :param qc:
    :return:
    """
    if all(x is None for x in qc.data.values()):
        # Get wheel period for alignment frame indices
        qc.load_data(load_video=False, **kwargs)
    length = camio.get_video_length(qc.video_path)
    indices = np.linspace(100, length - 100, qc.n_samples).astype(int)
    frame_ids = np.insert(indices, 0, 0)  # First read is not saved and may be
    # re-read
    wheel_present = camQC.data_for_keys(('position', 'timestamps', 'period'), qc.data['wheel'])
    if wheel_present and qc.label != 'body':
        a, b = qc.data.wheel.period
        mask = np.logical_and(qc.data.timestamps >= a, qc.data.timestamps <= b)
        wheel_align_frames, = np.where(mask)
        # Again, the first read is not saved and may be re-read, so repeat the first index
        wheel_align_frames = np.insert(wheel_align_frames, 0, wheel_align_frames[0])
        frame_ids = np.r_[frame_ids, wheel_align_frames]

    # load and save the frames to file
    frames = vidio.get_video_frames_preload(qc.video_path, frame_ids)
    file = base.IntegrationTest.default_data_root() / 'camera' / (qc.eid + '_frame_samples.npy')
    if not file.parent.exists():
        file.parent.mkdir()
    np.save(file, frames)
    assert file.exists()


class TestTrainingCameraExtractor(base.IntegrationTest):

    required_files = ['camera/FMR007/2021-02-25/001', 'camera/ZFM-01867/2021-03-23/002']

    def setUp(self) -> None:
        self.training_eid = '7082d576-4eb4-41dc-a16e-8a742829a83a'  # For reference
        self.session_path = self.data_path / 'camera' / 'FMR007' / '2021-02-25' / '001'
        self.n_frames = 107913  # Number of frames in video
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Use non-interactive backend
        self.addCleanup(matplotlib.use, backend)
        self.addCleanup(plt.close, 'all')

    @base.disable_log(level=logging.WARNING)
    def test_groom_pin_state(self):
        """
        e7098000-62a0-46a4-99df-981ee2b56988 (ZFM-01867/2/2021-03-23)
            In this session there were occasions where the GPIO would change twice after
            an audio TTL, perhaps because the audio TTLs are often split up into two short
            TTLs on Bpod, some of which are caught by the camera, others not.

            The function removes the short audio TTLs then assigns the rest to the GPIO fronts.
            The unassigned audio TTL fronts and GPIO changes are removed.  Usually if the TTL
            low-to-high is not assigned to a GPIO front, neither is the high-to-low, so both are
            removed.  However sometimes because of mis-assigning (due to clock drift, short TTLs,
            faulty wiring, etc.) there are some 'orphaned' TTLs/GPIO fronts leaving us with two
            low-to-high fronts (or high-to-low) in a row.  These so-called orphaned fronts
            should be removed too.  The goal is to end up with an array of audio TTL times and
            GPIO times that are the same length.

            Debugging output states:
            - 2316 fronts TLLs less than 5ms apart
            - 11 audio TTL rises were not detected by the camera
            - 346 pin state rises could not be attributed to an audio TTL
            - 10 audio TTL falls were not detected by the camera
            - 345 pin state falls could not be attributed to an audio TTL
            - 3 orphaned TTLs removed

            The output arrays are not aligned per se, but should at least have *most* GPIO fronts
            correctly assigned to the corresponding audio TTLs.
        :return:
        """
        root = self.data_path
        session_path = root.joinpath('camera', 'ZFM-01867', '2021-03-23', '002')
        _, ts = raw.load_camera_ssv_times(session_path, 'left')
        _, (*_, gpio) = raw.load_embedded_frame_data(session_path, 'left')
        bpod_trials = raw.load_data(session_path)
        _, audio = raw.load_bpod_fronts(session_path, bpod_trials)

        # NB: syncing the timestamps to the audio doesn't work very well but we don't need it to
        # for the extraction, so long as the audio and GPIO fronts match.
        gpio, audio, _ = camio.groom_pin_state(gpio, audio, ts,
                                               take='nearest', tolerance=.5, min_diff=5e-3)
        # Do some checks
        self.assertEqual(gpio['indices'].size, audio['times'].size)
        expected = np.array([446328, 446812, 446814, 447251, 447253], dtype=int)
        np.testing.assert_array_equal(gpio['indices'][-5:], expected)
        expected = np.array([4448.100798, 4452.912398, 4452.934398, 4457.313998, 4457.335998])
        np.testing.assert_array_almost_equal(audio['times'][-5:], expected)

    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_extraction(self, mock_vc):
        """
        Mock the VideoCapture class of cv2 so that we can control the number of frames
        :param mock_vc:
        :return:
        """
        mock_vc().get.return_value = self.n_frames
        mock_vc().isOpened.return_value = True

        ext = camio.CameraTimestampsBpod(self.session_path)
        ts, _ = ext.extract(save=False)
        self.assertEqual(ts.size, self.n_frames, 'unexpected size')
        self.assertTrue(not np.isnan(ts).any(), 'nans in timestamps')
        self.assertTrue(np.all(np.diff(ts) > 0), 'timestamps not strictly increasing')
        expected = [1.27701818, 1.33762424, 1.36792727, 1.3982303, 1.42853333,
                    1.45883636, 1.48913939, 1.51944242, 1.54974545, 1.58004848]
        np.testing.assert_array_almost_equal(ts[:10], expected)

        # Test extraction parameters
        mock_vc().get.return_value = self.n_frames
        ts, _ = ext.extract(save=False, display=True, extrapolate_missing=False)
        self.assertEqual(ts.size, self.n_frames, 'unexpected size')
        self.assertEqual(np.isnan(ts).sum(), 388, 'unexpected number of nans')
        # Verify plots
        figs = [plt.figure(i) for i in plt.get_fignums()]
        lines = figs[0].axes[0].lines
        actual = {ln._label: len(ln._xy) for ln in lines}
        expected = {
            'assigned GPIO up state': 93,
            'unassigned GPIO up state': 1422,
            'sync TTL onset': 2391,
            'assigned TTL onset': 1515,
            'sync TTLs': 6624,
            'GPIO': 126,
            'cam times': 107913,
            'assigned sync TTL': 124
        }
        self.assertEqual(actual, expected, 'unexpected plot')

        lines = figs[1].axes[0].lines
        actual = {ln._label: len(ln._xy) for ln in lines}
        expected = {'GPIO': 107913, 'DAQ timestamps': 107913, 'sync TTL': 186}
        self.assertEqual(actual, expected, 'unexpected plot')

        # Test behaviour when some Bpod input values are empty
        """
        I haven't yet seen this behaviour in the wild although
        CameraTimestampsBpod._times_from_bpod looks for this.  Hence I don't expect the
        extraction to be successful, but we may need to if we discover such sessions.
        """
        trials = raw.load_data(self.session_path)
        for i in range(5):
            trials[i]['behavior_data']['Events timestamps']['Port1In'] = None
        # Should fall back on the basic extraction
        with self.assertLogs(logging.getLogger('ibllib.io.extractors.camera'), logging.CRITICAL):
            ts, _ = ext.extract(save=False, bpod_trials=trials)
        expected = np.array([25.0232, 25.0536, 25.0839, 25.1143, 25.1447])
        np.testing.assert_array_almost_equal(ts[:5], expected)

        # Test behaviour when frame count array longer than number of frames
        mock_vc().get.return_value = self.n_frames - 400
        ts, _ = ext.extract(save=False)

    @mock.patch('ibllib.io.extractors.camera.raw.load_embedded_frame_data')
    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_basic_extraction(self, mock_vc, mock_aux):
        """
        Tests extraction of a session without pin state and GPIO files, etc.
        :param mock_vc: A mock OpenCV VideoCapture object for stubbing the video length
        :param mock_aux: A mock object for stubbing the load_embedded_frame_data function
        :return:
        """
        mock_vc().get.return_value = self.n_frames
        mock_vc().isOpened.return_value = True

        # Act as though the embedded frame data files don't exist
        mock_aux.return_value = (None, [None] * 4)
        ext = camio.CameraTimestampsBpod(self.session_path)
        ts, _ = ext.extract(save=False)

        # Verify behaviour when no frame data and fewer timestamps than frames
        self.assertEqual(ts.size, self.n_frames)
        expected = np.array([15901.65804537, 15934.55278222,
                             15967.44751906, 16000.3422559, 16033.23699274])
        np.testing.assert_array_almost_equal(ts[-5:], expected)

        # Verify behaviour when no frame data and frames than timestamps
        # Expect raw Bpod times to be returned
        mock_vc().get.return_value = self.n_frames - 400
        ts, _ = ext.extract(save=False)
        self.assertEqual(ts.size, mock_vc().get.return_value)  # NB: This behaviour will change in the future


class TestTrainingCameraExtractorNew(base.IntegrationTest):
    """Starting with version 8.28.0 of IBLRIG, the first trial of the state machine is no longer used for timing
    the initial delay and waiting for the first camera trigger. Instead, the first trial is now identical to the
    remaining trials. Here we test the extraction of the camera timestamps from such sessions. The integration data
    has a purposefully added 1-second delay between starting the camera and start of the first trial - this is not
    normally the case. It has been added here to test the effect of decoupling camera start and first trial on the
    synchronization of the camera timestamps with the Bpod trials."""
    required_files = ['training/8.28.0/2025-06-12/002/*',]

    def setUp(self):
        session_path = get_session_path(next(self.data_path.glob(self.required_files[0])))
        frame_data = raw.load_camera_frameData(session_path, camera='left')
        self.n_frames = len(frame_data)
        bpod_trials = raw.load_data(session_path, task_collection='raw_task_data_00')
        self.t0_bpod = bpod_trials[0]["behavior_data"]["Trial start timestamp"]
        self.ext = camio.CameraTimestampsBpod(session_path)

    @mock.patch("ibllib.io.extractors.camera.cv2.VideoCapture")
    def test_extraction_sound_sync(self, mock_vc):
        """In this test the camera timestamps are synchronized by means of the sound TTL recorded by the camera.
        The extractor should be able to handle this and extrapolate the camera triggers that have been missed before
        start of the first trial."""
        mock_vc().get.return_value = self.n_frames
        mock_vc().isOpened.return_value = True
        ts, _ = self.ext.extract(save=False, task_collection='raw_task_data_00')
        # The first camera timestamp should be BEFORE the start of the first trial.
        self.assertGreater(self.t0_bpod, ts[0])

    @mock.patch("ibllib.io.extractors.camera.raw.load_embedded_frame_data")
    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_extraction_no_frame_data(self, mock_vc, mock_aux):
        """In this test the camera can not be synchronized by means of the sound TTL. This will lead to a shift of the
        camera timestamps with respect to the Bpod trials leading to faulty synchronization."""
        mock_vc().get.return_value = self.n_frames
        mock_vc().isOpened.return_value = True
        mock_aux.return_value = (None, [None] * 4)
        ts, _ = self.ext.extract(save=False, task_collection='raw_task_data_00')
        # The first camera timestamp should be AFTER the start of the first trial.
        self.assertLess(self.t0_bpod, ts[0])


class TestEphysCameraExtractor(base.IntegrationTest):

    required_files = ['camera/SWC_054/2020-10-07/001',
                      'ephys/ephys_choice_world_task/ibl_witten_27/2021-01-21/001',
                      'Subjects_init/ZM_1098/2019-01-25/001/*.mp4']

    def setUp(self) -> None:
        self.ephys_eid = '6c6983ef-7383-4989-9183-32b1a300d17a'
        self.session_path = self.data_path / 'camera' / 'SWC_054' / '2020-10-07' / '001'
        self.groom_session_path = self.data_path.joinpath('ephys', 'ephys_choice_world_task',
                                                          'ibl_witten_27', '2021-01-21', '001')
        self.n_frames = OrderedDict(left=255617, right=641484, body=128035)
        backend = matplotlib.get_backend()
        matplotlib.use('Agg')  # Use non-interactive backend
        self.addCleanup(matplotlib.use, backend)
        self.addCleanup(plt.close, 'all')

    def tearDown(self) -> None:
        self._remove_frameData_file(self.groom_session_path, label='left')
        self._remove_frameData_file(self.session_path, label='left')
        self._remove_frameData_file(self.session_path, label='body')
        self._remove_frameData_file(self.session_path, label='right')
        return super().tearDown()

    # @unittest.skip
    # def test_load_embedded_frame_data(self):
    #     # eid = 'c7832bca-5cfb-4676-a1ec-f87cd7640ae5'  # messed up pin state
    #     pass

    def _make_frameData_file(self, session_path, label='left') -> np.array:
        """
        Creates a frameData file from old timestamps, gpio and frame_counter files
        for testing purposes
        :param session_path: Path to the session to which the frameData file belongs
        :param label: Label of the frameData file (left, right, body)
        :return: A numpy array of the frameData file
        """
        fpath = next(session_path.rglob(f'_iblrig_{label.lower()}Camera.timestamps*.ssv'), None)
        bns_ts, _ = raw.load_camera_ssv_times(session_path, camera=label)
        # Need to load the raw camera ts data from the file, apply fix for wrong order
        with open(fpath, 'r') as f:
            line = f.readline()
        type_map = OrderedDict(bonsai='<M8[ns]', camera='<u4')
        try:
            int(line.split(' ')[1])
        except ValueError:
            type_map.move_to_end('bonsai')
        ssv_params = dict(names=type_map.keys(), dtype=','.join(type_map.values()), delimiter=' ')
        ssv_times = np.genfromtxt(fpath, **ssv_params)  # np.loadtxt is slower for some reason
        cam_ts = ssv_times['camera']
        # Cast to floats to avoid type errors in reloading the data
        bns_ts = bns_ts.astype(np.float64)
        cam_ts = cam_ts.astype(np.float64)
        fc = raw.load_camera_frame_count(session_path, label).astype(np.float64)
        GPIO_file = session_path.joinpath('raw_video_data', f'_iblrig_{label}Camera.GPIO.bin')
        raw_gpio = np.fromfile(GPIO_file, dtype=np.float64).astype(np.float64)
        out = np.vstack((bns_ts, cam_ts, fc[:len(cam_ts)], raw_gpio[:len(cam_ts)])).T
        out.astype(np.float64).tofile(
            session_path / 'raw_video_data' / f'_iblrig_{label}Camera.frameData.bin')
        return out

    def _remove_frameData_file(self, session_path, label='left'):
        f = session_path.joinpath('raw_video_data', f'_iblrig_{label}Camera.frameData.bin')
        if f.exists():
            f.unlink()

    def _groom_pin_state(self):
        # ibl_witten_27\2021-01-14\001  # Can't assign a pin state
        # CSK-im-002\2021-01-16\001  # Another example
        session_path = self.groom_session_path
        _, ts = raw.load_camera_ssv_times(session_path, 'left')
        _, (*_, gpio) = raw.load_embedded_frame_data(session_path, 'left')
        bpod_trials = raw.load_data(session_path)
        _, audio = raw.load_bpod_fronts(session_path, bpod_trials)
        # NB: syncing the timestamps to the audio doesn't work very well but we don't need it to
        # for the extraction, so long as the audio and GPIO fronts match.
        gpio, audio, _ = camio.groom_pin_state(gpio, audio, ts)
        # Do some checks
        self.assertEqual(gpio['indices'].size, audio['times'].size)
        expected = np.array([164179, 164391, 164397, 164900, 164906], dtype=int)
        np.testing.assert_array_equal(gpio['indices'][-5:], expected)
        expected = np.array([2734.4496, 2737.9659, 2738.0659, 2746.4488, 2746.5488])
        np.testing.assert_array_almost_equal(audio['times'][-5:], expected)

        # Verify behaviour when audio and GPIO match in size
        _, audio_, _ = camio.groom_pin_state(gpio, audio, ts, take='nearest', tolerance=.5)
        self.assertEqual(audio, audio_)

        # Verify behaviour when there are GPIO fronts beyond number of video frames
        ts_short = ts[:gpio['indices'].max() - 10]
        gpio_, *_ = camio.groom_pin_state(gpio, audio, ts_short)
        self.assertFalse(np.any(gpio_['indices'] >= ts.size))

    def test_groom_pin_state(self):
        self._groom_pin_state()
        # Create a frameData file from old timestamps, gpio, and frame_counter files
        self._make_frameData_file(self.groom_session_path, label='left')
        # Rerun same test
        self._groom_pin_state()

    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_extraction(self, mock_vc):
        """
        Mock the VideoCapture class of cv2 so that we can control the number of frames
        :param mock_vc:
        :return:
        """
        side = 'left'
        n_frames = self.n_frames[side]  # Number of frames in video
        mock_vc().get.return_value = n_frames
        mock_vc().isOpened.return_value = True

        # out = camio.extract_all(session_path, save=False)
        ext = camio.CameraTimestampsFPGA(side, self.session_path)
        sync, chmap = get_main_probe_sync(self.session_path)

        ts, _ = ext.extract(save=False, sync=sync, chmap=chmap)
        self.assertEqual(ts.size, n_frames, 'unexpected size')
        self.assertTrue(not np.isnan(ts).any(), 'nans in timestamps')
        self.assertTrue(np.all(np.diff(ts) > 0), 'timestamps not strictly increasing')
        expected = np.array([197.76558813, 197.79905145, 197.81578311, 197.83251477,
                             197.84924643, 197.86597809, 197.88270975, 197.89944141,
                             197.91617307, 197.93290473])
        np.testing.assert_array_almost_equal(ts[:10], expected)

        # Test extraction parameters
        ts, _ = ext.extract(save=False, sync=sync, chmap=chmap,
                            display=True, extrapolate_missing=False)
        self.assertEqual(ts.size, n_frames, 'unexpected size')
        self.assertEqual(np.isnan(ts).sum(), 499, 'unexpected number of nans')
        # Verify plots
        figs = [plt.figure(i) for i in plt.get_fignums()]
        lines = figs[0].axes[0].lines
        actual = {ln._label: len(ln._xy) for ln in lines}
        expected = {
            'sync TTLs': 3400,
            'GPIO': 3394,
            'cam times': 255617,
            'assigned sync TTL': 3392
        }
        self.assertEqual(actual, expected, 'unexpected plot')

        lines = figs[1].axes[0].lines
        actual = {ln._label: len(ln._xy) for ln in lines}
        expected = {'GPIO': 255617, 'DAQ timestamps': 255617, 'sync TTL': 5088}
        self.assertEqual(actual, expected, 'unexpected plot')

    @mock.patch('ibllib.io.extractors.camera.raw.load_embedded_frame_data')
    @mock.patch('ibllib.io.extractors.camera.cv2.VideoCapture')
    def test_basic_extraction(self, mock_vc, mock_aux):
        """
        Tests extraction of a session without pin state and GPIO files, etc.
        :param mock_vc: A mock OpenCV VideoCapture object for stubbing the video length
        :param mock_aux: A mock object for stubbing the load_embedded_frame_data function
        :return:
        """
        side = 'left'
        mock_vc().get.return_value = self.n_frames[side]
        mock_vc().isOpened.return_value = True

        # Act as though the embedded frame data files don't exist
        mock_aux.return_value = (None, [None] * 4)
        ext = camio.CameraTimestampsFPGA(side, self.session_path)
        sync, chmap = get_main_probe_sync(self.session_path)
        ts, _ = ext.extract(save=False, sync=sync, chmap=chmap)

        # Verify returns unaltered FPGA times.  This behaviour will change in the future
        self.assertEqual(ts.size, 255505)
        expected = np.array([0.01363197, 0.03036363, 0.04709529, 0.06382695, 0.08055861])
        np.testing.assert_array_almost_equal(ts[:5], expected)

        # Now test fallback when GPIO or audio data are unusable (i.e. raise an assertion)
        n = 888  # Number of GPIOs (number not important)
        gpio = {'indices': np.sort(np.random.choice(np.arange(self.n_frames[side]), n)),
                'polarities': np.insert(np.random.choice([-1, 1], n - 1), 0, -1)}
        mock_aux.return_value = (np.arange(self.n_frames[side]), [None, None, None, gpio])
        with self.assertLogs(logging.getLogger('ibllib.io.extractors.camera'), logging.CRITICAL):
            ts, _ = ext.extract(save=False, sync=sync, chmap=chmap)
        # Should fallback to basic extraction
        np.testing.assert_array_almost_equal(ts[:5], expected)

    def test_get_video_length(self):
        # Verify using URL
        url = (one.params.get().HTTP_DATA_SERVER +
               '/mainenlab/Subjects/ZM_1743/2019-06-14/001/raw_video_data/'
               '_iblrig_leftCamera.raw.71cfeef2-2aa5-46b5-b88f-ca07e3d92474.mp4')
        length = camio.get_video_length(url)
        self.assertEqual(length, 144120)

        # Verify using local path
        video_path = next(
            self.data_path.joinpath('Subjects_init/ZM_1098/2019-01-25/001').rglob('*.mp4')
        )
        length = camio.get_video_length(video_path)
        self.assertEqual(length, 34442)


class TestVideoQC(base.IntegrationTest):
    """Test the video QC on some failure cases"""

    required_files = ['camera/10_sec_vids', 'camera/10sec_bad_location']

    @classmethod
    def setUpClass(cls) -> None:
        """Load a few 10 second videos for testing the various video QC checks"""
        data_path = base.IntegrationTest.default_data_root()
        video_path = data_path.joinpath('camera')
        videos = sorted(video_path.rglob('*.mp4'))
        # Instantiate using session with a video path to fool constructor.
        # To remove once we use ONE cache file
        one = ONE(**base.TEST_DB)
        dummy_id = 'd3372b15-f696-4279-9be5-98f15783b5bb'
        qc = CameraQC(dummy_id, 'left', n_samples=10, stream=False, one=one)
        qc.one = None
        qc._type = 'ephys'  # All videos come from ephys sessions
        qcs = OrderedDict()
        for video in videos:
            qc.video_path = video
            qc.label = vidio.label_from_path(video)
            qc.n_samples = 10
            qc.load_video_data()
            qcs[video] = qc.data.copy()
        cls.qc = qc
        cls.data = qcs

    def tearDown(self) -> None:
        plt.close('all')

    def test_video_checks(self, display=False):
        # A tuple of QC checks and the expected outcome for each 10 second video
        video_checks = (
            (self.qc.check_position, [10] * 5 + [30] + [10] * 5 + [40] * 4 + [10] * 3),
            (self.qc.check_focus, [10] * 18),
            (self.qc.check_brightness, [10] * 3 + [30, 10, 30, 30, 10, 10, 30, 10, 40] + [10] * 3 + [30, 10, 30]),
            (self.qc.check_file_headers, [10] * 18),
            (self.qc.check_resolution, [10] * 5 + [40, 40, 10, 10, 40, 10, 40] + [10] * 5 + [40])
        )

        # For each check get the outcome and determine whether it matches our expected outcome
        # for each video
        for (check, expected) in video_checks:
            name = check.__name__
            outcomes = []
            frame_samples = []
            for path, data in self.data.items():
                self.qc.data = data
                self.qc.label = vidio.label_from_path(path)
                outcomes.append(check())
                frame_samples.append(data.frame_samples[0])

            # If display if True, plot 1 frame per video along with its outcome
            # This is purely for manual inspection
            if display:  # Check outcomes look reasonable by eye
                fig, axes = plt.subplots(int(len(self.data) / 4), 4)
                [self.qc.imshow(frm, ax=ax, title=o.name)
                 for frm, ax, o in zip(frame_samples, axes.flatten(), outcomes)]
                fig.suptitle(name)
                plt.show()

            # Verify the outcome for each video matches what we expect
            expected = list(map(spec.QC.validate, expected))
            self.assertEqual(expected, outcomes, f'Unexpected outcome(s) for {name} video')


class TestCameraQC(base.IntegrationTest):
    """Test the video QC on some failure cases."""

    required_files = [
        'Subjects_init/ZM_1098/2019-01-25/001',
        'camera/SWC_054/2020-10-07/001',
        'camera/FMR007/2021-02-25/001',
        'camera/NR_0020/2022-01-27/001',
        'camera/6c6983ef-7383-4989-9183-32b1a300d17a_frame_samples.npy',
        'camera/7082d576-4eb4-41dc-a16e-8a742829a83a_frame_samples.npy',
        'camera/451bc9c0-113c-408e-a924-122ffe44306e_frame_samples.npy'
    ]

    def setUp(self) -> None:
        self.incomplete = self.data_path.joinpath('Subjects_init', 'ZM_1098', '2019-01-25', '001')
        self.ephys = (
            '6c6983ef-7383-4989-9183-32b1a300d17a',
            self.data_path.joinpath('camera', 'SWC_054', '2020-10-07', '001')
        )
        self.training = (
            '7082d576-4eb4-41dc-a16e-8a742829a83a',
            self.data_path.joinpath('camera', 'FMR007', '2021-02-25', '001')
        )

        self.habituation = (
            '451bc9c0-113c-408e-a924-122ffe44306e',
            self.data_path.joinpath('camera', 'NR_0020', '2022-01-27', '001')
        )
        self._call_count = -1
        self.frames = np.array([])
        self.one = ONE(mode='local', silent=True)

    def test_incomplete_session(self):
        # Verify using local path
        session_path = self.incomplete

        qc = CameraQC(session_path, 'left', stream=False, one=self.one, n_samples=20)
        outcome, extended = qc.run(update=False)
        self.assertIs(spec.QC.NOT_SET, outcome)
        expected = {}
        self.assertEqual(expected, extended)

    @mock.patch('ibllib.qc.camera.get_video_meta')
    @mock.patch('ibllib.io.video.cv2.VideoCapture')
    def test_ephys_session(self, mock_ext, mock_meta):
        """
        Tests the full QC process for an ephys session.

        Mock a load of things so we don't need the full video file.

        :param mock_ext: mock cv.VideoCapture in camera extractor module
        :param mock_meta: mock get_video_meta
        :return:
        """
        n_samples = 100
        length = 255617
        eid, session_path = self.ephys
        self.frames = np.load(self.data_path / 'camera' / f'{eid}_frame_samples.npy')
        mock_meta.return_value = \
            Bunch({'length': length, **CameraQC.video_meta['ephys']['left']})
        mock_ext().get.return_value = length
        mock_ext().read.side_effect = self.side_effect()

        # Run QC for the left label
        one = self.one
        # Now mock the video data so that extraction and QC succeed
        video_path = session_path.joinpath('raw_video_data', '_iblrig_leftCamera.raw.mp4')
        if not video_path.exists():
            video_path.touch()
            self.addCleanup(video_path.unlink)
        qc = camQC.run_all_qc(session_path, cameras=('left',), stream=False, update=False, one=one,
                              n_samples=n_samples, extract_times=True)
        self.assertIsInstance(qc, dict)
        self.assertEqual(qc['left'].type, 'ephys')
        expected = {
            '_videoLeft_brightness': spec.QC.PASS,
            '_videoLeft_camera_times': (spec.QC.PASS, 0),
            '_videoLeft_dropped_frames': (spec.QC.WARNING, 1, 1),
            '_videoLeft_file_headers': spec.QC.PASS,
            '_videoLeft_focus': spec.QC.PASS,
            '_videoLeft_framerate': (spec.QC.PASS, 59.767),
            '_videoLeft_pin_state': (spec.QC.WARNING, 2, 1),
            '_videoLeft_position': spec.QC.PASS,
            '_videoLeft_resolution': spec.QC.PASS,
            '_videoLeft_timestamps': spec.QC.PASS,
            '_videoLeft_wheel_alignment': (spec.QC.WARNING, 7)
        }
        self.assertEqual(expected, qc['left'].metrics)

    @mock.patch('ibllib.qc.camera.get_video_meta')
    @mock.patch('ibllib.io.video.cv2.VideoCapture')
    def test_training_session(self, mock_ext, mock_meta):
        """
        Tests the full QC process for a training session.  Mock a load of things so we don't need
        the ful video file.
        :param mock_ext: mock cv.VideoCapture in camera extractor module
        :param mock_meta: mock get_video_meta
        :return:
        """
        n_samples = 100
        length = 107913
        eid, session_path = self.training
        self.frames = np.load(self.data_path / 'camera' / f'{eid}_frame_samples.npy')
        mock_meta.return_value = \
            Bunch({'length': length, **CameraQC.video_meta['training']['left']})
        mock_ext().get.return_value = length
        mock_ext().read.side_effect = self.side_effect()

        qc = CameraQC(session_path, 'left',
                      stream=False, n_samples=n_samples, one=self.one, protocol='_iblrig_task_trainingChoiceWorld')
        # Add a dummy video path (we stub the VideoCapture class anyway)
        qc.video_path = session_path.joinpath('raw_video_data', '_iblrig_leftCamera.raw.mp4')
        qc.load_data(extract_times=True)
        outcome, extended = qc.run(update=False)
        self.assertEqual(spec.QC.FAIL, outcome)
        expected = {
            '_videoLeft_brightness': spec.QC.PASS,
            '_videoLeft_camera_times': (spec.QC.PASS, 0),
            '_videoLeft_dropped_frames': (spec.QC.PASS, 1, 0),
            '_videoLeft_file_headers': spec.QC.PASS,
            '_videoLeft_focus': spec.QC.FAIL,
            '_videoLeft_framerate': (spec.QC.FAIL, 32.895),
            '_videoLeft_pin_state': (spec.QC.WARNING, 1151, 0),
            '_videoLeft_position': spec.QC.PASS,
            '_videoLeft_resolution': spec.QC.PASS,
            '_videoLeft_timestamps': spec.QC.PASS,
            '_videoLeft_wheel_alignment': (spec.QC.WARNING, 27)
        }
        self.assertEqual(expected, extended)

    @mock.patch('ibllib.qc.camera.get_video_meta')
    @mock.patch('ibllib.io.video.cv2.VideoCapture')
    def test_habituation_session(self, mock_ext, mock_meta):
        """
        Tests the full QC process for a habituation session.
        :return:
        """

        n_samples = 100
        length = 66198
        eid, session_path = self.habituation
        self.frames = np.load(self.data_path / 'camera' / f'{eid}_frame_samples.npy')
        mock_meta.return_value = \
            Bunch({'length': length, **CameraQC.video_meta['training']['left']})
        mock_ext().get.return_value = length
        mock_ext().read.side_effect = self.side_effect()

        qc = CameraQC(session_path, 'left',
                      stream=False, one=self.one, protocol='_iblrig_task_habituationChoiceWorld',
                      sync_type='bpod', n_samples=n_samples)
        # Add a dummy video path (we stub the VideoCapture class anyway)
        qc.video_path = session_path.joinpath('raw_video_data', '_iblrig_leftCamera.raw.mp4')
        qc.load_data(extract_times=True)
        outcome, extended = qc.run(update=False)
        self.assertEqual(spec.QC.WARNING, outcome)
        expected = {
            '_videoLeft_brightness': spec.QC.PASS,
            '_videoLeft_camera_times': (spec.QC.PASS, 0),
            '_videoLeft_dropped_frames': (spec.QC.PASS, 0, 0),
            '_videoLeft_file_headers': spec.QC.PASS,
            '_videoLeft_focus': spec.QC.PASS,
            '_videoLeft_framerate': (spec.QC.PASS, 30.03),
            '_videoLeft_pin_state': (spec.QC.WARNING, 350, 0),
            '_videoLeft_position': spec.QC.WARNING,
            '_videoLeft_resolution': spec.QC.PASS,
            '_videoLeft_timestamps': spec.QC.PASS,
        }
        self.assertEqual(expected, extended)

    def test_load_sess_params(self):
        """Test that the session description file is used in the load_data method."""
        # First, copy files to new collection
        _, session_path = self.ephys
        task_collection = 'raw_task_data_00'
        shutil.copytree(session_path.joinpath('raw_behavior_data'),
                        session_path.joinpath(task_collection))
        self.addCleanup(shutil.rmtree, session_path.joinpath(task_collection))
        # Second, create and experiment description file
        sess_params = acquisition_description_legacy_session(session_path, save=False)
        video_meta = {'fps': 80, 'width': 640, 'height': 512}
        sess_params['devices']['cameras']['left'].update(video_meta)
        sess_params['sync'] = {'bpod': {'collection': task_collection}}  # Change to bpod sync
        self.addCleanup(session_params.write_params(session_path, sess_params).unlink)
        # Check load data method
        qc = CameraQC(session_path, camera='left', stream=False, one=self.one, n_samples=0)
        qc.load_data(load_video=False)
        self.assertCountEqual(video_meta, qc.video_meta['training']['left'])
        self.assertNotEqual(qc.video_meta['training']['left'], CameraQC.video_meta['training']['left'])
        self.assertEqual('training', qc.type)
        self.assertEqual('bpod', qc.sync)
        self.assertEqual(task_collection, qc.sync_collection)
        # Override with kwargs
        qc = CameraQC(session_path, camera='left', stream=False, one=self.one, n_samples=0,
                      sync_type='nidq', sync_collection='raw_ephys_data')
        qc.load_data(load_video=False)
        self.assertEqual('ephys', qc.type)
        self.assertEqual('nidq', qc.sync)
        self.assertEqual('raw_ephys_data', qc.sync_collection)
        # Check unrecognised namespace
        with mock.patch('ibllib.qc.camera.get_sync_namespace', return_value='foo'):
            self.assertRaises(NotImplementedError, qc.load_data, load_video=False)

    def side_effect(self):
        for frame in self.frames:
            yield True, frame


class TestWheelMotionNRG(base.IntegrationTest):

    required_files = ['camera/6c6983ef-7383-4989-9183-32b1a300d17a_frame_samples.npy',
                      'camera/SWC_054/2020-10-07/001']

    def setUp(self) -> None:
        real_eid = '6c6983ef-7383-4989-9183-32b1a300d17a'
        self.frames = np.load(self.data_path / 'camera' / f'{real_eid}_frame_samples.npy')
        self.one = ONE(**base.TEST_DB)
        self.dummy_id = self.one.search(subject='flowers')[0]  # Some eid for connecting to Alyx

    def side_effect(self):
        for frame in self.frames:
            yield True, frame

    def test_wheel_motion(self):
        side = 'left'
        period = np.array([1730.3513333, 1734.1743333])
        with mock.patch('ibllib.io.video.cv2.VideoCapture') as mock_cv:
            mock_cv().read.side_effect = self.side_effect()
            aln = MotionAlignment(self.dummy_id, one=self.one)
            aln.session_path = self.data_path / 'camera' / 'SWC_054' / '2020-10-07' / '001'
            cam = alfio.load_object(aln.session_path / 'alf', f'{side}Camera')
            aln.data.camera_times = {side: cam['times']}
            aln.video_paths = {
                side: aln.session_path / 'raw_video_data' / f'_iblrig_{side}Camera.raw.mp4'
            }
            aln.data.wheel = alfio.load_object(aln.session_path / 'alf', 'wheel')

            # Test value error when invalid period given
            with self.assertRaises(ValueError):
                aln.align_motion(period=[5000, 5000.01], side=side)

            dt_i, c, df = aln.align_motion(period=period, side=side)
        expected = np.array([0.90278801, 0.68067675, 0.73734772, 0.82648895, 0.80950881,
                             0.88054471, 0.84264046, 0.302118, 0.94302567, 0.86188695])
        np.testing.assert_array_almost_equal(expected, df[:10])
        self.assertEqual(1, dt_i)
        self.assertEqual(18.74374, round(c, 5))

        # Test saving alignment video
        with tempfile.TemporaryDirectory() as tdir:
            aln.plot_alignment(save=tdir)
            vid = next(Path(tdir).glob('*.mp4'))
            self.assertEqual(vid.name, '2018-07-13_1_flowers_l.mp4')
            # Check number of frames
            cap = cv2.VideoCapture(str(vid))
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertEqual(227, n_frames)


class TestWheelAlignment(base.IntegrationTest):

    required_files = ['training/CSHL_003/2019-04-05/001', 'ephys/choice_world_init/KS022/2019-12-10/001']

    def setUp(self) -> None:
        self.training_folder = self.data_path.joinpath('training', 'CSHL_003', '2019-04-05', '001')
        self.ephys_folder = self.data_path.joinpath(
            'ephys', 'choice_world_init', 'KS022', '2019-12-10', '001')
        if not self.ephys_folder.exists():
            raise FileNotFoundError(f'Fixture {self.ephys_folder} does not exist')
        if not self.training_folder.exists():
            raise FileNotFoundError(f'Fixture {self.training_folder} does not exist')

    def test_alignment_ephys_session(self):

        motion_class = MotionAlignmentFullSession(session_path=self.ephys_folder, label='right')
        motion_class.camera_meta['length'] = motion_class.camera_meta['fps'] * 200  # only run on 20s snippet of video
        _ = motion_class.process()

        motion_class.camera_times = motion_class.camera_times[motion_class.tdiff:]
        np.testing.assert_array_equal(motion_class.shifts[:10], np.array([8., 8., 8., 7., 4., 4., 4., 4., 4., 4.]))
        np.testing.assert_array_equal(motion_class.shifts_filt[:10], np.array([4., 4., 4., 4., 4., 4., 4., 4., 4., 4.]))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

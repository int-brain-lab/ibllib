import logging
import shutil
import tempfile
from pathlib import Path
import numpy as np
import unittest.mock

from one.api import ONE

from ibllib.pipes.video_tasks import (
    VideoCompress,
    VideoSyncQcBpod,
    VideoSyncQcNidq,
    VideoConvert,
    VideoSyncQcCamlog,
    LightningPose
)
from ibllib.io.video import get_video_meta
from ibllib.io.extractors.ephys_fpga import get_sync_and_chn_map
from ibllib.io.extractors.camera import extract_camera_sync

from ibllib.tests import base

_logger = logging.getLogger('ibllib')


@unittest.skip('TODO')
class TestVideoRegisterRaw(base.IntegrationTest):
    def setUp(self) -> None:
        pass

    def test_register(self):
        pass


class TestVideoEphysCompress(base.IntegrationTest):
    def setUp(self) -> None:
        self.folder_path = self.data_path.joinpath('ephys', 'ephys_video_init', 'ZM_1735', '2019-08-01', '001', 'raw_video_data')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('ZM_1735', '2019-08-01', '001')
        shutil.copytree(self.folder_path, self.session_path.joinpath('raw_video_data'))
        # files in this folder are not named correctly so rename
        avi_files = self.session_path.joinpath('raw_video_data').glob('*.avi')
        for file in avi_files:
            new_file = self.session_path.joinpath('raw_video_data', file.name.replace('.avi', '.raw.avi'))
            file.replace(new_file)
        self.one = ONE(**base.TEST_DB, mode='local')

    def test_compress(self):
        task = VideoCompress(self.session_path, device_collection='raw_video_data', cameras=['left', 'right', 'body'],
                             sync='nidq', one=self.one)
        status = task.run()
        assert status == 0
        task.assert_expected_outputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestVideoCompress(base.IntegrationTest):
    def setUp(self) -> None:
        self.folder_path = self.data_path.joinpath('Subjects_init', 'ZM_1085', '2019-02-12', '002', 'raw_video_data')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('ZM_1085', '2019-02-12', '002')
        shutil.copytree(self.folder_path, self.session_path.joinpath('raw_video_data'))
        self.one = ONE(**base.TEST_DB, mode='local')

    def test_compress(self):
        task = VideoCompress(self.session_path, one=self.one, device_collection='raw_video_data', cameras=['left'], sync='bpod')
        status = task.run()
        assert status == 0
        task.assert_expected_outputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestVideoConvert(base.IntegrationTest):
    def setUp(self) -> None:
        self.folder_path = self.data_path.joinpath('widefield', 'widefieldChoiceWorld', 'JC076', '2022-02-04', '002',
                                                   'raw_video_data')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('JC076', '2022-02-04', '002')
        shutil.copytree(self.folder_path, self.session_path.joinpath('raw_video_data'))
        self.orig_video = next(self.session_path.joinpath('raw_video_data').glob('*.avi'))
        self.orig_video_path = self.session_path.joinpath('raw_video_data', 'orig')
        self.orig_video_path.mkdir()
        self.avi_file = self.orig_video_path.joinpath(self.orig_video.name)
        shutil.copy(self.orig_video, self.avi_file)
        self.one = ONE(**base.TEST_DB, mode='local')

    def test_video_convert(self):
        task = VideoConvert(self.session_path, one=self.one, device_collection='raw_video_data', cameras=['left'])
        status = task.run()
        self.assertEqual(status, 0)
        task.assert_expected_outputs()

        # check that the original video has been removed
        self.assertFalse(self.orig_video.exists())

        # compare the avi and mp4 videos and make sure they give the same results
        mp4_file = next(self.session_path.joinpath('raw_video_data').glob('*.mp4'))
        avi_meta = get_video_meta(self.avi_file)
        _ = avi_meta.pop('size')
        mp4_meta = get_video_meta(mp4_file)

        # Make sure metadata is the same
        for key in avi_meta.keys():
            with self.subTest(key=key):
                self.assertEqual(avi_meta[key], mp4_meta[key])

        # This tends to fails so we'll leave it out for now
        # # Choose 3 random frames and make sure they are the same
        # frame_idx = sample(range(avi_meta['length']), 3)
        # frame_idx = [0] + frame_idx + [avi_meta['length']]  # make sure to check last and first just in case
        # for fr in frame_idx:
        #     with self.subTest(frame_id=fr):
        #         frame_avi = get_video_frame(self.avi_file, fr)
        #         frame_mp4 = get_video_frame(mp4_file, fr)
        #         np.testing.assert_array_almost_equal(frame_avi, frame_mp4)

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestVideoSyncQCBpod(base.IntegrationTest):
    def setUp(self) -> None:
        self.folder_path = self.data_path.joinpath('Subjects_init', 'ZM_1085', '2019-02-12', '002')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('ZM_1085', '2019-02-12', '002')
        shutil.copytree(self.folder_path, self.session_path)
        self.one = ONE(**base.TEST_DB, mode='local')
        task = VideoCompress(self.session_path, one=self.one, device_collection='raw_video_data', cameras=['left'])
        task.run()

    @unittest.mock.patch('ibllib.pipes.video_tasks.CameraQC')
    def test_videosync(self, mock_qc):
        task = VideoSyncQcBpod(self.session_path, device_collection='raw_video_data', cameras=['left'], sync='bpod',
                               one=self.one, collection='raw_behavior_data')
        status = task.run()
        self.assertEqual(mock_qc.call_count, 1)
        self.assertEqual(status, 0)
        task.assert_expected_outputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestVideoSyncQcCamlog(base.IntegrationTest):
    def setUp(self) -> None:
        data_path = self.data_path.joinpath('widefield', 'widefieldChoiceWorld', 'FD_01', '2022-08-04', '001')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('FD_01', '2022-08-04', '002')
        shutil.copytree(data_path.joinpath('raw_video_data'), self.session_path.joinpath('raw_video_data'))
        shutil.copytree(data_path.joinpath('raw_sync_data'), self.session_path.joinpath('raw_sync_data'))
        shutil.copytree(data_path.joinpath('raw_behavior_data'), self.session_path.joinpath('raw_behavior_data'))
        self.session_path.joinpath('raw_video_data', '_iblrig_leftCamera.raw.mp4').touch()
        self.video_length = 244162

        self.patch = unittest.mock.patch('ibllib.io.extractors.camera.get_video_length',
                                         return_value=self.video_length)
        self.patch.start()
        self.one = ONE(**base.TEST_DB, mode='local')

    @unittest.mock.patch('ibllib.qc.camera.CameraQCCamlog')
    def test_videosync(self, mock_qc):

        task = VideoSyncQcCamlog(self.session_path, device_collection='raw_video_data', sync='nidq', sync_namespace='spikeglx',
                                 sync_collection='raw_sync_data', cameras=['left'], one=self.one)
        status = task.run()
        self.assertEqual(status, 0)
        task.assert_expected_outputs()

        # check the timestamps make sense, they should just be the fpga times
        times = np.load(task.outputs[0])

        sync, chmap = get_sync_and_chn_map(self.session_path, 'raw_sync_data')
        cam_times = extract_camera_sync(sync=sync, chmap=chmap)
        left_cam_times = cam_times['left']

        np.testing.assert_array_equal(times, left_cam_times)
        mock_qc.assert_called_once()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        self.patch.stop()


class TestVideoSyncQCNidq(base.IntegrationTest):
    def setUp(self) -> None:

        self.folder_path = self.data_path.joinpath('ephys', 'choice_world_init', 'KS022', '2019-12-10', '001')
        self.temp_dir = Path(tempfile.TemporaryDirectory().name)
        self.session_path = self.temp_dir.joinpath('KS022', '2019-12-10', '001')
        self.one = ONE(**base.TEST_DB, mode='local')

        for ff in self.folder_path.rglob('*.*'):
            link = self.session_path.joinpath(ff.relative_to(self.folder_path))
            if 'alf' in link.parts and not ('dlc' in link.name or 'ROIMotionEnergy' in link.name):
                continue
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)

    @unittest.mock.patch('ibllib.qc.camera.CameraQC')
    def test_videosync(self, mock_qc):
        task = VideoSyncQcNidq(self.session_path, device_collection='raw_video_data', sync='nidq', sync_namespace='spikeglx',
                               sync_collection='raw_ephys_data', cameras=['left', 'right', 'body'], one=self.one)
        status = task.run()
        self.assertEqual(3, mock_qc.call_count)
        self.assertEqual(status, 0)
        task.assert_expected_outputs()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir)


class TestLightningPose(base.IntegrationTest):
    def setUp(self) -> None:

        self.folder_path = self.data_path.joinpath('ephys', 'choice_world_init', 'KS022', '2019-12-10', '001')
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_path = Path(self.temp_dir.name).joinpath('KS022', '2019-12-10', '001')

        for ff in self.folder_path.rglob('*.*'):
            link = self.session_path.joinpath(ff.relative_to(self.folder_path))
            if 'alf' in link.parts and not ('lightningPose' in link.name or 'ROIMotionEnergy' in link.name):
                # We symlink the lp output files as we don't actually want to run the full task during the test
                continue
            link.parent.mkdir(exist_ok=True, parents=True)
            link.symlink_to(ff)

    @unittest.mock.patch(
        "ibllib.pipes.video_tasks.LightningPose._check_env", return_value=unittest.mock.MagicMock('mock_version')
    )
    @unittest.mock.patch.object(
        LightningPose, 'scripts', Path(__file__).parents[3].joinpath('deploy', 'serverpc', 'litpose')
    )
    def test_litpose(self, mock_check_env):
        # Test the existence of the relevant iblscripts scripts separately as we are mocking the relevant check
        task = LightningPose(self.session_path,
                             device_collection='raw_video_data',
                             cameras=['left', 'right', 'body'])
        self.assertEqual(len(list(task.scripts.rglob('run_litpose.*'))), 2)
        status = task.run(overwrite=False)
        self.assertEqual(status, 0)
        task.assert_expected_outputs()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

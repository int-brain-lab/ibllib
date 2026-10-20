import logging
import shutil
import unittest.mock

import numpy as np
from numpy.testing import assert_array_almost_equal
from iblutil.util import Bunch
from one.api import ONE

from ibllib.pipes.widefield_tasks import WidefieldPreprocess, WidefieldCompress, WidefieldSync, WidefieldRegisterRaw
from ibllib.io.extractors.ephys_fpga import get_sync_and_chn_map, get_sync_fronts

from ibllib.tests import base

_logger = logging.getLogger('ibllib')


class TestWidefieldRegisterRaw(base.IntegrationTest):
    required_files = ['widefield/widefieldChoiceWorld/CSK-im-011/2021-07-21/001']

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.session_path = cls.default_data_root().joinpath(
            'widefield', 'widefieldChoiceWorld', 'CSK-im-011', '2021-07-21', '001'
        )
        if not cls.session_path.exists():
            raise unittest.SkipTest(reason=f'File not found: {cls.session_path}')
        cls.one = ONE(**base.TEST_DB, mode='local')
        # Move the data into the correct folder
        cls.data_folder = cls.session_path.joinpath('orig')
        cls.widefield_folder = cls.session_path.joinpath('raw_widefield_data')
        cls.widefield_folder.mkdir(parents=True, exist_ok=True)
        cls.alf_folder = cls.session_path.joinpath('alf', 'widefield')

        # Symlink data from original folder to the new folder
        orig_cam_file = next(cls.data_folder.glob('*.camlog'))
        new_cam_file = cls.widefield_folder.joinpath(orig_cam_file.name)
        try:
            new_cam_file.symlink_to(orig_cam_file)
        except OSError as e:
            _logger.error(f'Error creating symlink: {e}')
            shutil.copy(orig_cam_file, new_cam_file)

        orig_data_file = next(cls.data_folder.glob('dorsal_cortex*'))
        new_data_file = cls.widefield_folder.joinpath(orig_data_file.name)
        try:
            new_data_file.symlink_to(orig_data_file)
        except OSError as e:
            _logger.error(f'Error creating symlink: {e}')
            shutil.copy(orig_data_file, new_data_file)

        orig_led_wiring_file = next(cls.data_folder.glob('*widefield_wiring*'))
        new_led_wiring_file = cls.widefield_folder.joinpath(orig_led_wiring_file.name)
        try:
            new_led_wiring_file.symlink_to(orig_led_wiring_file)
        except OSError as e:
            _logger.error(f'Error creating symlink: {e}')
            shutil.copy(orig_led_wiring_file, new_led_wiring_file)

        orig_wiring_file = next(cls.data_folder.glob('*configuration.json'))  # note this might change
        new_wiring_file = cls.widefield_folder.joinpath(orig_wiring_file.name)
        try:
            new_wiring_file.symlink_to(orig_wiring_file)
        except OSError as e:
            _logger.error(f'Error creating symlink: {e}')
            shutil.copy(orig_wiring_file, new_wiring_file)

    def test_rename(self):
        """Test that the task renames the files correctly."""
        task = WidefieldRegisterRaw(self.session_path, one=self.one)
        # NB A full integration test of snapnapshot registration can be found in
        # ibllib.tests.test_base_tasks.TestRegisterRawDataTask.test_register_snapshots;
        # WidefieldRegisterRaw calls the same superclass method for snapshot registration.
        with unittest.mock.patch.object(task, 'register_snapshots') as mock_register_snapshots:
            status = task.run()
            mock_register_snapshots.assert_called_once()
        self.assertEqual(0, status)

        for exp_files in task.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            self.assertTrue(file.exists())

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._writable_tempdir is None:
            shutil.rmtree(cls.widefield_folder, ignore_errors=True)
            shutil.rmtree(cls.alf_folder.parent, ignore_errors=True)
        super().tearDownClass()


@unittest.skip('Tests failing')
class TestWidefieldPreprocessAndCompress(base.IntegrationTest):
    session_path = None
    widefield_folder = None
    data_folder = None
    alf_folder = None

    required_files = ['widefield/widefieldChoiceWorld/CSK-im-011/2021-07-21/001']

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.session_path = cls.data_path.joinpath('widefield', 'widefieldChoiceWorld', 'CSK-im-011', '2021-07-21', '001')
        if not cls.session_path.exists():
            raise unittest.SkipTest(reason=f'File not found: {cls.session_path}')
        cls.one = ONE(**base.TEST_DB, mode='local')
        # Move the data into the correct folder
        cls.data_folder = cls.session_path.joinpath('orig')
        cls.widefield_folder = cls.session_path.joinpath('raw_widefield_data')
        cls.widefield_folder.mkdir(parents=True, exist_ok=True)
        cls.alf_folder = cls.session_path.joinpath('alf', 'widefield')

        # Symlink data from original folder to the new folder
        orig_cam_file = next(cls.data_folder.glob('*.camlog'))
        new_cam_file = cls.widefield_folder.joinpath('widefieldEvents.raw.camlog')
        try:
            new_cam_file.symlink_to(orig_cam_file)
        except OSError as e:
            _logger.error(f'Error creating symlink: {e}')
            shutil.copy(orig_cam_file, new_cam_file)

        orig_data_file = next(cls.data_folder.glob('*.dat'))
        new_data_file = cls.widefield_folder.joinpath(orig_data_file.name)
        # .dat file must be a writable copy: wfield opens it with mode='r+' to write motion data
        if new_data_file.is_symlink():
            new_data_file.unlink()
        if not new_data_file.exists():
            shutil.copy(orig_data_file, new_data_file)

    def test_preprocess(self):
        task = WidefieldPreprocess(self.session_path, one=self.one)
        status = task.run(upload_plots=False)
        self.assertEqual(0, status)

        for exp_files in task.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            self.assertTrue(file.exists())
            self.assertIn(file, task.outputs)

        # Test content of files
        PRECISION = 4  # Desired decimal precision
        # U
        assert_array_almost_equal(
            np.load(self.data_folder.joinpath('U.npy')),
            np.load(self.alf_folder.joinpath('widefieldU.images.npy')),
            decimal=PRECISION,
        )
        # SVT
        assert_array_almost_equal(
            np.load(self.data_folder.joinpath('SVT.npy')),
            np.load(self.alf_folder.joinpath('widefieldSVT.uncorrected.npy')),
            decimal=PRECISION,
        )

        # Haemo corrected SVT
        assert_array_almost_equal(
            np.load(self.data_folder.joinpath('SVTcorr.npy')),
            np.load(self.alf_folder.joinpath('widefieldSVT.haemoCorrected.npy')),
            decimal=PRECISION,
        )

        # Frame average
        assert_array_almost_equal(
            np.load(self.data_folder.joinpath('frames_average.npy')),
            np.load(self.alf_folder.joinpath('widefieldChannels.frameAverage.npy')),
            decimal=PRECISION,
        )

        task.wf.remove_files()

        self.assertEqual(0, len(list(self.widefield_folder.glob('motion*'))))

    def test_compress(self):
        task = WidefieldCompress(self.session_path, one=self.one)
        status = task.run()

        self.assertEqual(0, status)

        for exp_files in task.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            self.assertTrue(file.exists())
            self.assertIn(file, task.outputs)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._writable_tempdir is None:
            shutil.rmtree(cls.widefield_folder, ignore_errors=True)
            shutil.rmtree(cls.alf_folder.parent, ignore_errors=True)
        super().tearDownClass()


class TestWidefieldSync(base.IntegrationTest):
    patch = None  # A mock of get_video_meta
    video_meta = Bunch()

    required_files = ['widefield/widefieldChoiceWorld/JC076/2022-02-04/002']

    def setUp(self):
        self.session_path = self.data_path.joinpath('widefield', 'widefieldChoiceWorld', 'JC076', '2022-02-04', '002')
        if not self.session_path.exists():
            return
        self.alf_folder = self.session_path.joinpath('alf', 'widefield')
        self.video_file = self.session_path.joinpath('raw_widefield_data', 'imaging.frames.mov')
        self.video_file.touch()
        self.video_meta.length = 2032
        self.patch = unittest.mock.patch('ibllib.io.extractors.widefield.get_video_meta', return_value=self.video_meta)
        self.patch.start()
        self.one = ONE(**base.TEST_DB, mode='local')

    def test_sync(self):
        task = WidefieldSync(self.session_path, sync_collection='raw_widefield_data', sync_namespace='spikeglx', one=self.one)
        status = task.run()
        self.assertEqual(0, status)

        for exp_files in task.signature['output_files']:
            file = self.session_path.joinpath(exp_files[1], exp_files[0])
            self.assertTrue(file.exists())
            self.assertIn(file, task.outputs)

        # Check integrity of outputs
        times = np.load(self.alf_folder.joinpath('imaging.times.npy'))
        self.assertEqual(len(times), self.video_meta['length'])
        self.assertTrue(np.all(np.diff(times) > 0))

        sync, chmap = get_sync_and_chn_map(self.session_path, 'raw_widefield_data')
        expected_times = get_sync_fronts(sync, chmap['frame_trigger'])

        np.testing.assert_array_equal(times, expected_times['times'][0::2])

        leds = np.load(self.alf_folder.joinpath('imaging.imagingLightSource.npy'))
        self.assertEqual(2, leds[0])
        self.assertCountEqual([1, 2], np.unique(leds))

    def test_video_led_sync_not_enough(self):
        # Mock video file with more frames than led timestamps
        self.video_meta.length = 2035
        task = WidefieldSync(self.session_path, sync_collection='raw_widefield_data', sync_namespace='spikeglx', one=self.one)
        expected_error = 'ValueError: More video frames than led frames detected'
        with self.assertLogs('ibllib.pipes.tasks', logging.ERROR) as log:
            status = task.run()
            self.assertTrue(len(log.output) <= 2, 'Expected at most 2 errors logged')
            self.assertIn(expected_error, log.output[-1])
        self.assertEqual(-1, status)

    def test_video_led_sync_too_many(self):
        # Mock video file with more than two extra led timestamps
        self.video_meta.length = 2029
        task = WidefieldSync(self.session_path, sync_collection='raw_widefield_data', sync_namespace='spikeglx', one=self.one)
        expected_error = 'ValueError: Led frames and video frames differ by more than 2'
        with self.assertLogs('ibllib.pipes.tasks', logging.ERROR) as log:
            status = task.run()
            self.assertTrue(len(log.output) <= 2, 'Expected at most 2 errors logged')
            self.assertIn(expected_error, log.output[-1])
        self.assertEqual(-1, status)

    def tearDown(self):
        self.video_file.unlink()
        if self.alf_folder.exists():
            shutil.rmtree(self.alf_folder.parent)
        self.patch.stop()

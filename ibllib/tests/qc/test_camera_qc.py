import unittest
from tempfile import TemporaryDirectory

import numpy as np

from oneibl.one import ONE
from ibllib.qc.camera import CameraQC
from ibllib.tests.fixtures import utils


class TestCameraQC(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.one = ONE(
            base_url="https://test.alyx.internationalbrainlab.org",
            username="test_user",
            password="TapetesBloc18",
        )

    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.session_path = utils.create_fake_session_folder(self.tempdir.name)
        utils.create_fake_raw_video_data_folder(self.session_path)
        # utils.create_fake_raw_behavior_data_folder(self.session_path)
        self.qc = CameraQC(self.session_path, one=self.one, n_samples=5,
                           side='left', stream=False, download_data=False)
        self.qc.type = 'ephys'

    def test_check_brightness(self):
        self.qc.data['frame_samples'] = np.random.rand(1280, 1024, self.qc.n_samples) * 50
        self.assertEqual('PASS', self.qc.check_brightness())
        self.assertEqual('FAIL', self.qc.check_brightness(bounds=(10, 20)))
        self.assertEqual('FAIL', self.qc.check_brightness(max_std=1e-6))
        self.qc.data['frame_samples'] = None
        self.assertEqual('NOT_SET', self.qc.check_brightness())

    def test_check_file_headers(self):
        self.qc.data['video'] = {'fps': 60.}
        self.assertEqual('PASS', self.qc.check_file_headers())
        self.qc.data['video']['fps'] = 150
        self.assertEqual('FAIL', self.qc.check_file_headers())
        self.qc.data['video'] = None
        self.assertEqual('NOT_SET', self.qc.check_file_headers())

    def test_check_framerate(self):
        FPS = 60.
        self.qc.data['video'] = {'fps': FPS}
        self.qc.data['frame_times'] = np.array([round(1 / FPS, 4)] * 1000).cumsum()
        self.assertEqual('PASS', self.qc.check_framerate())
        self.assertEqual('FAIL', self.qc.check_framerate(threshold=1e-2))
        self.qc.data['frame_times'] = None
        self.assertEqual('NOT_SET', self.qc.check_framerate())

    @unittest.skip
    def test_check_pin_state(self):
        pass

    @unittest.skip
    def test_check_dropped_frames(self):
        pass

    @unittest.skip
    def test_check_focus(self):
        pass

    @unittest.skip
    def test_check_position(self):
        pass

    def test_check_resolution(self):
        self.qc.data['video'] = {'width': 1280, 'height': 1024}
        self.assertEqual('PASS', self.qc.check_resolution())
        self.qc.data['video']['width'] = 150
        self.assertEqual('FAIL', self.qc.check_resolution())
        self.qc.data['video'] = None
        self.assertEqual('NOT_SET', self.qc.check_resolution())

    @unittest.skip
    def test_check_timestamps(self):
        pass

    @unittest.skip
    def test_check_wheel_alignment(self):
        pass

    def tearDown(self) -> None:
        self.tempdir.cleanup()


if __name__ == '__main__':
    unittest.main()

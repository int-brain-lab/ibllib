import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from one.api import ONE
from ibllib.tests import TEST_DB
from ibllib.qc.dlc import DlcQC
from ibllib.tests.fixtures import utils


class TestDlcQC(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.one = ONE(**TEST_DB)

    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.session_path = utils.create_fake_session_folder(self.tempdir.name)
        self.alf_path = utils.create_fake_alf_folder_dlc_data(self.session_path)
        self.qc = DlcQC(self.session_path, one=self.one, side='left', download_data=False)
        self.eid = 'd3372b15-f696-4279-9be5-98f15783b5bb'

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @classmethod
    def tearDownClass(cls) -> None:
        # Clear overwritten methods by destroying cached instance
        ONE.cache_clear()

    def test_ensure_data(self):
        self.qc.eid = self.eid
        self.qc.download_data = False
        # Remove file so that the test fails as intended
        if self.qc.session_path.exists():
            self.qc.session_path.joinpath('alf/_ibl_leftCamera.dlc.pqt').unlink()
        with self.assertRaises(AssertionError) as excp:
            self.qc.run(update=False)
        msg = excp.exception.args[0]
        self.assertEqual(msg, 'Dataset _ibl_leftCamera.dlc.* not found locally and download_data is False')
        # Set download_data to True. Data is not in the database so we expect a (different) error trying to download
        self.qc.download_data = True
        with self.assertRaises(AssertionError) as excp:
            self.qc.run(update=False)
        msg = excp.exception.args[0]
        self.assertEqual(msg, 'Dataset _ibl_leftCamera.dlc.* not found locally and failed to download')

    def test_check_time_trace_length_match(self):
        self.qc.data['dlc_coords'] = {'nose_tip': np.ones((2, 20)), 'pupil_r': np.ones((2, 20))}
        self.qc.data['camera_times'] = np.ones((20,))
        outcome = self.qc.check_time_trace_length_match()
        self.assertEqual('PASS', outcome)
        self.qc.data['dlc_coords']['nose_tip'] = np.ones((2, 19))
        outcome = self.qc.check_time_trace_length_match()
        self.assertEqual('FAIL', outcome)
        self.qc.data['dlc_coords']['pupil_r'] = np.ones((2, 21))
        outcome = self.qc.check_time_trace_length_match()
        self.assertEqual('FAIL', outcome)

    def test_check_trace_all_nan(self):
        self.qc.data['dlc_coords'] = {'nose_tip': np.random.random((2, 10)),
                                      'tube_r': np.random.random((2, 10))}
        outcome = self.qc.check_trace_all_nan()
        self.assertEqual('PASS', outcome)
        self.qc.data['dlc_coords']['tube_r'] = np.ones((2, 10)) * np.nan
        outcome = self.qc.check_trace_all_nan()
        self.assertEqual('PASS', outcome)
        self.qc.data['dlc_coords']['nose_tip'] = np.ones((2, 10)) * np.nan
        outcome = self.qc.check_trace_all_nan()
        self.assertEqual('FAIL', outcome)
        return

    def test_check_mean_in_bbox(self):
        self.qc.data['dlc_coords'] = {
            'nose_tip': np.vstack((np.random.randint(400, 500, (1, 10)),
                                   np.random.randint(350, 450, size=(1, 10)))),
            'tube_r': np.vstack((np.ones((2, 10)) * np.nan))}
        outcome = self.qc.check_mean_in_bbox()
        self.assertEqual('PASS', outcome)
        for side in ['body', 'right']:
            self.qc.side = side
            outcome = self.qc.check_mean_in_bbox()
            self.assertEqual('FAIL', outcome)
        self.qc.side = 'left'

    def test_check_pupil_blocked(self):
        rng = np.random.default_rng(2021)
        self.qc.data['pupilDiameter_raw'] = rng.normal(scale=7, size=100)
        # Body camera, no pupil diameter calculated, return not set
        self.qc.side = 'body'
        outcome = self.qc.check_pupil_blocked()
        self.assertEqual('NOT_SET', outcome)
        # Left camera, np.std threshold is 10 should pass
        self.qc.side = 'left'
        outcome = self.qc.check_pupil_blocked()
        self.assertEqual('PASS', outcome)
        # Right camera, np.std threshold is 5, should fail
        self.qc.side = 'right'
        outcome = self.qc.check_pupil_blocked()
        self.assertEqual('FAIL', outcome)
        # Too many nans, should fail
        self.qc.data['pupilDiameter_raw'] *= np.nan
        self.qc.side = 'left'
        outcome = self.qc.check_pupil_blocked()
        self.assertEqual('FAIL', outcome)

    def test_check_lick_detection(self):
        self.qc.side = 'body'
        outcome = self.qc.check_lick_detection()
        self.assertEqual('NOT_SET', outcome)
        self.qc.side = 'left'
        self.qc.data['dlc_coords'] = {'tongue_end_l': np.ones((2, 10)),
                                      'tongue_end_r': np.ones((2, 10))}
        outcome = self.qc.check_lick_detection()
        self.assertEqual('FAIL', outcome)
        self.qc.data['dlc_coords']['tongue_end_l'] *= np.nan
        outcome = self.qc.check_lick_detection()
        self.assertEqual('PASS', outcome)

    def test_check_pupil_diameter_snr(self):
        pupil_path = Path(__file__).parent.joinpath('..', 'fixtures', 'qc').resolve()
        pupil_data = np.load(pupil_path.joinpath('pupil_diameter.npy'))
        self.qc.data['pupilDiameter_raw'] = pupil_data[:, 0]
        self.qc.data['pupilDiameter_smooth'] = pupil_data[:, 1]
        self.qc.side = 'body'
        outcome = self.qc.check_pupil_diameter_snr()
        self.assertEqual('NOT_SET', outcome)
        self.qc.side = 'left'
        outcome = self.qc.check_pupil_diameter_snr()
        self.assertEqual(('FAIL', 6.624), outcome)
        self.qc.side = 'right'
        outcome = self.qc.check_pupil_diameter_snr()
        self.assertEqual(('PASS', 6.624), outcome)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

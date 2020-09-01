# Mock dataset
import unittest
import shutil

import numpy as np

from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.qc.oneutils import download_bpodqc_raw_data
from oneibl.one import ONE

one = ONE(
    base_url="https://test.alyx.internationalbrainlab.org",
    username="test_user",
    password="TapetesBloc18",
)


class TestBpodQCExtractors(unittest.TestCase):
    """TODO: should be an integration test in iblscripts"""

    def setUp(self):
        self.one = one
        self.eid = 'b1c968ad-4874-468d-b2e4-5ffa9b9964e9'
        self.eid_incomplete = '4e0b3320-47b7-416e-b842-c34dc9004cf8'  # Missing required datasets
        # Make sure the data exists locally
        download_bpodqc_raw_data(self.eid, one=one, fpga=True)
        self.session_path = self.one.path_from_eid(self.eid)

    def test_lazy_extract(self):
        ex = TaskQCExtractor(self.session_path, lazy=True, one=self.one)
        self.assertIsNone(ex.data)

    def test_extraction(self):
        ex = TaskQCExtractor(self.session_path, lazy=True, one=self.one)
        self.assertIsNone(ex.raw_data)

        # Test loading raw data
        ex.load_raw_data()
        self.assertIsNotNone(ex.raw_data, 'Failed to load raw data')
        self.assertIsNotNone(ex.settings, 'Failed to load task settings')
        self.assertIsNotNone(ex.BNC1, 'Failed to load BNC1')
        self.assertIsNotNone(ex.BNC2, 'Failed to load BNC2')

        # Test extraction
        data = ex.extract_data(bpod_only=True)
        expected = [
            'stimOnTrigger_times', 'stimOffTrigger_times', 'stimOn_times', 'stimOff_times',
            'stimFreeze_times', 'stimFreezeTrigger_times', 'errorCueTrigger_times', 'itiIn_times',
            'choice', 'feedbackType', 'goCueTrigger_times', 'wheel_timestamps', 'wheel_position',
            'wheel_moves_intervals', 'wheel_moves_peak_amplitude', 'firstMovement_times', 'phase',
            'goCue_times', 'rewardVolume', 'response_times', 'feedback_times', 'probabilityLeft',
            'position', 'contrast', 'quiescence', 'contrastRight', 'contrastLeft', 'outcome',
            'errorCue_times', 'valveOpen_times', 'correct', 'intervals_0', 'intervals_1'
        ]
        self.assertCountEqual(expected, data.keys())
        self.assertEqual('ephys', ex.type)
        self.assertEqual('X1', ex.wheel_encoding)

    def test_partial_extraction(self):
        ex = TaskQCExtractor(self.session_path, lazy=True, one=self.one)
        ex.load_raw_data()
        data = ex.extract_data(bpod_only=True, partial=True)
        expected = [
            'stimOnTrigger_times', 'stimOffTrigger_times', 'stimOn_times', 'stimOff_times',
            'stimFreeze_times', 'stimFreezeTrigger_times', 'errorCueTrigger_times', 'itiIn_times',
            'position', 'contrast', 'quiescence', 'phase', 'probabilityLeft', 'contrastRight',
            'contrastLeft'
        ]
        self.assertCountEqual(expected, data.keys())

    def test_download_data(self):
        """Test behavior when ensure_data flag is True
        """
        path = self.one.path_from_eid(self.eid_incomplete)  # FIXME Returns None
        ex = TaskQCExtractor(path, lazy=True, one=self.one, ensure_data=True)
        self.assertTrue(ex.lazy, 'Failed to set lazy flag')

        shutil.rmtree(self.session_path)  # Remove downloaded data
        assert self.session_path.exists() is False, 'Failed to remove session folder'
        TaskQCExtractor(self.session_path, lazy=True, one=self.one, ensure_data=True)
        files = list(self.session_path.rglob('*.*'))
        expected = 6  # NB This session is missing raw ephys data and missing some datasets
        self.assertEqual(len(files), expected)


if __name__ == "__main__":
    unittest.main(exit=False)

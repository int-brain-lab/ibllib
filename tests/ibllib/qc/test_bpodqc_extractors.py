# Mock dataset
import unittest

import numpy as np

from ibllib.qc.bpodqc_extractors import BpodQCExtractor
from oneibl.one import ONE

one = ONE(
    base_url="https://test.alyx.internationalbrainlab.org",
    username="test_user",
    password="TapetesBloc18",
)

dstypes = [
    "_iblrig_taskData.raw",
    "_iblrig_taskSettings.raw",
    "_iblrig_encoderPositions.raw",
    "_iblrig_encoderEvents.raw",
    "_iblrig_stimPositionScreen.raw",
    "_iblrig_syncSquareUpdate.raw",
    "_iblrig_encoderTrialInfo.raw",
    "_iblrig_ambientSensorData.raw",
]


class TestBpodQCExtractors(unittest.TestCase):
    """TODO: should be an integration test in iblscripts"""

    def setUp(self):
        self.one = one
        self.eid = "b1c968ad-4874-468d-b2e4-5ffa9b9964e9"
        # Make sure the data exists locally
        self.one.load(self.eid, dataset_types=dstypes, download_only=True)
        self.session_path = self.one.path_from_eid(self.eid)

    def _BpodQCExtractor_load_lazy(self):
        # Wrong path
        with self.assertRaises(TypeError):
            BpodQCExtractor('/random/path')
        # Should load raw_data, details, BNC1, BNC2 and wheel_data not trial_data
        self.extractor = BpodQCExtractor(self.session_path, lazy=True)
        with self.assertRaises(AttributeError):
            self.extractor.trial_data
        self.assertTrue(np.all(np.isnan(self.extractor.BNC1['times'])))
        self.assertTrue(all(np.isnan(self.extractor.BNC2['times'])))
        self.assertTrue(self.extractor.details is not None)
        self.assertTrue(self.extractor.raw_data is not None)
        self.assertTrue(isinstance(self.extractor.wheel_data, dict))

    def _BpodQCExtractor_extract(self):
        self.extractor.extract_trial_data()
        self.assertTrue(isinstance(self.extractor.trial_data, dict))

    def _BpodQCExtractor_load_extract(self):
        # Test lazy
        extractor = BpodQCExtractor(self.session_path, lazy=False)
        self.assertTrue(extractor is not None)

    def test_object(self):
        self._BpodQCExtractor_load_lazy()
        self._BpodQCExtractor_extract()
        self._BpodQCExtractor_load_extract()


if __name__ == "__main__":
    unittest.main(exit=False)

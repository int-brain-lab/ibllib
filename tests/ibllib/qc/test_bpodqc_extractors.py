# Mock dataset
import unittest

import numpy as np

import ibllib.io.raw_data_loaders as raw
from ibllib.qc.bpodqc_extractors import BpodQCExtractor, extract_bpod_trial_data
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
        self.biased_eid = ""
        self.training_eid = "8dd0fcb0-1151-4c97-ae35-2e2421695ad7"
        # Make sure the data exists locally
        self.one.load(self.eid, dataset_types=dstypes, download_only=True)
        # self.one.load(self.biased_eid, dataset_types=dstypes, download_only=True)
        self.one.load(self.training_eid, dataset_types=dstypes, download_only=True)
        self.session_path = self.one.path_from_eid(self.eid)
        self.biased_session_path = self.one.path_from_eid(self.biased_eid)
        self.training_session_path = self.one.path_from_eid(self.training_eid)

    def _BpodQCExtractor_load_lazy(self):
        # Wrong path
        with self.assertRaises(TypeError):
            BpodQCExtractor("/random/path")
        # Should load raw_data, details, BNC1, BNC2 and wheel_data not trial_data
        self.extractor = BpodQCExtractor(self.session_path, lazy=True)
        self.training_extractor = BpodQCExtractor(self.training_session_path, lazy=True)
        with self.assertRaises(AttributeError):
            self.extractor.trial_data
        with self.assertRaises(AttributeError):
            self.training_extractor.trial_data
        self.assertTrue(np.all(np.isnan(self.extractor.BNC1["times"])))
        self.assertFalse(np.all(np.isnan(self.training_extractor.BNC1["times"])))

        self.assertTrue(all(np.isnan(self.extractor.BNC2["times"])))
        self.assertFalse(all(np.isnan(self.training_extractor.BNC2["times"])))

        self.assertTrue(self.extractor.details is not None)
        self.assertTrue(self.training_extractor.details is not None)

        self.assertTrue(self.extractor.raw_data is not None)
        self.assertTrue(self.training_extractor.raw_data is not None)

        self.assertTrue(isinstance(self.extractor.wheel_data, dict))
        self.assertTrue(isinstance(self.training_extractor.wheel_data, dict))

    def _BpodQCExtractor_extract(self):
        self.extractor = BpodQCExtractor(self.session_path, lazy=True)
        self.training_extractor = BpodQCExtractor(self.training_session_path, lazy=True)
        self.extractor.extract_trial_data()
        self.training_extractor.extract_trial_data()
        self.assertTrue(isinstance(self.extractor.trial_data, dict))
        self.assertTrue(isinstance(self.training_extractor.trial_data, dict))

    def _BpodQCExtractor_load_extract(self):
        # Test lazy
        extractor = BpodQCExtractor(self.session_path, lazy=False)
        training_extractor = BpodQCExtractor(self.training_session_path, lazy=False)
        self.assertTrue(extractor is not None)
        self.assertTrue(training_extractor is not None)

    def test_object(self):
        self._BpodQCExtractor_load_lazy()
        self._BpodQCExtractor_extract()
        self._BpodQCExtractor_load_extract()

    def test_extract_bpod_trial_data(self):
        raw_data = raw.load_data(self.session_path)
        raw_settings = raw.load_settings(self.session_path)
        trials_table = extract_bpod_trial_data(
            self.session_path, raw_bpod_trials=raw_data, raw_settings=raw_settings
        )
        bla = np.array([len(v) for k, v in trials_table.items()])
        self.assertTrue(np.all(bla == bla[0]))


if __name__ == "__main__":
    unittest.main(exit=False)

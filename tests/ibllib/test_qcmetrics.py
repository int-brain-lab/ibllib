# Mock dataset
import unittest

import numpy as np

from ibllib.ephys import qcmetrics


class TestBpodTask(unittest.TestCase):
    def setUp(self):
        self.data = self.fake_bpod_data()
        # random eid will not be used if data is passed
        self.eid = '7be8fec4-406b-4e74-8548-d2885dcc3d5e'

    def fake_bpod_data(self):
        """Create fake extractor output of bpodqc.load_data"""
        data = {
            "position": None,
            "contrast": None,
            "quiescence": None,
            "phase": None,
            "prob_left": None,
            "choice": None,
            "feedbackType": None,
            "correct": None,
            "outcome": None,
            "intervals": None,
            "stimOnTrigger_times": None,
            "stimOn_times": None,
            "stimOn_times_training": None,
            "stimOffTrigger_times": None,
            "stimOff_times": None,
            "stimOff_times_from_state": None,
            "stimFreezeTrigger_times": None,
            "stimFreeze_times": None,
            "goCueTrigger_times": None,
            "goCue_times": None,
            "errorCueTrigger_times": None,
            "errorCue_times": None,
            "valveOpen_times": None,
            "response_times": None,
            "feedback_times": None,
            "itiIn_times": None,
            "intervals_0": None,
            "intervals_1": None,
        }
        return data

    def test_load_stimon_gocue_delays(self):
        metric = qcmetrics.load_stimon_gocue_delays(eid, data=self.data, pass_crit=False)
        # Tests of metric


if __name__ == "__main__":
    unittest.main(exit=False)

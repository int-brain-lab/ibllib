# Mock dataset
import unittest
from functools import partial
import numpy as np

from ibllib.qc import bpodqc_metrics as qcmetrics


class TestBpodQCMetrics(unittest.TestCase):
    def setUp(self):
        self.load_fake_bpod_data()
        # random eid will not be used if data is passed
        self.eid = "7be8fec4-406b-4e74-8548-d2885dcc3d5e"
        self.data = self.load_fake_bpod_data()
    @staticmethod
    def load_fake_bpod_data():
        """Create fake extractor output of bpodqc.load_data"""
        n = 5  # number of trials
        trigg_delay = 1e-4  # an ideal delay between triggers and measured times
        resp_feeback_delay = 1e-3  # delay between feedback and response
        N = partial(np.random.normal, (n,))  # Convenience function for norm dist sampling

        choice = np.ones((n,), dtype=int)
        choice[[1, 3]] = -1  # a couple of incorrect trials
        choice[0] = 0  # a nogo trial
        # One trial of each type incorrect
        correct = choice != 0
        correct[np.argmax(choice == 1)] = 0
        correct[np.argmax(choice == -1)] = 0

        quiescence_length = 0.2 + np.random.standard_exponential(size=(n,))
        iti_length = 0.5  # inter-trial interval
        # trial lengths include quiescence period, a couple small trigger delays and iti
        trial_lengths = quiescence_length + resp_feeback_delay + 1e-1 + iti_length
        # add on 60s for nogos + feedback time (1 or 2s) + ~0.5s for other responses
        trial_lengths += (choice == 0) * 60 + (~correct + 1) + (choice != 0) * N(0.5)
        start_times = np.concatenate(([0], np.cumsum(trial_lengths)[:-1]))
        end_times = np.cumsum(trial_lengths) - 1e-2

        data = {
            "phase": np.random.uniform(low=0, high=2 * np.pi, size=(n,)),
            "quiescence": start_times + quiescence_length,
            "choice": choice,
            "correct": correct,
            "intervals_0": start_times,
            "intervals_1": end_times,
            "itiIn_times": end_times - iti_length,
        }

        data["stimOnTrigger_times"] = data["quiescence"] + 1e-4
        data["stimOn_times"] = data["stimOnTrigger_times"] + 1e-1
        data["goCueTrigger_times"] = data["stimOn_times"] + 1e-3
        data["goCue_times"] = data["goCueTrigger_times"] + trigg_delay

        # data["goCueTrigger_times"] = data["quiescence"] + 1e-3
        # data["goCue_times"] = data["goCueTrigger_times"] + trigg_delay
        # data['stimOn_times'] = data['goCue_times'] + 1e-3
        # data['stimOn_times_training'] = data['stimOn_times']
        # data['stimOnTrigger_times'] = data['stimOn_times'] - trigg_delay
        data["response_times"] = end_times - (
            resp_feeback_delay + 1e-1 + iti_length + (~correct + 1)
        )
        data["feedback_times"] = data["response_times"] + resp_feeback_delay
        data["stimFreeze_times"] = data["response_times"] + 1e-2
        data["stimFreezeTrigger_times"] = data["stimFreeze_times"] - trigg_delay
        data["feedbackType"] = np.vectorize(lambda x: -1 if x == 0 else x)(data["correct"])
        outcome = data["feedbackType"].copy()
        outcome[data["choice"] == 0] = 0
        data["outcome"] = outcome
        # Delay of 1 second if correct, 2 seconds if incorrect
        data["stimOffTrigger_times"] = data["feedback_times"] + (~correct + 1)
        data["stimOff_times"] = data["stimOffTrigger_times"] + trigg_delay
        # Error tone times nan on incorrect trials
        outcome_times = np.vectorize(lambda x, y: x + 1e-2 if y else np.nan)
        data["errorCueTrigger_times"] = outcome_times(data["feedback_times"], ~data["correct"])
        data["errorCue_times"] = data["errorCueTrigger_times"] + trigg_delay
        data["valveOpen_times"] = outcome_times(data["feedback_times"], data["correct"])
        data["rewardVolume"] = ~np.isnan(data["valveOpen_times"]) * 3.0

        # data = {
        #     "position": None,
        #     "contrast": None,
        #     "prob_left": None,
        #     "intervals": None,
        #     "stimOff_times_from_state": None,
        # }
        #  trial_length = 1  # length of trial
        return data

    def test_load_stimOn_goCue_delays(self):
        metric = qcmetrics.load_stimOn_goCue_delays(self.eid, data=self.data, apply_criteria=False)
        self.assertTrue(np.allclose(metric, 0.0011), "failed to return correct metric")
        # Set incorrect timestamp (goCue occurs before stimOn)
        self.data["goCue_times"][-1] = self.data["stimOn_times"][-1] - 1e-4
        passed = qcmetrics.load_stimOn_goCue_delays(self.eid, data=self.data, apply_criteria=True)
        n = len(self.data["stimOn_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_response_feedback_delays(self):
        metric = qcmetrics.load_response_feedback_delays(
            self.eid, data=self.data, apply_criteria=False
        )
        self.assertTrue(np.allclose(metric, 0.001), "failed to return correct metric")
        # Set incorrect timestamp (feedback occurs before response)
        self.data["feedback_times"][-1] = self.data["response_times"][-1] - 1e-4
        passed = qcmetrics.load_response_feedback_delays(
            self.eid, data=self.data, apply_criteria=True
        )
        n = len(self.data["feedback_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_response_stimFreeze_delays(self):
        metric = qcmetrics.load_response_stimFreeze_delays(
            self.eid, data=self.data, apply_criteria=False
        )
        self.assertTrue(np.allclose(metric, 1e-2), "failed to return correct metric")
        # Set incorrect timestamp (stimFreeze occurs before response)
        self.data["stimFreeze_times"][-1] = self.data["response_times"][-1] - 1e-4
        passed = qcmetrics.load_response_stimFreeze_delays(
            self.eid, data=self.data, apply_criteria=True
        )
        n = len(self.data["feedback_times"]) - np.sum(self.data["choice"] == 0)
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    @unittest.skip("not implemented")
    def test_load_wheel_move_during_closed_loop(self):
        pass

    @unittest.skip("not implemented")
    def test_load_stimulus_move_before_goCue(self):
        pass

    def test_load_positive_feedback_stimOff_delays(self):
        metric = qcmetrics.load_positive_feedback_stimOff_delays(
            self.eid, data=self.data, apply_criteria=False
        )
        self.assertTrue(
            np.allclose(metric[self.data["correct"]], 1e-4), "failed to return correct metric"
        )
        # Set incorrect timestamp (stimOff occurs just after response)
        id = np.argmax(self.data["correct"])
        self.data["stimOff_times"][id] = self.data["response_times"][id] + 1e-2
        passed = qcmetrics.load_positive_feedback_stimOff_delays(
            self.eid, data=self.data, apply_criteria=True
        )
        expected = (self.data["correct"].sum() - 1) / self.data["correct"].sum()
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_negative_feedback_stimOff_delays(self):
        err_trial = ~self.data["correct"] & self.data["outcome"] != 0
        metric = qcmetrics.load_negative_feedback_stimOff_delays(
            self.eid, data=self.data, apply_criteria=False
        )
        self.assertTrue(np.allclose(metric[err_trial], 1e-2), "failed to return correct metric")
        # Set incorrect timestamp (stimOff occurs 1s after response)
        id = np.argmax(err_trial)
        self.data["stimOff_times"][id] = self.data["response_times"][id] + 1
        passed = qcmetrics.load_negative_feedback_stimOff_delays(
            self.eid, data=self.data, apply_criteria=True
        )
        expected = (err_trial.sum() - 1) / err_trial.sum()
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")  # TODO verify

    def test_load_valve_pre_trial(self):
        correct = self.data["correct"]
        metric = qcmetrics.load_valve_pre_trial(self.eid, data=self.data, apply_criteria=False)
        self.assertTrue(np.all(metric), "failed to return correct metric")
        # Set incorrect timestamp (valveOpen_times occurs before goCue)
        id = np.argmax(correct)
        self.data["valveOpen_times"][id] = self.data["goCue_times"][id] - 0.021
        passed = qcmetrics.load_valve_pre_trial(self.eid, data=self.data, apply_criteria=True)
        expected = (correct.sum() - 1) / correct.sum()
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    @unittest.skip("not implemented")
    def test_load_audio_pre_trial(self):
        pass

    def test_load_error_trial_event_sequence(self):
        metric = qcmetrics.load_error_trial_event_sequence(
            self.eid, data=self.data, apply_criteria=False
        )
        self.assertTrue(np.all(metric), "failed to return correct metric")
        # Set incorrect timestamp (itiIn occurs before errorCue)
        err_trial = ~self.data["correct"]
        (id,) = np.where(err_trial)
        self.data["intervals_0"][id[0]] = np.inf
        self.data["errorCue_times"][id[1]] = 0
        passed = qcmetrics.load_error_trial_event_sequence(
            self.eid, data=self.data, apply_criteria=True
        )
        expected = (err_trial.sum() - 2) / err_trial.sum()
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_correct_trial_event_sequence(self):
        metric = qcmetrics.load_correct_trial_event_sequence(
            self.eid, data=self.data, apply_criteria=False
        )
        self.assertTrue(np.all(metric), "failed to return correct metric")
        # Set incorrect timestamp
        correct = self.data["correct"]
        id = np.argmax(correct)
        self.data["intervals_0"][id] = np.inf
        passed = qcmetrics.load_correct_trial_event_sequence(
            self.eid, data=self.data, apply_criteria=True
        )
        expected = (correct.sum() - 1) / correct.sum()
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_trial_length(self):
        metric = qcmetrics.load_trial_length(self.eid, data=self.data, apply_criteria=False)
        self.assertTrue(np.all(metric), "failed to return correct metric")
        # Set incorrect timestamp
        self.data["goCue_times"][-1] = 0
        passed = qcmetrics.load_trial_length(self.eid, data=self.data, apply_criteria=True)
        n = len(self.data["goCue_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_goCue_delays(self):
        metric = qcmetrics.load_goCue_delays(self.eid, data=self.data, apply_criteria=False)
        self.assertTrue(np.allclose(metric, 1e-4), "failed to return correct metric")
        # Set incorrect timestamp
        self.data["goCue_times"][1] = self.data["goCueTrigger_times"][1] + 0.1
        passed = qcmetrics.load_goCue_delays(self.eid, data=self.data, apply_criteria=True)
        n = len(self.data["goCue_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_errorCue_delays(self):
        metric = qcmetrics.load_errorCue_delays(self.eid, data=self.data, apply_criteria=False)
        err_trial = ~self.data["correct"]
        self.assertTrue(np.allclose(metric[err_trial], 1e-4), "failed to return correct metric")
        # Set incorrect timestamp
        id = np.argmax(err_trial)
        self.data["errorCue_times"][id] = self.data["errorCueTrigger_times"][id] + 0.1
        passed = qcmetrics.load_errorCue_delays(self.eid, data=self.data, apply_criteria=True)
        n = err_trial.sum()
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_stimOn_delays(self):
        metric = qcmetrics.load_stimOn_delays(self.eid, data=self.data, apply_criteria=False)
        self.assertTrue(np.allclose(metric, 1e-1), "failed to return correct metric")
        # Set incorrect timestamp
        self.data["stimOn_times"][-1] = self.data["stimOnTrigger_times"][-1] + 0.2
        passed = qcmetrics.load_stimOn_delays(self.eid, data=self.data, apply_criteria=True)
        n = len(self.data["stimOn_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_stimOff_delays(self):
        metric = qcmetrics.load_stimOff_delays(self.eid, data=self.data, apply_criteria=False)
        self.assertTrue(np.allclose(metric, 1e-4), "failed to return correct metric")
        # Set incorrect timestamp
        self.data["stimOff_times"][-1] = self.data["stimOffTrigger_times"][-1] + 0.2
        passed = qcmetrics.load_stimOff_delays(self.eid, data=self.data, apply_criteria=True)
        n = len(self.data["stimOff_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_stimFreeze_delays(self):
        metric = qcmetrics.load_stimFreeze_delays(self.eid, data=self.data, apply_criteria=False)
        self.assertTrue(np.allclose(metric, 1e-4), "failed to return correct metric")
        # Set incorrect timestamp
        self.data["stimFreeze_times"][-1] = self.data["stimFreezeTrigger_times"][-1] + 0.2
        passed = qcmetrics.load_stimFreeze_delays(self.eid, data=self.data, apply_criteria=True)
        n = len(self.data["stimFreeze_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_reward_volumes(self):
        metric = qcmetrics.load_reward_volumes(self.eid, data=self.data, apply_criteria=False)
        self.assertTrue(np.all([x in {0.0, 3.0} for x in metric]), "failed to return correct metric")
        # Set incorrect volume
        id = np.argmax(self.data["correct"])
        self.data["rewardVolume"][id] = 4.0
        passed = qcmetrics.load_reward_volumes(self.eid, data=self.data, apply_criteria=True)
        self.assertTrue(np.nanmean(passed) == 0.2, "failed to detect incorrect reward volume")


if __name__ == "__main__":
    unittest.main(exit=False)

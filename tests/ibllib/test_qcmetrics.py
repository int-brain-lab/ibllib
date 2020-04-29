# Mock dataset
import unittest
import random
from functools import partial

import numpy as np

from ibllib.ephys import qcmetrics


class TestBpodTask(unittest.TestCase):
    def setUp(self):
        self.load_fake_bpod_data()
        # random eid will not be used if data is passed
        self.eid = '7be8fec4-406b-4e74-8548-d2885dcc3d5e'

    def load_fake_bpod_data(self):
        """Create fake extractor output of bpodqc.load_data"""
        n = 5  # number of trials
        duration = 3 * 60  # 3 min block
        trigg_delay = 1e-4  # an ideal delay between triggers and measured times
        N = partial(np.random.normal, (n,))  # Convenience function for norm dist sampling
        start_times = np.sort(random.sample(set(np.arange(duration)), n) +
                              np.random.random(size=(n,)))
        assert np.all(np.diff(start_times) > 3)  # FIXME Fails on occasion
        data = {
            "phase": np.random.uniform(low=0, high=2 * np.pi, size=(n,)),
            'goCue_times': start_times + N(2e-3)
        }
        data['stimOn_times'] = data['goCue_times'] + 1e-3
        data['stimOn_times_training'] = data['stimOn_times']
        data['stimOnTrigger_times'] = data['stimOn_times'] - trigg_delay
        data['response_times'] = data['goCue_times'] + N(.5)
        data['feedback_times'] = data['response_times'] + 1e-3
        data['stimFreeze_times'] = data['response_times'] + 1e-2
        data['stimFreezeTrigger_times'] = data['stimFreeze_times'] - trigg_delay
        choice = np.ones((n,), dtype=int)
        choice[[1,3]] = -1  # a couple of incorrect trials
        choice[0] = 0  # a nogo trial
        data['choice'] = choice
        # One trial of each type incorrect
        correct = choice != 0
        correct[np.argmax(choice == 1)] = 0
        correct[np.argmax(choice == -1)] = 0
        data['correct'] = correct
        data['feedbackType'] = np.vectorize(lambda x: -1 if x == 0 else x)(data['correct'])
        outcome = data['feedbackType']
        outcome[data['choice'] == 0] = 0
        data['outcome'] = outcome
        # Delay of 1 second if correct, 2 seconds if incorrect
        data['stimOffTrigger_times'] = data['feedback_times'] + (~correct + 1)
        data['stimOff_times'] = data['stimOffTrigger_times'] + trigg_delay
        # Error tone times nan on incorrect trials FIXME Verify 
        outcome_times = np.vectorize(lambda x,y: x + 1e-2 if y else np.nan)
        data['errorCueTrigger_times'] = outcome_times(data['feedback_times'], ~data['correct'])
        data['errorCue_times'] = data['errorCueTrigger_times'] + trigg_delay
        data['valveOpen_times'] = outcome_times(data['feedback_times'], data['correct'])
        # FIXME interval end before iti?
        data['intervals_1'] = data['stimOff_times'] + 1e-1
        data['itiIn_times'] = data['intervals_1'] + .8
        data['intervals_0'] = data['startTimes']

        # data = {
        #     "position": None,
        #     "contrast": None,
        #     "quiescence": None,
        #     "prob_left": None,
        #     "intervals": None,
        #     "stimOff_times_from_state": None,
        # }
        #  trial_length = 1  # length of trial
        self.data = data

    def test_load_stimon_gocue_delays(self):
        metric = qcmetrics.load_stimon_gocue_delays(self.eid, data=self.data, pass_crit=False)
        self.assertTrue(np.allclose(metric, 0.001), 'failed to return correct metric')
        # Set incorrect timestamp (stimOn occurs before goCue)
        self.data['stimOn_times'][-1] = self.data['goCue_times'][-1] - 1e-4
        passed = qcmetrics.load_stimon_gocue_delays(self.eid, data=self.data, pass_crit=True)
        n = len(self.data['stimOn_times'])
        expected = (n - 1) / n
        self.assertEqual(passed, expected, 'failed to detect dodgy timestamp')

    def test_load_response_feedback_delays(self):
        metric = qcmetrics.load_response_feedback_delays(self.eid, data=self.data, pass_crit=False)
        self.assertTrue(np.allclose(metric, 0.001), 'failed to return correct metric')
        # Set incorrect timestamp (feedback occurs before response)
        self.data['feedback_times'][-1] = self.data['response_times'][-1] - 1e-4
        passed = qcmetrics.load_response_feedback_delays(self.eid, data=self.data, pass_crit=True)
        n = len(self.data['feedback_times'])
        expected = (n - 1) / n
        self.assertEqual(passed, expected, 'failed to detect dodgy timestamp')

    def test_load_response_stimFreeze_delays(self):
        metric = qcmetrics.load_response_stimFreeze_delays(self.eid, data=self.data, pass_crit=False)
        self.assertTrue(np.allclose(metric, 1e-2), 'failed to return correct metric')
        # Set incorrect timestamp (stimFreeze occurs before response)
        self.data['stimFreeze_times'][-1] = self.data['response_times'][-1] - 1e-4
        passed = qcmetrics.load_response_stimFreeze_delays(self.eid, data=self.data, pass_crit=True)
        n = len(self.data['feedback_times']) - np.sum(self.data['choice'] == 0)
        expected = (n - 1) / n
        self.assertEqual(passed, expected, 'failed to detect dodgy timestamp')

    @unittest.skip("not implemented")
    def test_load_wheel_move_during_closed_loop(self):
        pass

    @unittest.skip("not implemented")
    def test_load_stimulus_move_before_goCue(self):
        pass
    
    def test_load_positive_feedback_stimOff_delays(self):
        metric = qcmetrics.load_positive_feedback_stimOff_delays(self.eid, data=self.data, pass_crit=False)
        self.assertTrue(np.allclose(metric[self.data['correct']], 1e-4), 'failed to return correct metric')
        # Set incorrect timestamp (stimOff occurs just after response)
        id = np.argmax(self.data['correct'])
        self.data['stimOff_times'][id] = self.data['response_times'][id] + 1e-2
        passed = qcmetrics.load_positive_feedback_stimOff_delays(self.eid, data=self.data, pass_crit=True)
        expected = (self.data['correct'] - 1) / self.data['correct'].sum()
        self.assertEqual(passed, expected, 'failed to detect dodgy timestamp')
        
    def test_load_negative_feedback_stimOff_delays(self):
        err_trial = self.data['feedbackType'] == -1
        metric = qcmetrics.load_negative_feedback_stimOff_delays(self.eid, data=self.data, pass_crit=False)
        self.assertTrue(np.allclose(metric[err_trial], 1e-2), 'failed to return correct metric')
        # Set incorrect timestamp (stimOff occurs 1s after response)
        id = np.argmax(~self.data['correct'])
        self.data['stimOff_times'][id] = self.data['response_times'][id] + 1
        passed = qcmetrics.load_negative_feedback_stimOff_delays(self.eid, data=self.data, pass_crit=True)
        expected = (~err_trial - 1) / ~err_trial.sum()
        self.assertEqual(passed, expected, 'failed to detect dodgy timestamp')  # TODO verify
        
    def test_load_valve_pre_trial(self):
        correct = self.data['correct']
        metric = qcmetrics.load_valve_pre_trial(self.eid, data=self.data, pass_crit=False)
        self.assertTrue(np.all(metric[correct]), 'failed to return correct metric')
        # Set incorrect timestamp (valveOpen_times occurs before goCue)
        id = np.argmax(correct)
        self.data['valveOpen_times'][id] = self.data['goCue_times'][id] - 1e-3
        passed = qcmetrics.load_valve_pre_trial(self.eid, data=self.data, pass_crit=True)
        expected = (correct.sum() - 1) / correct.sum()
        self.assertEqual(passed, expected, 'failed to detect dodgy timestamp')  # TODO fails due to nans

    @unittest.skip("not implemented")
    def test_load_audio_pre_trial(self):
        pass

    def test_load_error_trial_event_sequence(self):
        pass # TODO intervals_0, itiIn_times


if __name__ == "__main__":
    unittest.main(exit=False)

import unittest
from functools import partial
from pathlib import Path

import numpy as np

from ibllib.qc.bpodqc_metrics import BpodQC
from ibllib.qc import bpodqc_metrics as qcmetrics
from ibllib.qc.oneutils import download_bpodqc_raw_data
from oneibl.one import ONE
from brainbox.behavior.wheel import cm_to_rad

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


class TestBpodQCMetricsObject(unittest.TestCase):
    def setUp(self):
        self.one = one
        self.eid = "b1c968ad-4874-468d-b2e4-5ffa9b9964e9"
        # Make sure the data exists locally
        self.one.load(self.eid, dataset_types=dstypes, download_only=True)
        self.session_path = self.one.path_from_eid(self.eid)

    def test_BpodQC_lazy(self):
        # Make sure the data exists locally
        self.one.load(self.eid, dataset_types=dstypes, download_only=True)
        # Make from eid
        bpodqc = BpodQC(self.eid, one=self.one, lazy=True)
        # Make from session_path
        bpodqc = BpodQC(self.session_path, one=self.one, lazy=True)
        # Load data
        bpodqc.load_data()
        self.assertTrue(bpodqc.extractor is not None)
        self.assertTrue(bpodqc.extractor.trial_data is not None)
        self.assertTrue(bpodqc.wheel_gain is not None)
        self.assertTrue(bpodqc.bpod_ntrials is not None)
        # Compute metrics
        self.assertTrue(bpodqc.metrics is None)
        self.assertTrue(bpodqc.passed is None)
        bpodqc.compute()
        self.assertTrue(bpodqc.metrics is not None)
        self.assertTrue(bpodqc.passed is not None)

    def test_BpodQC(self):
        # Make sure the data exists locally
        download_bpodqc_raw_data(self.eid, one=self.one)
        bpodqc = BpodQC(self.eid, one=self.one, lazy=False)
        self.assertTrue(bpodqc is not None)


class TestBpodQCMetrics(unittest.TestCase):
    def setUp(self):
        self.load_fake_bpod_data()
        # random eid will not be used if data is passed
        self.eid = "7be8fec4-406b-4e74-8548-d2885dcc3d5e"
        self.data = self.load_fake_bpod_data()
        self.wheel_gain = 4
        self.wheel = self.load_fake_wheel_data(self.data, wheel_gain=self.wheel_gain)

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
            "quiescence": quiescence_length,
            "choice": choice,
            "correct": correct,
            "intervals_0": start_times,
            "intervals_1": end_times,
            "itiIn_times": end_times - iti_length,
            "position": np.ones_like(choice) * 35
        }

        data["stimOnTrigger_times"] = start_times + data["quiescence"] + 1e-4
        data["stimOn_times"] = data["stimOnTrigger_times"] + 1e-1
        data["goCueTrigger_times"] = data["stimOn_times"] + 1e-3
        data["goCue_times"] = data["goCueTrigger_times"] + trigg_delay

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

        return data

    @staticmethod
    def load_fake_wheel_data(trial_data, wheel_gain=4):
        # Load a wheel fragment: a numpy array of the form [timestamps, positions], for a wheel
        # movement during one trial.  Wheel is X1 bpod RE in radians.
        wh_path = Path(__file__).parent.joinpath('..', 'fixtures', 'qc').resolve()
        wheel_frag = np.load(wh_path.joinpath('wheel.npy'))
        resolution = np.mean(np.abs(np.diff(wheel_frag[:, 1])))  # pos diff between samples
        # abs displacement, s, in mm required to move 35 visual degrees
        POS_THRESH = 35
        s_mm = np.abs(POS_THRESH / wheel_gain)  # don't care about direction
        # convert abs displacement to radians (wheel pos is in rad)
        pos_thresh = cm_to_rad(s_mm * 1e-1)
        # index of threshold cross
        pos_thresh_idx = np.argmax(np.abs(wheel_frag[:, 1]) > pos_thresh)

        def qt_wheel_fill(start, end, t_step=0.001, p_step=None):
            if p_step is None:
                p_step = 2 * np.pi / 1024
            t = np.arange(start, end, t_step)
            p = np.random.randint(-1, 2, len(t))
            t = t[p != 0]
            p = p[p != 0].cumsum() * p_step
            return t, p

        wheel_data = []  # List generated of wheel data fragments

        def add_frag(t, p):
            """Add wheel data fragments to list, adjusting positions to be within one sample of
            one another"""
            last_samp = getattr(add_frag, 'last_samp', (0, 0))
            p += last_samp[1]
            if np.abs(p[0] - last_samp[1]) == 0:
                p += resolution
            wheel_data.append((t, p))
            add_frag.last_samp = (t[-1], p[-1])

        for i in np.arange(len(trial_data['choice'])):
            # Iterate over trials generating wheel samples for the necessary periods
            # trial start to stim on; should be below quiescence threshold
            stimOn_trig = trial_data['stimOnTrigger_times'][i]
            trial_start = trial_data['intervals_0'][i]
            t, p = qt_wheel_fill(trial_start, stimOn_trig, .5, resolution)
            if len(t) > 0:  # Possible for no movement during quiescence
                add_frag(t, p)

            # stim on to trial end
            trial_end = trial_data['intervals_1'][i]
            if trial_data['choice'][i] == 0:
                # Add random wheel movements for duration of trial
                goCue = trial_data['goCue_times'][i]
                t, p = qt_wheel_fill(goCue, trial_end, .1, resolution)
                add_frag(t, p)
            else:
                # Align wheel fragment with response time
                response_time = trial_data['response_times'][i]
                t = wheel_frag[:, 0] + response_time - wheel_frag[pos_thresh_idx, 0]
                p = np.abs(wheel_frag[:, 1]) * trial_data['choice'][i]
                assert t[0] > add_frag.last_samp[0]
                add_frag(t, p)
                # Fill in random movements between end of response and trial end
                t, p = qt_wheel_fill(t[-1] + 0.01, trial_end, p_step=resolution)
                add_frag(t, p)

        # Stitch wheel fragments and assert no skips
        wheel_data = np.concatenate(list(map(np.column_stack, wheel_data)))
        assert np.all(np.diff(wheel_data[:, 0]) > 0), "timestamps don't strictly increase"
        np.testing.assert_allclose(np.abs(np.diff(wheel_data[:, 1])), resolution)
        return {
            're_ts': wheel_data[:, 0],
            're_pos': wheel_data[:, 1]
        }

    def test_load_stimOn_goCue_delays(self):
        metric, passed = qcmetrics.load_stimOn_goCue_delays(self.data)
        self.assertTrue(np.allclose(metric, 0.0011), "failed to return correct metric")
        # Set incorrect timestamp (goCue occurs before stimOn)
        self.data["goCue_times"][-1] = self.data["stimOn_times"][-1] - 1e-4
        metric, passed = qcmetrics.load_stimOn_goCue_delays(self.data)
        n = len(self.data["stimOn_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_response_feedback_delays(self):
        metric, passed = qcmetrics.load_response_feedback_delays(self.data)
        self.assertTrue(np.allclose(metric, 0.001), "failed to return correct metric")
        # Set incorrect timestamp (feedback occurs before response)
        self.data["feedback_times"][-1] = self.data["response_times"][-1] - 1e-4
        metric, passed = qcmetrics.load_response_feedback_delays(self.data)
        n = len(self.data["feedback_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_response_stimFreeze_delays(self):
        metric, passed = qcmetrics.load_response_stimFreeze_delays(self.data)
        self.assertTrue(np.allclose(metric, 1e-2), "failed to return correct metric")
        # Set incorrect timestamp (stimFreeze occurs before response)
        self.data["stimFreeze_times"][-1] = self.data["response_times"][-1] - 1e-4
        metric, passed = qcmetrics.load_response_stimFreeze_delays(self.data)
        n = len(self.data["feedback_times"]) - np.sum(self.data["choice"] == 0)
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_positive_feedback_stimOff_delays(self):
        metric, passed = qcmetrics.load_positive_feedback_stimOff_delays(self.data)
        self.assertTrue(
            np.allclose(metric[self.data["correct"]], 1e-4), "failed to return correct metric"
        )
        # Set incorrect timestamp (stimOff occurs just after response)
        id = np.argmax(self.data["correct"])
        self.data["stimOff_times"][id] = self.data["response_times"][id] + 1e-2
        metric, passed = qcmetrics.load_positive_feedback_stimOff_delays(self.data)
        expected = (self.data["correct"].sum() - 1) / self.data["correct"].sum()
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_negative_feedback_stimOff_delays(self):
        err_trial = ~self.data["correct"] & self.data["outcome"] != 0
        metric, passed = qcmetrics.load_negative_feedback_stimOff_delays(self.data)
        self.assertTrue(np.allclose(metric[err_trial], 1e-2), "failed to return correct metric")
        # Set incorrect timestamp (stimOff occurs 1s after response)
        id = np.argmax(err_trial)
        self.data["stimOff_times"][id] = self.data["response_times"][id] + 1
        metric, passed = qcmetrics.load_negative_feedback_stimOff_delays(self.data)
        expected = (err_trial.sum() - 1) / err_trial.sum()
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_valve_pre_trial(self):
        correct = self.data["correct"]
        metric, passed = qcmetrics.load_valve_pre_trial(self.data)
        self.assertTrue(np.all(metric), "failed to return correct metric")
        # Set incorrect timestamp (valveOpen_times occurs before goCue)
        idx = np.argmax(correct)
        self.data["valveOpen_times"][idx] = self.data["goCue_times"][idx] - 0.021
        metric, passed = qcmetrics.load_valve_pre_trial(self.data)
        expected = (correct.sum() - 1) / correct.sum()
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_error_trial_event_sequence(self):
        metric, passed = qcmetrics.load_error_trial_event_sequence(self.data)
        self.assertTrue(np.all(metric), "failed to return correct metric")
        # Set incorrect timestamp (itiIn occurs before errorCue)
        err_trial = ~self.data["correct"]
        (id,) = np.where(err_trial)
        self.data["intervals_0"][id[0]] = np.inf
        self.data["errorCue_times"][id[1]] = 0
        metric, passed = qcmetrics.load_error_trial_event_sequence(self.data)
        expected = (err_trial.sum() - 2) / err_trial.sum()
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_correct_trial_event_sequence(self):
        metric, passed = qcmetrics.load_correct_trial_event_sequence(self.data)
        self.assertTrue(np.all(metric), "failed to return correct metric")
        # Set incorrect timestamp
        correct = self.data["correct"]
        id = np.argmax(correct)
        self.data["intervals_0"][id] = np.inf
        metric, passed = qcmetrics.load_correct_trial_event_sequence(self.data)
        expected = (correct.sum() - 1) / correct.sum()
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_trial_length(self):
        metric, passed = qcmetrics.load_trial_length(self.data)
        self.assertTrue(np.all(metric), "failed to return correct metric")
        # Set incorrect timestamp
        self.data["goCue_times"][-1] = 0
        metric, passed = qcmetrics.load_trial_length(self.data)
        n = len(self.data["goCue_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_goCue_delays(self):
        metric, passed = qcmetrics.load_goCue_delays(self.data)
        self.assertTrue(np.allclose(metric, 1e-4), "failed to return correct metric")
        # Set incorrect timestamp
        self.data["goCue_times"][1] = self.data["goCueTrigger_times"][1] + 0.1
        metric, passed = qcmetrics.load_goCue_delays(self.data)
        n = len(self.data["goCue_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_errorCue_delays(self):
        metric, passed = qcmetrics.load_errorCue_delays(self.data)
        err_trial = ~self.data["correct"]
        self.assertTrue(np.allclose(metric[err_trial], 1e-4), "failed to return correct metric")
        # Set incorrect timestamp
        id = np.argmax(err_trial)
        self.data["errorCue_times"][id] = self.data["errorCueTrigger_times"][id] + 0.1
        metric, passed = qcmetrics.load_errorCue_delays(self.data)
        n = err_trial.sum()
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_stimOn_delays(self):
        metric, passed = qcmetrics.load_stimOn_delays(self.data)
        self.assertTrue(np.allclose(metric, 1e-1), "failed to return correct metric")
        # Set incorrect timestamp
        self.data["stimOn_times"][-1] = self.data["stimOnTrigger_times"][-1] + 0.2
        metric, passed = qcmetrics.load_stimOn_delays(self.data)
        n = len(self.data["stimOn_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_stimOff_delays(self):
        metric, passed = qcmetrics.load_stimOff_delays(self.data)
        self.assertTrue(np.allclose(metric, 1e-4), "failed to return correct metric")
        # Set incorrect timestamp
        self.data["stimOff_times"][-1] = self.data["stimOffTrigger_times"][-1] + 0.2
        metric, passed = qcmetrics.load_stimOff_delays(self.data)
        n = len(self.data["stimOff_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_stimFreeze_delays(self):
        metric, passed = qcmetrics.load_stimFreeze_delays(self.data)
        self.assertTrue(np.allclose(metric, 1e-4), "failed to return correct metric")
        # Set incorrect timestamp
        self.data["stimFreeze_times"][-1] = self.data["stimFreezeTrigger_times"][-1] + 0.2
        metric, passed = qcmetrics.load_stimFreeze_delays(self.data)
        n = len(self.data["stimFreeze_times"])
        expected = (n - 1) / n
        self.assertEqual(np.nanmean(passed), expected, "failed to detect dodgy timestamp")

    def test_load_reward_volumes(self):
        metric, passed = qcmetrics.load_reward_volumes(self.data)
        self.assertTrue(
            np.all([x in {0.0, 3.0} for x in metric]), "failed to return correct metric"
        )
        # Set incorrect volume
        id = np.argmax(self.data["correct"])
        self.data["rewardVolume"][id] = 4.0
        metric, passed = qcmetrics.load_reward_volumes(self.data)
        self.assertTrue(np.nanmean(passed) == 0.2, "failed to detect incorrect reward volume")

    def test_load_audio_pre_trial(self):
        # Create Sound sync fake data that is OK
        BNC2_OK = {
            "times": self.data["goCue_times"] + 1e-1,
            "polarities": np.array([1, -1, 1, -1, 1]),
        }
        # Create Sound sync fake data that is NOT OK
        BNC2_NOK = {
            "times": self.data["goCue_times"] - 1e-1,
            "polarities": np.array([1, -1, 1, -1, 1]),
        }
        metric, passed = qcmetrics.load_audio_pre_trial(self.data, BNC2=BNC2_OK)
        self.assertTrue(~np.all(metric))
        self.assertTrue(np.all(passed))
        metric, passed = qcmetrics.load_audio_pre_trial(self.data, BNC2=BNC2_NOK)
        self.assertTrue(np.all(metric))
        self.assertTrue(~np.all(passed))

    def test_load_wheel_freeze_during_quiescence(self):
        metric, passed = qcmetrics.load_wheel_freeze_during_quiescence(self.data, self.wheel)
        self.assertTrue(np.all(passed))

        # Make one trial move more
        n = 1  # Index of trial to manipulate
        t1 = self.data['intervals_0'][n]
        t2 = self.data['stimOnTrigger_times'][n]
        ts, pos = self.wheel.values()
        wh_idx = np.argmax(ts > t1)
        if ts[wh_idx] > self.data['stimOnTrigger_times'][n]:
            # No sample during quiescence; insert one
            self.wheel['re_ts'] = np.insert(ts, wh_idx, t2 - .001)
            self.wheel['re_pos'] = np.insert(pos, wh_idx, np.inf)
        else:  # Otherwise make one sample infinite
            self.wheel['re_pos'][wh_idx] = np.inf
        metric, passed = qcmetrics.load_wheel_freeze_during_quiescence(self.data, self.wheel)
        self.assertFalse(passed[n])
        self.assertTrue(metric[n] > 2)

    def test_load_wheel_move_before_feedback(self):
        metric, passed = qcmetrics.load_wheel_move_before_feedback(self.data, self.wheel)
        nogo = self.data['choice'] == 0
        self.assertTrue(np.all(passed[~nogo]))
        self.assertTrue(np.isnan(metric[nogo]).all())
        self.assertTrue(np.isnan(passed[nogo]).all())

        # Remove wheel data around feedback for choice trial
        assert self.data['choice'].any(), 'no choice trials in test data'
        n = np.argmax(self.data['choice'] != 0)  # Index of choice trial
        mask = np.logical_xor(self.wheel['re_ts'] > self.data['feedback_times'][n] - 1,
                              self.wheel['re_ts'] < self.data['feedback_times'][n] + 1)
        self.wheel['re_ts'] = self.wheel['re_ts'][mask]
        self.wheel['re_pos'] = self.wheel['re_pos'][mask]

        metric, passed = qcmetrics.load_wheel_move_before_feedback(self.data, self.wheel)
        self.assertFalse(passed[n] or metric[n] != 0)

    def test_load_wheel_move_during_closed_loop(self):
        gain = self.wheel_gain or 4
        metric, passed = qcmetrics.load_wheel_move_during_closed_loop(self.data, self.wheel, gain)
        nogo = self.data['choice'] == 0
        self.assertTrue(np.all(passed[~nogo]))
        self.assertTrue(np.isnan(metric[nogo]).all())
        self.assertTrue(np.isnan(passed[nogo]).all())

        # Remove wheel data for choice trial
        assert self.data['choice'].any(), 'no choice trials in test data'
        n = np.argmax(self.data['choice'] != 0)  # Index of choice trial
        mask = np.logical_xor(self.wheel['re_ts'] < self.data['goCue_times'][n],
                              self.wheel['re_ts'] > self.data['response_times'][n])
        self.wheel['re_ts'] = self.wheel['re_ts'][mask]
        self.wheel['re_pos'] = self.wheel['re_pos'][mask]

        metric, passed = qcmetrics.load_wheel_move_during_closed_loop(self.data, self.wheel, gain)
        self.assertFalse(passed[n])

    def test_load_wheel_integrity(self):
        metric, passed = qcmetrics.load_wheel_integrity(self.wheel, re_encoding='X1')
        self.assertTrue(np.all(passed))

        # Insert some violations and verify that they're caught
        idx = np.random.randint(self.wheel['re_ts'].size, size=2)
        self.wheel['re_ts'][idx[0] + 1] -= 1
        self.wheel['re_pos'][idx[1]] -= 1

        metric, passed = qcmetrics.load_wheel_integrity(self.wheel, re_encoding='X1')
        self.assertFalse(passed[idx].any())

    @unittest.skip("not implemented")
    def test_load_stimulus_move_before_goCue(self):
        # TODO Nicco?
        pass


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

from pathlib import Path
import unittest
from unittest import mock
import numpy as np
import pickle
import copy

from iblutil.util import Bunch
from one.api import ONE

import brainbox.behavior.wheel as wheel
import brainbox.behavior.training as train
from ibllib.tests import TEST_DB


class TestWheel(unittest.TestCase):

    def setUp(self):
        """ Load pickled test data
        Test data is in the form ((inputs), (outputs)) where inputs is a tuple containing a
        numpy array of timestamps and one of positions; outputs is a tuple of outputs from
        the function under test, i.e. wheel.movements

        The first set - test_data[0] - comes from Rigbox MATLAB and contains around 200
        seconds of (reasonably) evenly sampled wheel data from a 1024 ppr device with X4
        encoding, in raw samples.  test_data[0] = ((t, pos), (onsets, offsets, amps, peak_vel))

        The second set - test_data[1] - comes from ibllib FPGA and contains around 180 seconds
        of unevenly sampled wheel data from a 1024 ppr device with X2 encoding, in linear cm units.
        test_data[1] = ((t, pos), (onsets, offsets, amps, peak_vel))
        """
        pickle_file = Path(__file__).parent.joinpath('fixtures', 'wheel_test.p')
        if not pickle_file.exists():
            self.test_data = None
        else:
            with open(pickle_file, 'rb') as f:
                self.test_data = pickle.load(f)

        # Trial timestamps for trial_data[0]
        self.trials = {
            'stimOn_times': np.array([0.2, 75, 100, 120, 164]),
            'feedback_times': np.array([60.2, 85, 103, 130, 188]),
            'intervals': np.array([[0, 62], [63, 90], [95, 110], [115, 135], [140, 200]])
        }

    def test_velocity_filtered(self):
        """Test for brainbox.behavior.wheel.velocity_filtered"""
        Fs = 1000
        pos, _ = wheel.interpolate_position(*self.test_data[1][0], freq=Fs)
        vel, acc = wheel.velocity_filtered(pos, Fs)
        self.assertEqual(vel.shape, pos.shape)
        expected = [-0.03020161, -0.02642356, -0.0229635, -0.01981592, -0.01697264,
                    -0.01442305, -0.01215438, -0.01015202, -0.00839981, -0.00688036]
        np.testing.assert_array_almost_equal(vel[-10:], expected)
        expected = [0., 187.41222339, 4.16291917, 3.94583813, 3.67112556,
                    3.33635025, 2.94002541, 2.48170905, 1.96209209, 1.38307198]
        np.testing.assert_array_almost_equal(acc[:10], expected)

    def test_movements(self):
        # These test data are the same as those used in the MATLAB code
        inputs = self.test_data[0][0]
        expected = self.test_data[0][1]
        on, off, amp, peak_vel = wheel.movements(
            *inputs, freq=1000, pos_thresh=8, pos_thresh_onset=1.5)
        self.assertTrue(np.array_equal(on, expected[0]), msg='Unexpected onsets')
        self.assertTrue(np.array_equal(off, expected[1]), msg='Unexpected offsets')
        self.assertTrue(np.array_equal(amp, expected[2]), msg='Unexpected move amps')
        # Differences due to convolution algorithm
        all_close = np.allclose(peak_vel, expected[3], atol=1.e-2)
        self.assertTrue(all_close, msg='Unexpected peak velocities')

    def test_movements_FPGA(self):
        # These test data are the same as those used in the MATLAB code.  Test data are from
        # extracted FPGA wheel data
        pos, t = wheel.interpolate_position(*self.test_data[1][0], freq=1000)
        expected = self.test_data[1][1]
        thresholds = wheel.samples_to_cm(np.array([8, 1.5]))
        on, off, amp, peak_vel = wheel.movements(
            t, pos, freq=1000, pos_thresh=thresholds[0], pos_thresh_onset=thresholds[1])
        self.assertTrue(np.allclose(on, expected[0], atol=1.e-5), msg='Unexpected onsets')
        self.assertTrue(np.allclose(off, expected[1], atol=1.e-5), msg='Unexpected offsets')
        self.assertTrue(np.allclose(amp, expected[2], atol=1.e-5), msg='Unexpected move amps')
        self.assertTrue(np.allclose(peak_vel, expected[3], atol=1.e-2),
                        msg='Unexpected peak velocities')

    def test_traces_by_trial(self):
        t, pos = self.test_data[0][0]
        start = self.trials['stimOn_times']
        end = self.trials['feedback_times']
        traces = wheel.traces_by_trial(t, pos, start=start, end=end)
        # Check correct number of tuples returned
        self.assertEqual(len(traces), start.size)
        expected_ids = (
            [144, 60143],
            [74944, 84943],
            [99944, 102943],
            [119944, 129943],
            [163944, 187943]
        )

        for trace, ind in zip(traces, expected_ids):
            trace_t, trace_pos = trace
            np.testing.assert_array_equal(trace_t[[0, -1]], t[ind])
            np.testing.assert_array_equal(trace_pos[[0, -1]], pos[ind])

    def test_direction_changes(self):
        """Test for brainbox.behavior.wheel.direction_changes"""
        t, pos = self.test_data[0][0]
        on, off, *_ = self.test_data[0][1]
        vel, _ = wheel.velocity_filtered(pos, 1000)
        times, indices = wheel.direction_changes(t, vel, np.c_[on, off])
        # import matplotlib.pyplot as plt
        # plt.plot(np.diff(pos) * 1000)
        # plt.plot(vel)
        self.assertTrue(len(times) == len(indices) == 14, 'incorrect number of arrays returned')

    def test_get_movement_onset(self):
        """Test for brainbox.behavior.wheel.get_movement_onset"""
        on, off, *_ = self.test_data[0][1]
        intervals = np.c_[on, off]
        times = wheel.get_movement_onset(intervals, self.trials['feedback_times'])
        expected = [np.nan, 79.66293334, 100.73593334, 129.26693334, np.nan]
        np.testing.assert_array_almost_equal(times, expected)
        with self.assertRaises(ValueError):
            wheel.get_movement_onset(intervals, np.random.permutation(self.trials['feedback_times']))


class TestTraining(unittest.TestCase):
    def setUp(self):
        """
        Test data contains training data from 10 consecutive sessions from subject SWC_054. It is
        a dict of trials objects with each key indication a session date. By using data
        combinations from different dates can test each of the different training criterion a
        subject goes through in the IBL training pipeline
        """
        pickle_file = Path(__file__).parent.joinpath('fixtures', 'trials_test.pickle')
        if not pickle_file.exists():
            self.trial_data = None
        else:
            with open(pickle_file, 'rb') as f:
                self.trial_data = pickle.load(f)
            # Convert to Bunch

        np.random.seed(0)

    def _get_trials(self, sess_dates):
        trials_copy = copy.deepcopy(self.trial_data)
        trials = Bunch(zip(sess_dates, [trials_copy[k] for k in sess_dates]))
        task_protocol = [trials[k].pop('task_protocol') for k in trials.keys()]
        return trials, task_protocol

    def test_psychometric_insufficient_data(self):
        # the psychometric aggregate should return NaN when there is no data for a given contrast
        trials, _ = self._get_trials(sess_dates=['2020-08-25', '2020-08-24', '2020-08-21'])
        trials_all = train.concatenate_trials(trials)
        trials_all['probability_left'] = trials_all['contrastLeft'] * 0 + 80
        psych_nan = train.compute_psychometric(trials_all, block=100)
        assert np.sum(np.isnan(psych_nan)) == 4

    def test_concatenate_and_computations(self):
        trials, _ = self._get_trials(sess_dates=['2020-08-25', '2020-08-24', '2020-08-21'])
        trials_total = np.sum([len(trials[k]['contrastRight']) for k in trials.keys()])
        trials_all = train.concatenate_trials(trials)
        assert (len(trials_all['contrastRight']) == trials_total)

        perf_easy = np.array([train.compute_performance_easy(trials[k]) for k in trials.keys()])
        n_trials = np.array([train.compute_n_trials(trials[k]) for k in trials.keys()])
        psych = train.compute_psychometric(trials_all)
        rt = train.compute_median_reaction_time(trials_all, contrast=0)
        np.testing.assert_allclose(perf_easy, [0.91489362, 0.9, 0.90853659])
        np.testing.assert_array_equal(n_trials, [617, 532, 719])
        np.testing.assert_allclose(psych, [4.04487042, 21.6293942, 1.91451396e-02, 1.72669957e-01],
                                   rtol=1e-5)
        assert (np.isclose(rt, 0.83655))

    def test_in_training(self):
        trials, task_protocol = self._get_trials(
            sess_dates=['2020-08-25', '2020-08-24', '2020-08-21'])
        assert (np.all(np.array(task_protocol) == 'training'))
        status, info = train.get_training_status(
            trials, task_protocol, ephys_sess_dates=[], n_delay=0)
        assert (status == 'in training')

    def test_trained_1a(self):
        trials, task_protocol = self._get_trials(
            sess_dates=['2020-08-26', '2020-08-25', '2020-08-24'])
        assert (np.all(np.array(task_protocol) == 'training'))
        status, info = train.get_training_status(trials, task_protocol, ephys_sess_dates=[],
                                                 n_delay=0)
        assert (status == 'trained 1a')

    def test_trained_1b(self):
        trials, task_protocol = self._get_trials(
            sess_dates=['2020-08-27', '2020-08-26', '2020-08-25'])
        assert (np.all(np.array(task_protocol) == 'training'))
        status, info = train.get_training_status(trials, task_protocol, ephys_sess_dates=[],
                                                 n_delay=0)
        self.assertEqual(status, 'trained 1b')

    def test_training_to_bias(self):
        trials, task_protocol = self._get_trials(
            sess_dates=['2020-08-31', '2020-08-28', '2020-08-27'])
        assert (~np.all(np.array(task_protocol) == 'training') and
                np.any(np.array(task_protocol) == 'training'))
        status, info = train.get_training_status(trials, task_protocol, ephys_sess_dates=[],
                                                 n_delay=0)
        assert (status == 'trained 1b')

    def test_ready4ephys(self):
        sess_dates = ['2020-09-01', '2020-08-31', '2020-08-28']
        trials, task_protocol = self._get_trials(sess_dates=sess_dates)
        assert (np.all(np.array(task_protocol) == 'biased'))
        status, info = train.get_training_status(trials, task_protocol, ephys_sess_dates=[],
                                                 n_delay=0)
        assert (status == 'ready4ephysrig')

    def test_ready4delay(self):
        sess_dates = ['2020-09-03', '2020-09-02', '2020-08-31']
        trials, task_protocol = self._get_trials(sess_dates=sess_dates)
        assert (np.all(np.array(task_protocol) == 'biased'))
        status, info = train.get_training_status(trials, task_protocol,
                                                 ephys_sess_dates=['2020-09-03'], n_delay=0)
        assert (status == 'ready4delay')

    def test_ready4recording(self):
        sess_dates = ['2020-09-01', '2020-08-31', '2020-08-28']
        trials, task_protocol = self._get_trials(sess_dates=sess_dates)
        assert (np.all(np.array(task_protocol) == 'biased'))
        status, info = train.get_training_status(trials, task_protocol,
                                                 ephys_sess_dates=sess_dates, n_delay=1)
        assert (status == 'ready4recording')

    def test_query_criterion(self):
        """Test for brainbox.behavior.training.query_criterion function."""
        one = ONE(**TEST_DB)

        subject = 'KS005'
        status_map = {
            'trained_1a': ['2019-04-04', 'aaf101c3-2581-450a-8abd-ddb8f557a5ad'],
            'trained_1b': ['2019-04-05', '1883dedd-4f25-4d3d-bc9a-97f3b366a0a0'],
            'in_training': ['2019-04-01', '01390fcc-4f86-4707-8a3b-4d9309feb0a1'],
            'ready4delay': ['2019-04-09', 'f33f41cc-347a-458d-98c8-7e1c2c9c7600'],
            'ready4ephysrig': ['2019-04-10', 'abf5109c-d780-44c8-9561-83e857c7bc01'],
            'ready4recording': ['2019-04-11', '7dc3c44b-225f-4083-be3d-07b8562885f4']
        }
        with mock.patch.object(one.alyx, 'rest', return_value={'json': {'trained_criteria': status_map}}):
            eid, n_sessions, n_days = train.query_criterion(subject, 'in_training', one=one)
            self.assertEqual('01390fcc-4f86-4707-8a3b-4d9309feb0a1', eid)
            self.assertEqual(1, n_sessions)
            self.assertEqual(0, n_days)

            eid, n_sessions, n_days = train.query_criterion(subject, 'ready4ephysrig', from_status='trained_1b', one=one)
            self.assertEqual('abf5109c-d780-44c8-9561-83e857c7bc01', eid)
            self.assertEqual(5, n_sessions)
            self.assertEqual(5, n_days)

            self.assertTrue(all(x is None for x in train.query_criterion(subject, 'untrainable', one=one)))
            eid, n_sessions, n_days = train.query_criterion(subject, 'trained_1b', from_status='ready4ephysrig', one=one)
            self.assertEqual('1883dedd-4f25-4d3d-bc9a-97f3b366a0a0', eid)
            self.assertIsNone(n_sessions)
            self.assertIsNone(n_days)
            self.assertRaises(ValueError, train.query_criterion, subject, 'foobar', one=one)

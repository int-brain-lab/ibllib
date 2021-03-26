from pathlib import Path
import unittest
import numpy as np
import pickle
import copy
from brainbox.core import Bunch
import brainbox.behavior.wheel as wheel
import brainbox.behavior.training as train
import brainbox.behavior.pyschofit as psy


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
        pickle_file = Path(__file__).parent.joinpath('wheel_test.p')
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

    def test_derivative(self):
        if self.test_data is None:
            return
        t = np.array([0, .5, 1., 1.5, 2, 3, 4, 4.5, 5, 5.5])
        p = np.arange(len(t))
        v = wheel.velocity(t, p)
        self.assertTrue(len(v) == len(t))
        self.assertTrue(np.all(v[0:4] == 2) and v[5] == 1 and np.all(v[7:] == 2))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(t[:-1] + np.diff(t) / 2, np.diff(p) / np.diff(t), '*-')
        # plt.plot(t, v, '-*')

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
        t, pos = self.test_data[0][0]
        on, off, *_ = self.test_data[0][1]
        vel, _ = wheel.velocity_smoothed(pos, 1000)
        times, indices = wheel.direction_changes(t, vel, np.c_[on, off])

        self.assertTrue(len(times) == len(indices) == 14, 'incorrect number of arrays returned')
        # Check first arrays
        np.testing.assert_allclose(times[0], [21.86593334, 22.12693334, 22.20193334, 22.66093334])
        np.testing.assert_array_equal(indices[0], [21809, 22070, 22145, 22604])


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
        rt = train.compute_median_reaction_time(trials_all)
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
        assert(status == 'trained 1b')

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


class PsychofitTest(unittest.TestCase):
    def setUp(self) -> None:
        """Data are 3 x n arrays"""
        data = {'weibull50': np.vstack([
            10 ** np.linspace(-4, -1, 8),
            np.ones(8) * 80,
            np.array([0.5125, 0.35, 0.5625, 0.5375, 0.8875, 0.8875, 0.9125, 0.8625])
        ]), 'weibull': np.vstack([
            10 ** np.linspace(-4, -1, 8),
            np.ones(8) * 80,
            np.array([0.125, 0.1125, 0.1375, 0.4, 0.8125, 0.9, 0.925, 0.875])
        ]), 'erf_psycho_2gammas': np.vstack([
            np.arange(-50, 50, 10),
            np.ones(10) * 40,
            np.array([0.175, 0.225, 0.35, 0.275, 0.725, 0.9, 0.925, 0.975, 1., 1.])
        ]), 'erf_psycho': np.vstack([
            np.arange(-50, 50, 10),
            np.ones(10) * 40,
            np.array([0.1, 0.125, 0.25, 0.15, 0.6, 0.65, 0.75, 0.9, 0.9, 0.85])
        ])}
        self.test_data = data
        np.random.seed(0)

    def test_weibull50(self):
        xx = self.test_data['weibull50'][0, :]

        # test parameters
        alpha = 10 ** -2.5
        beta = 2.
        gamma = 0.1

        # fake experimental data given those parameters
        actual = psy.weibull50((alpha, beta, gamma), xx)
        expected = np.array(
            [0.5003998, 0.50286841, 0.5201905, 0.62446761, 0.87264857, 0.9, 0.9, 0.9]
        )
        self.assertTrue(np.allclose(expected, actual))

        with self.assertRaises(ValueError):
            psy.weibull50((alpha, beta), xx)

    def test_weibull(self):
        xx = self.test_data['weibull'][0, :]

        # test parameters
        alpha = 10 ** -2.5
        beta = 2.
        gamma = 0.1

        # fake experimental data given those parameters
        actual = psy.weibull((alpha, beta, gamma), xx)
        expected = np.array(
            [0.1007996, 0.10573682, 0.14038101, 0.34893523, 0.84529714, 0.9, 0.9, 0.9]
        )
        self.assertTrue(np.allclose(expected, actual))

        with self.assertRaises(ValueError):
            psy.weibull((alpha, beta), xx)

    def test_erf_psycho(self):
        xx = self.test_data['erf_psycho'][0, :]

        # test parameters
        bias = -10.
        threshold = 20.
        lapse = .1

        # fake experimental data given those parameters
        actual = psy.erf_psycho((bias, threshold, lapse), xx)
        expected = np.array(
            [0.10187109, 0.11355794, 0.16291968, 0.29180005, 0.5,
             0.70819995, 0.83708032, 0.88644206, 0.89812891, 0.89983722]
        )
        self.assertTrue(np.allclose(expected, actual))

        with self.assertRaises(ValueError):
            psy.erf_psycho((bias, threshold, lapse, lapse), xx)

    def test_erf_psycho_2gammas(self):
        xx = self.test_data['erf_psycho_2gammas'][0, :]

        # test parameters
        bias = -10.
        threshold = 20.
        gamma1 = .2
        gamma2 = 0.

        # fake experimental data given those parameters
        actual = psy.erf_psycho_2gammas((bias, threshold, gamma1, gamma2), xx)
        expected = np.array(
            [0.20187109, 0.21355794, 0.26291968, 0.39180005, 0.6,
             0.80819995, 0.93708032, 0.98644206, 0.99812891, 0.99983722]
        )
        self.assertTrue(np.allclose(expected, actual))

        with self.assertRaises(ValueError):
            psy.erf_psycho_2gammas((bias, threshold, gamma1), xx)

    def test_neg_likelihood(self):
        data = self.test_data['erf_psycho']
        with self.assertRaises(ValueError):
            psy.neg_likelihood((10, 20, .05), data[1:, :])
        with self.assertRaises(TypeError):
            psy.neg_likelihood('(10, 20, .05)', data)

        ll = psy.neg_likelihood((-20, 30, 2), data, P_model='erf_psycho',
                                parmin=np.array((-10, 20, 0)), parmax=np.array((10, 10, .05)))
        self.assertTrue(ll > 10000)

    def test_mle_fit_psycho(self):
        expected = {
            'weibull50': (np.array([0.0034045, 3.9029162, .1119576]), -334.1149693046583),
            'weibull': (np.array([0.00316341, 1.72552866, 0.1032307]), -261.235178611311),
            'erf_psycho': (np.array([-9.78747259, 10., 0.15967605]), -193.0509031440323),
            'erf_psycho_2gammas': (np.array([-11.45463779, 9.9999999, 0.24117732, 0.0270835]),
                                   -147.02380025592902)
        }
        for model in self.test_data.keys():
            pars, L = psy.mle_fit_psycho(self.test_data[model], P_model=model, nfits=10)
            expected_pars, expected_L = expected[model]
            self.assertTrue(np.allclose(expected_pars, pars, atol=1e-3),
                            f'unexpected pars for {model}')
            self.assertTrue(np.isclose(expected_L, L, atol=1e-3),
                            f'unexpected likelihood for {model}')

        # Test on of the models with function pars
        params = {
            'parmin': np.array([-5., 10., 0.]),
            'parmax': np.array([5., 15., .1]),
            'parstart': np.array([0., 11., 0.1]),
            'nfits': 5
        }
        model = 'erf_psycho'
        pars, L = psy.mle_fit_psycho(self.test_data[model], P_model=model, **params)
        expected = [-5, 15, 0.1]
        self.assertTrue(np.allclose(expected, pars, rtol=.01), f'unexpected pars for {model}')
        self.assertTrue(np.isclose(-195.55603, L, atol=1e-5), f'unexpected likelihood for {model}')

    def tearDown(self):
        np.random.seed()

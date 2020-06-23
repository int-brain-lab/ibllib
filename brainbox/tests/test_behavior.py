from pathlib import Path
import unittest
import numpy as np
import pickle
import brainbox.behavior.wheel as wheel


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

from pathlib import Path
import unittest
import numpy as np
import pickle
import brainbox.behavior.wheel as wheel


class TestWheel(unittest.TestCase):

    def setUp(self):
        # Test data is in the form ((inputs), (outputs)) where inputs is a tuple containing a
        # numpy array of timestamps and one of positions; outputs is a tuple of outputs from
        # teh function under test, e.g. wheel.movements
        pickle_file = Path(__file__).parent.joinpath('wheel_test.p')
        if not pickle_file.exists():
            self.test_data = None
        else:
            with open(pickle_file, 'rb') as f:
                self.test_data = pickle.load(f)

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

import unittest
import numpy as np

import ibllib.behaviour.wheel as wheel


class TestWheel(unittest.TestCase):

    def test_derivative(self):
        t = np.array([0, .5, 1., 1.5, 2, 3, 4, 4.5, 5, 5.5])
        p = np.arange(len(t))
        v = wheel.velocity(t, p)
        self.assertTrue(len(v) == len(t))
        self.assertTrue(np.all(v[0:4] == 2) and v[5] == 1 and np.all(v[7:] == 2))
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.plot(t[:-1] + np.diff(t) / 2, np.diff(p) / np.diff(t), '*-')
        # plt.plot(t, v, '-*')

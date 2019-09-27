from brainbox.singlecell import acorr
import unittest
import numpy as np


class TestPopulation(unittest.TestCase):
    def test_acorr_0(self):
        spike_times = np.array([0, 10, 10, 20])
        bin_size = 1
        winsize_bins = 2 * 3 + 1

        c_expected = np.zeros(7, dtype=np.int32)
        c_expected[3] = 1

        c = acorr(spike_times, bin_size=bin_size, window_size=winsize_bins)

        self.assertTrue(np.allclose(c, c_expected))


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main(exit=False)

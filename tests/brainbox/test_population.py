from brainbox.population import xcorr
import unittest
import numpy as np


def _random_data(max_cluster):
    nspikes = 10000
    spike_times = np.cumsum(np.random.exponential(scale=.025, size=nspikes))
    spike_clusters = np.random.randint(0, max_cluster, nspikes)
    return spike_times, spike_clusters


class TestPopulation(unittest.TestCase):
    def test_xcorr_0(self):
        # 0: 0, 10
        # 1: 10, 20
        spike_times = np.array([0, 10, 10, 20])
        spike_clusters = np.array([0, 1, 0, 1])
        bin_size = 1
        winsize_bins = 2 * 3 + 1

        c_expected = np.zeros((2, 2, 7), dtype=np.int32)

        c_expected[1, 0, 3] = 1
        c_expected[0, 1, 3] = 1

        c = xcorr(spike_times, spike_clusters, bin_size=bin_size, window_size=winsize_bins)

        self.assertTrue(np.allclose(c, c_expected))

    def test_xcorr_1(self):
        # 0: 2, 10, 12, 30
        # 1: 3, 24
        # 2: 20
        spike_times = np.array([2, 3, 10, 12, 20, 24, 30, 40], dtype=np.uint64)
        spike_clusters = np.array([0, 1, 0, 0, 2, 1, 0, 2])
        bin_size = 1
        winsize_bins = 2 * 3 + 1

        c_expected = np.zeros((3, 3, 7), dtype=np.int32)
        c_expected[0, 1, 4] = 1
        c_expected[1, 0, 2] = 1
        c_expected[0, 0, 1] = 1
        c_expected[0, 0, 5] = 1

        c = xcorr(spike_times, spike_clusters, bin_size=bin_size, window_size=winsize_bins)

        self.assertTrue(np.allclose(c, c_expected))

    def test_xcorr_2(self):
        max_cluster = 10
        spike_times, spike_clusters = _random_data(max_cluster)
        bin_size, winsize_bins = .001, .05

        c = xcorr(spike_times, spike_clusters, bin_size=bin_size, window_size=winsize_bins)

        self.assertEqual(c.shape, (max_cluster, max_cluster, 51))


if __name__ == "__main__":
    np.random.seed(0)
    unittest.main(exit=False)

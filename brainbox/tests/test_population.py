from pathlib import Path
import pickle
from sklearn.naive_bayes import MultinomialNB
from brainbox.population import xcorr, classify, regress, get_spike_counts_in_bins
import unittest
import numpy as np


def _random_data(max_cluster):
    nspikes = 10000
    spike_times = np.cumsum(np.random.exponential(scale=.025, size=nspikes))
    spike_clusters = np.random.randint(0, max_cluster, nspikes)
    return spike_times, spike_clusters


class TestPopulation(unittest.TestCase):

    def setUp(self):
        # Test data is a dictionary of spike times and clusters and event times and groups
        pickle_file = Path(__file__).parent.joinpath('fixtures', 'ephys_test.p')
        if not pickle_file.exists():
            self.test_data = None
        else:
            with open(pickle_file, 'rb') as f:
                self.test_data = pickle.load(f)

    def test_get_spike_counts_in_bins(self):
        if self.test_data is None:
            return
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        times = np.column_stack(((event_times - 0.5), (event_times + 0.5)))
        counts, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, times)
        num_clusters = np.size(np.unique(spike_clusters))
        self.assertEqual(counts.shape, (num_clusters, np.size(event_times)))
        self.assertTrue(np.size(cluster_ids) == num_clusters)

    def test_classify(self):
        if self.test_data is None:
            return
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        event_groups = self.test_data['event_groups']
        clf = MultinomialNB()
        times = np.column_stack(((event_times - 0.5), (event_times + 0.5)))
        counts, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, times)
        counts = counts.T
        accuracy, pred, prob = classify(counts, event_groups, clf)
        self.assertTrue(accuracy == 0.8888888888888888)
        self.assertEqual(pred.shape, event_groups.shape)
        self.assertEqual(prob.shape, event_groups.shape)

    def test_regress(self):
        if self.test_data is None:
            return
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        event_groups = self.test_data['event_groups']
        times = np.column_stack(((event_times - 0.5), (event_times + 0.5)))
        counts, cluster_ids = get_spike_counts_in_bins(spike_times, spike_clusters, times)
        counts = counts.T
        pred = regress(counts, event_groups)
        self.assertEqual(pred.shape, event_groups.shape)

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

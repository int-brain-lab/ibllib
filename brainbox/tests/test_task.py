from pathlib import Path
import pickle
import unittest
import numpy as np
import brainbox as bb


class TestTask(unittest.TestCase):

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
        counts, cluster_ids = bb.task._get_spike_counts_in_bins(spike_times,
                                                                spike_clusters,
                                                                times)
        num_clusters = np.size(np.unique(spike_clusters))
        self.assertEqual(counts.shape, (num_clusters, np.size(event_times)))
        self.assertTrue(np.size(cluster_ids) == num_clusters)

    def test_responsive_units(self):
        if self.test_data is None:
            return
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        alpha = 0.5
        sig_units, stats, p_values, cluster_ids = bb.task.responsive_units(spike_times,
                                                                           spike_clusters,
                                                                           event_times,
                                                                           pre_time=[0.5, 0],
                                                                           post_time=[0, 0.5],
                                                                           alpha=alpha)
        num_clusters = np.size(np.unique(spike_clusters))
        self.assertTrue(np.size(sig_units) == 232)
        self.assertTrue(np.sum(p_values < alpha) == np.size(sig_units))
        self.assertTrue(np.size(cluster_ids) == num_clusters)

    def test_differentiate_units(self):
        if self.test_data is None:
            return
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        event_groups = self.test_data['event_groups']
        alpha = 0.99
        sig_units, stats, p_values, cluster_ids = bb.task.differentiate_units(spike_times,
                                                                              spike_clusters,
                                                                              event_times,
                                                                              event_groups,
                                                                              pre_time=0.5,
                                                                              post_time=0.5,
                                                                              alpha=alpha)
        num_clusters = np.size(np.unique(spike_clusters))
        self.assertTrue(np.size(sig_units) == 0)
        self.assertTrue(np.sum(p_values < alpha) == np.size(sig_units))
        self.assertTrue(np.size(cluster_ids) == num_clusters)

    def test_roc_single_event(self):
        if self.test_data is None:
            return
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        auc_roc, cluster_ids = bb.task.roc_single_event(spike_times,
                                                        spike_clusters,
                                                        event_times,
                                                        pre_time=[0.5, 0],
                                                        post_time=[0, 0.5])
        num_clusters = np.size(np.unique(spike_clusters))
        self.assertTrue(np.sum(auc_roc < 0.3) == 4)
        self.assertTrue(np.sum(auc_roc > 0.6) == 25)
        self.assertTrue(np.size(cluster_ids) == num_clusters)

    def test_roc_between_two_events(self):
        if self.test_data is None:
            return
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        event_groups = self.test_data['event_groups']
        auc_roc, cluster_ids = bb.task.roc_between_two_events(spike_times,
                                                              spike_clusters,
                                                              event_times,
                                                              event_groups,
                                                              pre_time=0.5,
                                                              post_time=0.5)
        num_clusters = np.size(np.unique(spike_clusters))
        self.assertTrue(np.sum(auc_roc < 0.3) == 24)
        self.assertTrue(np.sum(auc_roc > 0.7) == 10)
        self.assertTrue(np.size(cluster_ids) == num_clusters)


if __name__ == "__main__":
    unittest.main(exit=False)

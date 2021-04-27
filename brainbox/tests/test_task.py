from pathlib import Path
import pickle
import unittest

import numpy as np
import brainbox.task.closed_loop as task
import pandas as pd


class TestTask(unittest.TestCase):

    def setUp(self):
        # Test ephys data is a dictionary of spike times and clusters and event times and groups
        pickle_file = Path(__file__).parent.joinpath('fixtures', 'ephys_test.p')
        assert pickle_file.exists()
        with open(pickle_file, 'rb') as f:
            self.test_data = pickle.load(f)

        # Test trials data is pandas dataframe with trials
        csv_file = Path(__file__).parent.joinpath('fixtures', 'trials_df_test.csv')
        self.test_trials = pd.read_csv(csv_file)

    def test_responsive_units(self):
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        alpha = 0.5
        sig_units, stats, p_values, cluster_ids = task.responsive_units(spike_times,
                                                                        spike_clusters,
                                                                        event_times,
                                                                        pre_time=[0.5, 0],
                                                                        post_time=[0, 0.5],
                                                                        alpha=alpha)
        num_clusters = np.size(np.unique(spike_clusters))
        self.assertTrue(np.sum(p_values < alpha) == np.size(sig_units))
        self.assertTrue(np.size(p_values) == np.size(cluster_ids))
        self.assertTrue(np.size(cluster_ids) == num_clusters)

    def test_differentiate_units(self):
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        event_groups = self.test_data['event_groups']
        alpha = 0.99
        sig_units, stats, p_values, cluster_ids = task.differentiate_units(spike_times,
                                                                           spike_clusters,
                                                                           event_times,
                                                                           event_groups,
                                                                           pre_time=0.5,
                                                                           post_time=0.5,
                                                                           alpha=alpha)
        num_clusters = np.size(np.unique(spike_clusters))
        self.assertTrue(np.sum(p_values < alpha) == np.size(sig_units))
        self.assertTrue(np.size(p_values) == np.size(cluster_ids))
        self.assertTrue(np.size(cluster_ids) == num_clusters)

    def test_roc_single_event(self):
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        auc_roc, cluster_ids = task.roc_single_event(spike_times,
                                                     spike_clusters,
                                                     event_times,
                                                     pre_time=[0.5, 0],
                                                     post_time=[0, 0.5])
        num_clusters = np.size(np.unique(spike_clusters))
        self.assertTrue(np.sum(auc_roc < 0.3) == 4)
        self.assertTrue(np.sum(auc_roc > 0.6) == 25)
        self.assertTrue(np.size(cluster_ids) == num_clusters)

    def test_roc_between_two_events(self):
        spike_times = self.test_data['spike_times']
        spike_clusters = self.test_data['spike_clusters']
        event_times = self.test_data['event_times']
        event_groups = self.test_data['event_groups']
        auc_roc, cluster_ids = task.roc_between_two_events(spike_times,
                                                           spike_clusters,
                                                           event_times,
                                                           event_groups,
                                                           pre_time=0.5,
                                                           post_time=0.5)
        num_clusters = np.size(np.unique(spike_clusters))
        self.assertTrue(np.sum(auc_roc < 0.3) == 24)
        self.assertTrue(np.sum(auc_roc > 0.7) == 10)
        self.assertTrue(np.size(cluster_ids) == num_clusters)

    def test_generate_pseudo_blocks(self):
        blocks = task.generate_pseudo_blocks(100,
                                             factor=60,
                                             min_=20,
                                             max_=100,
                                             first5050=90)
        self.assertTrue(len(blocks.shape) == 1)
        self.assertTrue(blocks.shape[0] == 100)
        self.assertTrue(blocks[0] == 0.5)
        self.assertTrue(np.sum(blocks == 0.5) == 90)

    def test_generate_pseudo_stimuli(self):
        p_left, contrast_l, contrast_r = task.generate_pseudo_stimuli(100,
                                                                      contrast_set=[0.5, 1],
                                                                      first5050=90)
        self.assertTrue(p_left.shape[0] == 100)
        self.assertTrue(np.sum(p_left == 0.5) == 90)
        self.assertTrue(p_left.shape == contrast_l.shape == contrast_r.shape)

    def test_generate_pseudo_session(self):
        test_trials = self.test_trials
        pseudo_trials = task.generate_pseudo_session(test_trials)
        self.assertTrue(pseudo_trials.shape[0] == test_trials.shape[0])


if __name__ == "__main__":
    unittest.main(exit=False)

import unittest
import pickle
from pathlib import Path
from brainbox.task.trials import find_trial_ids
import numpy as np

#class TestTraining(unittest.TestCase):
#    def setUp(self):
#        """
#        Test data contains training data from 10 consecutive sessions from subject SWC_054. It is
#        a dict of trials objects with each key indication a session date. By using data
#        combinations from different dates can test each of the different training criterion a
#        subject goes through in the IBL training pipeline
#        """
#        pickle_file = Path(__file__).parent.joinpath('fixtures', 'trials_test.pickle')
#        if not pickle_file.exists():
#            self.trial_data = None
#        else:
#            with open(pickle_file, 'rb') as f:
#                self.trial_data = pickle.load(f)
#
#
pickle_file = Path(r'C:\Users\Mayo\iblenv\ibllib-repo\brainbox\tests\fixtures\trials_test.pickle')

with open(pickle_file, 'rb') as f:
    trial_data = pickle.load(f)

trials = trial_data['2020-08-26']
trials['goCue_times'] = trials['stimOn_times']

# Test the default returns all trials
ids, div = find_trial_ids(trials)
expected_ids = np.arange(len(trials['probabilityLeft']))
assert(all(ids == expected_ids))

# Test that filtering by correct
ids, div = find_trial_ids(trials, choice='correct')
expected_ids = np.where(trials['feedbackType'] == 1)[0]
assert(all(ids == expected_ids))

# Test that filtering by incorrect
ids, div = find_trial_ids(trials, choice='incorrect')
expected_ids = np.where(trials['feedbackType'] == -1)[0]
assert(all(ids == expected_ids))

# Test that filtering by left
ids, div = find_trial_ids(trials, side='left')
expected_ids = np.where(~np.isnan(trials['contrastLeft']))[0]
assert(all(ids == expected_ids))

# Test that filtering by right
ids, div = find_trial_ids(trials, side='right')
expected_ids = np.where(~np.isnan(trials['contrastRight']))[0]
assert(all(ids == expected_ids))

# Test filter choice and side
# right correct
ids, div = find_trial_ids(trials, side='right', choice='correct')
righ_corr_expected_ids = np.where(np.bitwise_and(~np.isnan(trials['contrastRight']),
                                  trials['feedbackType'] == 1))[0]
assert(all(ids == righ_corr_expected_ids))

# right incorrect
ids, div = find_trial_ids(trials, side='right', choice='incorrect')
righ_incor_expected_ids = np.where(np.bitwise_and(~np.isnan(trials['contrastRight']),
                                   trials['feedbackType'] == -1))[0]
assert(all(ids == righ_incor_expected_ids))

# left correct
ids, div = find_trial_ids(trials, side='left', choice='correct')
left_corr_expected_ids = np.where(np.bitwise_and(~np.isnan(trials['contrastLeft']),
                                  trials['feedbackType'] == 1))[0]
assert(all(ids == left_corr_expected_ids))

# left incorrect
ids, div = find_trial_ids(trials, side='left', choice='incorrect')
left_incorr_expected_ids = np.where(np.bitwise_and(~np.isnan(trials['contrastLeft']),
                                    trials['feedbackType'] == -1))[0]
assert(all(ids == left_incorr_expected_ids))

# Test sorting
ids, div = find_trial_ids(trials, sort='choice and side')
expected_ids = np.r_[left_corr_expected_ids, left_incorr_expected_ids,
                     righ_corr_expected_ids, righ_incor_expected_ids]
assert(all(ids == expected_ids))

ids, div = find_trial_ids(trials, side='left', sort='choice')
expected_ids = np.r_[left_corr_expected_ids, left_incorr_expected_ids]
assert(all(ids == expected_ids))

ids, div = find_trial_ids(trials, side='left', sort='choice and side')
assert(all(ids == expected_ids))

ids, div = find_trial_ids(trials, side='left', sort='side')
expected_ids = np.where(~np.isnan(trials['contrastLeft']))[0]
assert(all(ids == expected_ids))

# Test ordering by reaction time
reaction_time = trials['response_times'] - trials['goCue_times']
expected_ids = np.argsort(reaction_time)
ids, div = find_trial_ids(trials, order='reaction time')
assert(all(ids == expected_ids))

ids, div = find_trial_ids(trials, side='left', choice='correct', order='reaction time')
expected_ids = left_corr_expected_ids[np.argsort(reaction_time[left_corr_expected_ids])]
assert(all(ids == expected_ids))








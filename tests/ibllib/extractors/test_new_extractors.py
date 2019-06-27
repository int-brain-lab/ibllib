import logging
import unittest
from pathlib import Path

import numpy as np

from ibllib.io import raw_data_loaders as loaders
import ibllib.io.extractors


class TestExtractTrialDataNew(unittest.TestCase):

    def setUp(self):
        self.new_session_training_path = Path(__file__).parent / 'data' / 'new_session_training'
        self.new_session_biased_path = Path(__file__).parent / 'data' / 'new_session_biased'
        # turn off logging for unit testing as we will purposely go into warning/error cases
        self.logger = logging.getLogger('ibllib').setLevel(50)

    def test_get_feedbackType(self):
        # Check training session
        ft = ibllib.io.extractors.training_trials.get_feedbackType(
            self.new_session_training_path, save=False, data=False)
        self.assertTrue(isinstance(ft, np.ndarray))
        # check if no 0's in feedbackTypes
        self.assertFalse(ft[ft == 0].size > 0)
        # check Biased session
        ft = ibllib.io.extractors.biased_trials.get_feedbackType(
            self.new_session_biased_path, save=False, data=False)
        self.assertTrue(isinstance(ft, np.ndarray))
        # check if no 0's in feedbackTypes
        self.assertFalse(ft[ft == 0].size > 0)

    def test_get_contrastLR(self):
        # Training session
        cl, cr = ibllib.io.extractors.training_trials.get_contrastLR(
            self.new_session_training_path)
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))
        # Biased session
        cl, cr = ibllib.io.extractors.biased_trials.get_contrastLR(self.new_session_biased_path)
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))

    def test_get_probabilityLeft(self):
        # Training session
        pl = ibllib.io.extractors.training_trials.get_probabilityLeft(
            self.new_session_training_path)
        self.assertTrue(isinstance(pl, np.ndarray))
        # Biased session
        pl = ibllib.io.extractors.biased_trials.get_probabilityLeft(
            self.new_session_biased_path)
        md = loaders.load_settings(self.new_session_biased_path)
        self.assertTrue(isinstance(pl, np.ndarray))
        probs = md['BLOCK_PROBABILITY_SET']
        probs.append(0.5)
        self.assertTrue(sum([x in probs for x in pl]) == len(pl))

    def test_get_choice(self):
        # Training session
        choice = ibllib.io.extractors.training_trials.get_choice(self.new_session_training_path)
        self.assertTrue(isinstance(choice, np.ndarray))
        data = loaders.load_data(self.new_session_training_path)
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in data])
        self.assertTrue(all(choice[trial_nogo]) == 0)
        # Biased session
        choice = ibllib.io.extractors.biased_trials.get_choice(self.new_session_biased_path)
        self.assertTrue(isinstance(choice, np.ndarray))
        data = loaders.load_data(self.new_session_biased_path)
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in data])
        self.assertTrue(all(choice[trial_nogo]) == 0)
        # # Don't think we need this any more
        # signed_contrast = np.array([t['signed_contrast'] for t in data])

        # if not all(signed_contrast == 0):
        #     return
        # else:
        #     # This will only fail is the mouse always answers with no go on a
        #     # 0% contrast OR if the choice has been extracted wrong
        #     self.assertTrue(any(choice[signed_contrast == 0] != 0))

    def test_get_repNum(self):
        # TODO: Test its sawtooth
        # Training session
        rn = ibllib.io.extractors.training_trials.get_repNum(
            self.new_session_training_path)
        self.assertTrue(isinstance(rn, np.ndarray))
        # Biased session
        rn = ibllib.io.extractors.biased_trials.get_repNum(
            self.new_session_biased_path)
        self.assertTrue(isinstance(rn, np.ndarray))

    def test_get_rewardVolume(self):
        # Training session
        rv = ibllib.io.extractors.training_trials.get_rewardVolume(
            self.new_session_training_path)
        self.assertTrue(isinstance(rv, np.ndarray))
        # Biased session
        rv = ibllib.io.extractors.biased_trials.get_rewardVolume(
            self.new_session_biased_path)
        self.assertTrue(isinstance(rv, np.ndarray))
        # Test if all non zero rewards are of the same value
        self.assertTrue(all([x == max(rv) for x in rv if x != 0]))

    def test_get_feedback_times(self):
        # Training session
        ft = ibllib.io.extractors.training_trials.get_feedback_times(
            self.new_session_training_path)
        self.assertTrue(isinstance(ft, np.ndarray))
        # Biased session
        ft = ibllib.io.extractors.biased_trials.get_feedback_times(
            self.new_session_biased_path)
        self.assertTrue(isinstance(ft, np.ndarray))

        print('.')
    # def test_stimOn_times(self):
    #     st = ibllib.io.extractors.training_trials.get_stimOn_times(
    # '', save=False, data=self.data)
    #     self.assertTrue(isinstance(st, np.ndarray))

    # def test_encoder_positions_duds(self):
    #     dy = loaders.load_encoder_positions(self.session_path)
    #     self.assertEqual(dy.bns_ts.dtype.name, 'object')
    #     self.assertTrue(dy.shape[0] == 14)

    # def test_encoder_events_duds(self):
    #     dy = loaders.load_encoder_events(self.session_path)
    #     self.assertEqual(dy.bns_ts.dtype.name, 'object')
    #     self.assertTrue(dy.shape[0] == 7)

    # def test_encoder_positions_clock_reset(self):
    #     dy = loaders.load_encoder_positions(self.session_path)
    #     dat = np.array([849736, 1532230, 1822449, 1833514, 1841566, 1848206, 1853979, 1859144])
    #     self.assertTrue(np.all(np.diff(dy['re_ts']) > 0))
    #     self.assertTrue(all(dy['re_ts'][6:] - 2**32 - dat == 0))

    # def test_encoder_positions_clock_errors(self):
    #     # here we test for 2 kinds of file corruption that happen
    #     # 1/2 the first sample time is corrupt and absurdly high and should be discarded
    #     # 2/2 2 samples are swapped and need to be swapped back
    #     dy = loaders.load_encoder_positions(self.session_path_biased)
    #     self.assertTrue(np.all(np.diff(np.array(dy.re_ts)) > 0))

    # def test_wheel_folder(self):
    #     # the wheel folder contains other errors in bpod output that had to be addressed
    #     # 2 first samples timestamp AWOL instead of only one
    #     wf = self.wheel_path / '_iblrig_encoderPositions.raw.2firstsamples.ssv'
    #     df = loaders._load_encoder_positions_file(wf)
    #     self.assertTrue(np.all(np.diff(np.array(df.re_ts)) > 0))
    #     # corruption in the middle of file
    #     wf = self.wheel_path / '_iblrig_encoderEvents.raw.CorruptMiddle.ssv'
    #     df = loaders._load_encoder_events_file(wf)
    #     self.assertTrue(np.all(np.diff(np.array(df.re_ts)) > 0))

    # def test_interpolation(self):
    #     # straight test that it returns an usable function
    #     ta = np.array([0., 1., 2., 3., 4., 5.])
    #     tb = np.array([0., 1.1, 2.0, 2.9, 4., 5.])
    #     finterp = ibllib.io.extractors.training_wheel.time_interpolation(ta, tb)
    #     self.assertTrue(np.all(finterp(ta) == tb))
    #     # next test if sizes are not similar
    #     tc = np.array([0., 1.1, 2.0, 2.9, 4., 5., 6.])
    #     finterp = ibllib.io.extractors.training_wheel.time_interpolation(ta, tc)
    #     self.assertTrue(np.all(finterp(ta) == tb))

    # def test_choice(self):

    # def test_goCue_times(self):
    #     gc_times = ibllib.io.extractors.training_trials.get_goCueOnset_times(self.session_path)
    #     self.assertTrue(not gc_times or gc_times)


if __name__ == "__main__":
    unittest.main(exit=False)
    print('.')

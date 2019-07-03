import logging
import unittest
from pathlib import Path

import numpy as np

from ibllib.io import raw_data_loaders as raw
import ibllib.io.extractors


class TestExtractTrialData(unittest.TestCase):

    def setUp(self):
        self.training_lt5_path = Path(
            __file__).parent / 'data' / 'session_training_lt5'
        self.biased_lt5_path = Path(
            __file__).parent / 'data' / 'session_biased_lt5'
        self.training_ge5_path = Path(
            __file__).parent / 'data' / 'session_training_ge5'
        self.biased_ge5_path = Path(
            __file__).parent / 'data' / 'session_biased_ge5'
        # turn off logging for unit testing as we will purposely go into warning/error cases
        self.wheel_path = Path(__file__).parent / 'data' / 'wheel'
        self.logger = logging.getLogger('ibllib').setLevel(50)

    def test_get_feedbackType(self):
        # TRAINING SESSIONS
        ft = ibllib.io.extractors.training_trials.get_feedbackType(
            self.training_lt5_path, save=False, data=False)
        self.assertTrue(isinstance(ft, np.ndarray))
        # check if no 0's in feedbackTypes
        self.assertFalse(ft[ft == 0].size > 0)
        # -- version >= 5.0.0
        ft = ibllib.io.extractors.training_trials.get_feedbackType(
            self.training_ge5_path, save=False, data=False)
        self.assertTrue(isinstance(ft, np.ndarray))
        # check if no 0's in feedbackTypes
        self.assertFalse(ft[ft == 0].size > 0)

        # BIASED SESSIONS
        ft = ibllib.io.extractors.biased_trials.get_feedbackType(
            self.biased_lt5_path, save=False, data=False)
        self.assertTrue(isinstance(ft, np.ndarray))
        # check if no 0's in feedbackTypes
        self.assertFalse(ft[ft == 0].size > 0)
        # -- version >= 5.0.0
        ft = ibllib.io.extractors.biased_trials.get_feedbackType(
            self.biased_ge5_path, save=False, data=False)
        self.assertTrue(isinstance(ft, np.ndarray))
        # check if no 0's in feedbackTypes
        self.assertFalse(ft[ft == 0].size > 0)

    def test_get_contrastLR(self):
        # TRAINING SESSIONS
        cl, cr = ibllib.io.extractors.training_trials.get_contrastLR(
            self.training_lt5_path)
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))
        # -- version >= 5.0.0
        cl, cr = ibllib.io.extractors.training_trials.get_contrastLR(
            self.training_ge5_path)
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))

        # BIASED SESSIONS
        cl, cr = ibllib.io.extractors.biased_trials.get_contrastLR(
            self.biased_lt5_path)
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))
        # -- version >= 5.0.0
        cl, cr = ibllib.io.extractors.biased_trials.get_contrastLR(
            self.biased_ge5_path)
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))

    def test_get_probabilityLeft(self):
        # TRAINING SESSIONS
        pl = ibllib.io.extractors.training_trials.get_probabilityLeft(
            self.training_lt5_path)
        self.assertTrue(isinstance(pl, np.ndarray))
        # -- version >= 5.0.0
        pl = ibllib.io.extractors.training_trials.get_probabilityLeft(
            self.training_ge5_path)
        self.assertTrue(isinstance(pl, np.ndarray))

        # BIASED SESSIONS
        pl = ibllib.io.extractors.biased_trials.get_probabilityLeft(
            self.biased_lt5_path)
        self.assertTrue(isinstance(pl, np.ndarray))
        # Test if only probs that are in prob set
        md = raw.load_settings(self.biased_lt5_path)
        if md:
            probs = md['BLOCK_PROBABILITY_SET']
            probs.append(0.5)
            self.assertTrue(sum([x in probs for x in pl]) == len(pl))
        # -- version >= 5.0.0
        pl = ibllib.io.extractors.biased_trials.get_probabilityLeft(
            self.biased_ge5_path)
        self.assertTrue(isinstance(pl, np.ndarray))
        # Test if only probs that are in prob set
        md = raw.load_settings(self.biased_ge5_path)
        probs = md['BLOCK_PROBABILITY_SET']
        probs.append(0.5)
        self.assertTrue(sum([x in probs for x in pl]) == len(pl))

    def test_get_choice(self):
        # TRAINING SESSIONS
        choice = ibllib.io.extractors.training_trials.get_choice(
            self.training_lt5_path)
        self.assertTrue(isinstance(choice, np.ndarray))
        data = raw.load_data(self.training_lt5_path)
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in data])
        if any(trial_nogo):
            self.assertTrue(all(choice[trial_nogo]) == 0)
        # -- version >= 5.0.0
        choice = ibllib.io.extractors.training_trials.get_choice(
            self.training_ge5_path)
        self.assertTrue(isinstance(choice, np.ndarray))
        data = raw.load_data(self.training_ge5_path)
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in data])
        if any(trial_nogo):
            self.assertTrue(all(choice[trial_nogo]) == 0)

        # BIASED SESSIONS
        choice = ibllib.io.extractors.biased_trials.get_choice(
            self.biased_lt5_path)
        self.assertTrue(isinstance(choice, np.ndarray))
        data = raw.load_data(self.biased_lt5_path)
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in data])
        if any(trial_nogo):
            self.assertTrue(all(choice[trial_nogo]) == 0)
        # -- version >= 5.0.0
        choice = ibllib.io.extractors.biased_trials.get_choice(
            self.biased_ge5_path)
        self.assertTrue(isinstance(choice, np.ndarray))
        data = raw.load_data(self.biased_ge5_path)
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in data])
        if any(trial_nogo):
            self.assertTrue(all(choice[trial_nogo]) == 0)

    def test_get_repNum(self):
        # TODO: Test its sawtooth
        # TRAINING SESSIONS
        rn = ibllib.io.extractors.training_trials.get_repNum(
            self.training_lt5_path)
        self.assertTrue(isinstance(rn, np.ndarray))
        # -- version >= 5.0.0
        rn = ibllib.io.extractors.training_trials.get_repNum(
            self.training_ge5_path)
        self.assertTrue(isinstance(rn, np.ndarray))

        # BIASED SESSIONS have no repeted trials

    def test_get_rewardVolume(self):
        # TRAINING SESSIONS
        rv = ibllib.io.extractors.training_trials.get_rewardVolume(
            self.training_lt5_path)
        self.assertTrue(isinstance(rv, np.ndarray))
        # -- version >= 5.0.0
        rv = ibllib.io.extractors.training_trials.get_rewardVolume(
            self.training_ge5_path)
        self.assertTrue(isinstance(rv, np.ndarray))

        # BIASED SESSIONS
        rv = ibllib.io.extractors.biased_trials.get_rewardVolume(
            self.biased_lt5_path)
        self.assertTrue(isinstance(rv, np.ndarray))
        # Test if all non zero rewards are of the same value
        self.assertTrue(all([x == max(rv) for x in rv if x != 0]))
        # -- version >= 5.0.0
        rv = ibllib.io.extractors.biased_trials.get_rewardVolume(
            self.biased_ge5_path)
        self.assertTrue(isinstance(rv, np.ndarray))
        # Test if all non zero rewards are of the same value
        self.assertTrue(all([x == max(rv) for x in rv if x != 0]))

    def test_get_feedback_times_ge5(self):
        # TRAINING SESSIONS
        ft = ibllib.io.extractors.training_trials.get_feedback_times_ge5(
            self.training_ge5_path)
        self.assertTrue(isinstance(ft, np.ndarray))

        # BIASED SESSIONS
        ft = ibllib.io.extractors.biased_trials.get_feedback_times_ge5(
            self.biased_ge5_path)
        self.assertTrue(isinstance(ft, np.ndarray))

    def test_get_feedback_times_lt5(self):
        # TRAINING SESSIONS
        ft = ibllib.io.extractors.training_trials.get_feedback_times_lt5(
            self.training_lt5_path)
        self.assertTrue(isinstance(ft, np.ndarray))

        # BIASED SESSIONS
        ft = ibllib.io.extractors.biased_trials.get_feedback_times_lt5(
            self.biased_lt5_path)
        self.assertTrue(isinstance(ft, np.ndarray))

    def test_get_feedback_times(self):
        # TRAINING SESSIONS
        ft = ibllib.io.extractors.training_trials.get_feedback_times(
            self.training_ge5_path)
        self.assertTrue(isinstance(ft, np.ndarray))
        ft = ibllib.io.extractors.training_trials.get_feedback_times(
            self.training_lt5_path, settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        self.assertTrue(isinstance(ft, np.ndarray))

        # BIASED SESSIONS
        ft = ibllib.io.extractors.biased_trials.get_feedback_times(
            self.biased_ge5_path)
        self.assertTrue(isinstance(ft, np.ndarray))
        ft = ibllib.io.extractors.biased_trials.get_feedback_times(
            self.biased_lt5_path, settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        self.assertTrue(isinstance(ft, np.ndarray))

    def test_get_stimOnTrigger_times(self):
        # TRAINING SESSIONS
        sott = ibllib.io.extractors.training_trials.get_stimOnTrigger_times(
            self.training_lt5_path)
        self.assertTrue(isinstance(sott, np.ndarray))
        # -- version >= 5.0.0
        sott = ibllib.io.extractors.training_trials.get_stimOnTrigger_times(
            self.training_ge5_path)
        self.assertTrue(isinstance(sott, np.ndarray))
        # BIASED SESSIONS
        sott = ibllib.io.extractors.biased_trials.get_stimOnTrigger_times(
            self.biased_lt5_path)
        self.assertTrue(isinstance(sott, np.ndarray))
        # -- version >= 5.0.0
        sott = ibllib.io.extractors.biased_trials.get_stimOnTrigger_times(
            self.biased_ge5_path)
        self.assertTrue(isinstance(sott, np.ndarray))

    def test_get_stimOn_times_lt5(self):
        # TRAINING SESSIONS
        st = ibllib.io.extractors.training_trials.get_stimOn_times_lt5(
            self.training_lt5_path)
        self.assertTrue(isinstance(st, np.ndarray))

        # BIASED SESSIONS
        st = ibllib.io.extractors.biased_trials.get_stimOn_times_lt5(
            self.biased_lt5_path)
        self.assertTrue(isinstance(st, np.ndarray))

    def test_get_stimOn_times_ge5(self):
        # TRAINING SESSIONS
        st = ibllib.io.extractors.training_trials.get_stimOn_times_ge5(
            self.training_ge5_path)
        self.assertTrue(isinstance(st, np.ndarray))

        # BIASED SESSIONS
        st = ibllib.io.extractors.biased_trials.get_stimOn_times_ge5(
            self.biased_ge5_path)
        self.assertTrue(isinstance(st, np.ndarray))

    def test_get_stimOn_times(self):
        # TRAINING SESSIONS
        st = ibllib.io.extractors.training_trials.get_stimOn_times(
            self.training_lt5_path, settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        self.assertTrue(isinstance(st, np.ndarray))
        st = ibllib.io.extractors.training_trials.get_stimOn_times(
            self.training_ge5_path)
        self.assertTrue(isinstance(st, np.ndarray))

        # BIASED SESSIONS
        st = ibllib.io.extractors.biased_trials.get_stimOn_times(
            self.biased_lt5_path, settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        self.assertTrue(isinstance(st, np.ndarray))
        st = ibllib.io.extractors.biased_trials.get_stimOn_times(
            self.biased_ge5_path)
        self.assertTrue(isinstance(st, np.ndarray))

    def test_get_intervals(self):
        # TRAINING SESSIONS
        di = ibllib.io.extractors.training_trials.get_intervals(
            self.training_lt5_path)
        self.assertTrue(isinstance(di, np.ndarray))
        self.assertFalse(np.isnan(di).all())
        # -- version >= 5.0.0
        di = ibllib.io.extractors.training_trials.get_intervals(
            self.training_ge5_path)
        self.assertTrue(isinstance(di, np.ndarray))
        self.assertFalse(np.isnan(di).all())

        # BIASED SESSIONS
        di = ibllib.io.extractors.biased_trials.get_intervals(
            self.training_lt5_path)
        self.assertTrue(isinstance(di, np.ndarray))
        self.assertFalse(np.isnan(di).all())
        # -- version >= 5.0.0
        di = ibllib.io.extractors.biased_trials.get_intervals(
            self.training_ge5_path)
        self.assertTrue(isinstance(di, np.ndarray))
        self.assertFalse(np.isnan(di).all())

    def test_get_iti_duration(self):
        # TRAINING SESSIONS
        iti = ibllib.io.extractors.training_trials.get_iti_duration(
            self.training_lt5_path)
        self.assertTrue(isinstance(iti, np.ndarray))
        # -- version >= 5.0.0 iti always == 0.5 sec no extract

        # BIASED SESSIONS
        iti = ibllib.io.extractors.biased_trials.get_iti_duration(
            self.biased_lt5_path)
        self.assertTrue(isinstance(iti, np.ndarray))
        # -- version >= 5.0.0 iti always == 0.5 sec no extract

    def test_get_response_times(self):
        # TRAINING SESSIONS
        rt = ibllib.io.extractors.training_trials.get_response_times(
            self.training_lt5_path)
        self.assertTrue(isinstance(rt, np.ndarray))
        # -- version >= 5.0.0
        rt = ibllib.io.extractors.training_trials.get_response_times(
            self.training_ge5_path)
        self.assertTrue(isinstance(rt, np.ndarray))

        # BIASED SESSIONS
        rt = ibllib.io.extractors.biased_trials.get_response_times(
            self.biased_lt5_path)
        self.assertTrue(isinstance(rt, np.ndarray))
        # -- version >= 5.0.0
        rt = ibllib.io.extractors.biased_trials.get_response_times(
            self.biased_ge5_path)
        self.assertTrue(isinstance(rt, np.ndarray))

    def test_get_goCueTrigger_times(self):
        # TRAINING SESSIONS
        data = raw.load_data(self.training_lt5_path)
        gct = np.array([tr['behavior_data']['States timestamps']
                        ['closed_loop'][0][0] for tr in data])
        self.assertTrue(isinstance(gct, np.ndarray))
        # -- version >= 5.0.0
        gct = ibllib.io.extractors.training_trials.get_goCueTrigger_times(
            self.training_ge5_path)
        self.assertTrue(isinstance(gct, np.ndarray))

        # BIASED SESSIONS
        data = raw.load_data(self.biased_lt5_path)
        gct = np.array([tr['behavior_data']['States timestamps']
                        ['closed_loop'][0][0] for tr in data])
        self.assertTrue(isinstance(gct, np.ndarray))
        # -- version >= 5.0.0
        gct = ibllib.io.extractors.biased_trials.get_goCueTrigger_times(
            self.biased_ge5_path)
        self.assertTrue(isinstance(gct, np.ndarray))

    def test_get_goCueOnset_times(self):
        # TRAINING SESSIONS
        gcot = ibllib.io.extractors.training_trials.get_goCueOnset_times(
            self.training_lt5_path)
        self.assertTrue(isinstance(gcot, np.ndarray))
        self.assertTrue(np.all(np.isnan(gcot)))
        self.assertTrue(gcot.size != 0 or gcot.size == 4)
        # -- version >= 5.0.0
        gcot = ibllib.io.extractors.training_trials.get_goCueOnset_times(
            self.training_ge5_path)
        self.assertTrue(isinstance(gcot, np.ndarray))
        self.assertFalse(np.any(np.isnan(gcot)))
        self.assertTrue(gcot.size != 0 or gcot.size == 12)

        # BIASED SESSIONS
        gcot = ibllib.io.extractors.biased_trials.get_goCueOnset_times(
            self.biased_lt5_path)
        self.assertTrue(isinstance(gcot, np.ndarray))
        self.assertFalse(np.any(np.isnan(gcot)))
        self.assertTrue(gcot.size != 0 or gcot.size == 4)
        # -- version >= 5.0.0
        gcot = ibllib.io.extractors.biased_trials.get_goCueOnset_times(
            self.biased_ge5_path)
        self.assertTrue(isinstance(gcot, np.ndarray))
        self.assertFalse(np.any(np.isnan(gcot)))
        self.assertTrue(gcot.size != 0 or gcot.size == 8)

    def test_get_included_trials_lt5(self):
        # TRAINING SESSIONS
        it = ibllib.io.extractors.training_trials.get_included_trials_lt5(
            self.training_lt5_path)
        self.assertTrue(isinstance(it, np.ndarray))
        # BIASED SESSIONS
        it = ibllib.io.extractors.biased_trials.get_included_trials_lt5(
            self.biased_lt5_path)
        self.assertTrue(isinstance(it, np.ndarray))

    def test_get_included_trials_ge5(self):
        # TRAINING SESSIONS
        it = ibllib.io.extractors.training_trials.get_included_trials_ge5(
            self.training_ge5_path)
        self.assertTrue(isinstance(it, np.ndarray))
        # BIASED SESSIONS
        it = ibllib.io.extractors.biased_trials.get_included_trials_ge5(
            self.biased_ge5_path)
        self.assertTrue(isinstance(it, np.ndarray))

    def test_get_included_trials(self):
        # TRAINING SESSIONS
        it = ibllib.io.extractors.training_trials.get_included_trials(
            self.training_lt5_path, settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        self.assertTrue(isinstance(it, np.ndarray))
        # -- version >= 5.0.0
        it = ibllib.io.extractors.training_trials.get_included_trials(
            self.training_ge5_path)
        self.assertTrue(isinstance(it, np.ndarray))

        # BIASED SESSIONS
        it = ibllib.io.extractors.biased_trials.get_included_trials(
            self.biased_lt5_path, settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        self.assertTrue(isinstance(it, np.ndarray))
        # -- version >= 5.0.0
        it = ibllib.io.extractors.biased_trials.get_included_trials(
            self.biased_ge5_path)
        self.assertTrue(isinstance(it, np.ndarray))

    def test_extract_all(self):
        # TRAINING SESSIONS
        ibllib.io.extractors.training_trials.extract_all(
            self.training_lt5_path, settings={'IBLRIG_VERSION_TAG': '4.9.9'}, save=True)
        # -- version >= 5.0.0
        ibllib.io.extractors.training_trials.extract_all(
            self.training_ge5_path, save=True)
        # BIASED SESSIONS
        ibllib.io.extractors.biased_trials.extract_all(
            self.biased_lt5_path, settings={'IBLRIG_VERSION_TAG': '4.9.9'}, save=True)
        # -- version >= 5.0.0
        ibllib.io.extractors.biased_trials.extract_all(
            self.biased_ge5_path, save=True)

    # ENCODER TESTS (Should be moved to a RawDataLoaders test suite)
    # ENCODER TESTS (Should be moved to a RawDataLoaders test suite)
    def test_encoder_positions_duds(self):
        # TRAINING SESSIONS
        path = self.training_lt5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
        dy = raw._load_encoder_positions_file_lt5(path)
        self.assertEqual(dy.bns_ts.dtype.name, 'object')
        self.assertTrue(dy.shape[0] == 14)
        # -- version >= 5.0.0
        path = self.training_ge5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
        dy = raw._load_encoder_positions_file_ge5(path)
        self.assertTrue(dy.shape[0] == 936)

        # BIASED SESSIONS
        path = self.biased_lt5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
        dy = raw._load_encoder_positions_file_lt5(path)
        self.assertEqual(dy.bns_ts.dtype.name, 'object')
        self.assertTrue(dy.shape[0] == 14)
        # -- version >= 5.0.0
        path = self.biased_ge5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
        dy = raw._load_encoder_positions_file_ge5(path)
        self.assertTrue(dy.shape[0] == 1122)

    def test_encoder_events_duds(self):
        # TRAINING SESSIONS
        path = self.training_lt5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderEvents.raw*.ssv"), None)
        dy = raw._load_encoder_events_file_lt5(path)
        self.assertEqual(dy.bns_ts.dtype.name, 'object')
        self.assertTrue(dy.shape[0] == 7)
        # -- version >= 5.0.0
        path = self.training_ge5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderEvents.raw*.ssv"), None)
        dy = raw._load_encoder_events_file_ge5(path)
        self.assertTrue(dy.shape[0] == 38)

        # BIASED SESSIONS
        path = self.biased_lt5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderEvents.raw*.ssv"), None)
        dy = raw._load_encoder_events_file_lt5(path)
        self.assertEqual(dy.bns_ts.dtype.name, 'object')
        self.assertTrue(dy.shape[0] == 7)
        # -- version >= 5.0.0
        path = self.biased_ge5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderEvents.raw*.ssv"), None)
        dy = raw._load_encoder_events_file_ge5(path)
        self.assertTrue(dy.shape[0] == 26)

    def test_encoder_positions_clock_reset(self):
        # TRAINING SESSIONS
        # TODO: clarify why dat? make general? when should this fail?
        # only for training?
        path = self.training_lt5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
        dy = raw._load_encoder_positions_file_lt5(path)
        dat = np.array([849736, 1532230, 1822449, 1833514, 1841566, 1848206, 1853979, 1859144])
        self.assertTrue(np.all(np.diff(dy['re_ts']) > 0))
        self.assertTrue(all(dy['re_ts'][6:] - 2 ** 32 - dat == 0))

    def test_encoder_positions_clock_errors(self):
        # here we test for 2 kinds of file corruption that happen
        # 1/2 the first sample time is corrupt and absurdly high and should be discarded
        # 2/2 2 samples are swapped and need to be swapped back
        path = self.biased_lt5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
        dy = raw._load_encoder_positions_file_lt5(path)
        self.assertTrue(np.all(np.diff(np.array(dy.re_ts)) > 0))
        # -- version >= 5.0.0
        path = self.biased_ge5_path / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
        dy = raw._load_encoder_positions_file_ge5(path)
        self.assertTrue(np.all(np.diff(np.array(dy.re_ts)) > 0))

    def test_wheel_folder(self):
        # the wheel folder contains other errors in bpod output that had to be addressed
        # 2 first samples timestamp AWOL instead of only one
        wf = self.wheel_path / '_iblrig_encoderPositions.raw.2firstsamples.ssv'
        df = raw._load_encoder_positions_file_lt5(wf)
        self.assertTrue(np.all(np.diff(np.array(df.re_ts)) > 0))
        # corruption in the middle of file
        wf = self.wheel_path / '_iblrig_encoderEvents.raw.CorruptMiddle.ssv'
        df = raw._load_encoder_events_file_lt5(wf)
        self.assertTrue(np.all(np.diff(np.array(df.re_ts)) > 0))

    def test_interpolation(self):
        # straight test that it returns an usable function
        ta = np.array([0., 1., 2., 3., 4., 5.])
        tb = np.array([0., 1.1, 2.0, 2.9, 4., 5.])
        finterp = ibllib.io.extractors.training_wheel.time_interpolation(ta, tb)
        self.assertTrue(np.all(finterp(ta) == tb))
        # next test if sizes are not similar
        tc = np.array([0., 1.1, 2.0, 2.9, 4., 5., 6.])
        finterp = ibllib.io.extractors.training_wheel.time_interpolation(ta, tc)
        self.assertTrue(np.all(finterp(ta) == tb))

    def test_load_encoder_positions(self):
        raw.load_encoder_positions(self.training_lt5_path,
                                   settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        raw.load_encoder_positions(self.training_ge5_path)
        raw.load_encoder_positions(self.biased_lt5_path,
                                   settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        raw.load_encoder_positions(self.biased_ge5_path)

    def test_load_encoder_events(self):
        raw.load_encoder_events(self.training_lt5_path,
                                settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        raw.load_encoder_events(self.training_ge5_path)
        raw.load_encoder_events(self.biased_lt5_path,
                                settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        raw.load_encoder_events(self.biased_ge5_path)

    def tearDown(self):
        [x.unlink() for x in self.training_lt5_path.rglob('alf/*') if x.is_file()]
        [x.unlink() for x in self.biased_lt5_path.rglob('alf/*') if x.is_file()]
        [x.unlink() for x in self.training_ge5_path.rglob('alf/*') if x.is_file()]
        [x.unlink() for x in self.biased_ge5_path.rglob('alf/*') if x.is_file()]
        [x.rmdir() for x in self.training_lt5_path.rglob('alf/') if x.is_dir()]
        [x.rmdir() for x in self.biased_lt5_path.rglob('alf/') if x.is_dir()]
        [x.rmdir() for x in self.training_ge5_path.rglob('alf/') if x.is_dir()]
        [x.rmdir() for x in self.biased_ge5_path.rglob('alf/') if x.is_dir()]


if __name__ == "__main__":
    unittest.main(exit=False)
    print('.')

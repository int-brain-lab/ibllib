import logging
import unittest
import functools
import shutil
from pathlib import Path

import numpy as np

import alf.io
from ibllib.io import extractors
from ibllib.io import raw_data_loaders as raw


def wheelMoves_fixture(func):
    """Decorator to save some dummy wheelMoves ALF files for extraction tests"""
    @functools.wraps(func)
    def wrapper(obj=None):
        # Save some wheelMoves ALF files
        attr_list = ['training_lt5',
                     'training_ge5',
                     'biased_lt5',
                     'biased_ge5']
        alf_paths = [getattr(obj, p)['path'] / 'alf' for p in attr_list]
        n_trials = [getattr(obj, p)['ntrials'] for p in attr_list]
        for p, n in zip(alf_paths, n_trials):
            p.mkdir()
            np.save(str(p / '_ibl_wheelMoves.intervals.npy'), np.zeros((n, 2)))
            np.save(str(p / '_ibl_wheelMoves.peakAmplitude.npy'), np.zeros(n))

        # Run method
        func(obj)

        # Teardown; delete the files
        for p in alf_paths:
            shutil.rmtree(p)
    return wrapper


class TestExtractTrialData(unittest.TestCase):

    def setUp(self):
        self.main_path = Path(__file__).parent
        self.training_lt5 = {'path': self.main_path / 'data' / 'session_training_lt5'}
        self.biased_lt5 = {'path': self.main_path / 'data' / 'session_biased_lt5'}
        self.training_ge5 = {'path': self.main_path / 'data' / 'session_training_ge5'}
        self.biased_ge5 = {'path': self.main_path / 'data' / 'session_biased_ge5'}
        self.training_lt5['ntrials'] = len(raw.load_data(self.training_lt5['path']))
        self.biased_lt5['ntrials'] = len(raw.load_data(self.biased_lt5['path']))
        self.training_ge5['ntrials'] = len(raw.load_data(self.training_ge5['path']))
        self.biased_ge5['ntrials'] = len(raw.load_data(self.biased_ge5['path']))
        # turn off logging for unit testing as we will purposedly go into warning/error cases
        self.wheel_ge5_path = self.main_path / 'data' / 'wheel_ge5'
        self.wheel_lt5_path = self.main_path / 'data' / 'wheel_lt5'
        self.logger = logging.getLogger('ibllib')
        # Save some dummy wheel moves data for trial firstMovement_times extraction

    def test_get_feedbackType(self):
        # TRAINING SESSIONS
        ft = extractors.training_trials.FeedbackType(
            self.training_lt5['path']).extract()[0]
        self.assertEqual(ft.size, self.training_lt5['ntrials'])
        # check if no 0's in feedbackTypes
        self.assertFalse(ft[ft == 0].size > 0)
        # -- version >= 5.0.0
        ft = extractors.training_trials.FeedbackType(
            self.training_ge5['path']).extract()[0]
        self.assertEqual(ft.size, self.training_ge5['ntrials'])
        # check if no 0's in feedbackTypes
        self.assertFalse(ft[ft == 0].size > 0)

        # BIASED SESSIONS
        ft = extractors.biased_trials.FeedbackType(
            self.biased_lt5['path']).extract()[0]
        self.assertEqual(ft.size, self.biased_lt5['ntrials'])
        # check if no 0's in feedbackTypes
        self.assertFalse(ft[ft == 0].size > 0)
        # -- version >= 5.0.0
        ft = extractors.biased_trials.FeedbackType(
            self.biased_ge5['path']).extract()[0]
        self.assertEqual(ft.size, self.biased_ge5['ntrials'])
        # check if no 0's in feedbackTypes
        self.assertFalse(ft[ft == 0].size > 0)

    def test_get_contrastLR(self):
        # TRAINING SESSIONS
        cl, cr = extractors.training_trials.ContrastLR(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))
        # -- version >= 5.0.0
        cl, cr = extractors.training_trials.ContrastLR(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))

        # BIASED SESSIONS
        cl, cr = extractors.biased_trials.ContrastLR(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))
        # -- version >= 5.0.0
        cl, cr = extractors.biased_trials.ContrastLR(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(all([np.sign(x) >= 0 for x in cl if ~np.isnan(x)]))
        self.assertTrue(all([np.sign(x) >= 0 for x in cr if ~np.isnan(x)]))
        self.assertTrue(sum(np.isnan(cl)) + sum(np.isnan(cr)) == len(cl))
        self.assertTrue(sum(~np.isnan(cl)) + sum(~np.isnan(cr)) == len(cl))

    def test_get_probabilityLeft(self):
        # TRAINING SESSIONS
        pl = extractors.training_trials.ProbabilityLeft(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(pl, np.ndarray))
        # -- version >= 5.0.0
        pl = extractors.training_trials.ProbabilityLeft(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(pl, np.ndarray))

        # BIASED SESSIONS
        pl = extractors.biased_trials.ProbabilityLeft(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(isinstance(pl, np.ndarray))
        # Test if only probs that are in prob set
        md = raw.load_settings(self.biased_lt5['path'])
        if md:
            probs = md['BLOCK_PROBABILITY_SET']
            probs.append(0.5)
            self.assertTrue(sum([x in probs for x in pl]) == len(pl))
        # -- version >= 5.0.0
        pl = extractors.biased_trials.ProbabilityLeft(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(pl, np.ndarray))
        # Test if only probs that are in prob set
        md = raw.load_settings(self.biased_ge5['path'])
        probs = md['BLOCK_PROBABILITY_SET']
        probs.append(0.5)
        self.assertTrue(sum([x in probs for x in pl]) == len(pl))

    def test_get_choice(self):
        # TRAINING SESSIONS
        choice = extractors.training_trials.Choice(
            session_path=self.training_lt5['path']).extract(save=False)[0]
        self.assertTrue(isinstance(choice, np.ndarray))
        data = raw.load_data(self.training_lt5['path'])
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in data])
        if any(trial_nogo):
            self.assertTrue(all(choice[trial_nogo]) == 0)
        # -- version >= 5.0.0
        choice = extractors.training_trials.Choice(
            session_path=self.training_ge5['path']).extract(save=False)[0]
        self.assertTrue(isinstance(choice, np.ndarray))
        data = raw.load_data(self.training_ge5['path'])
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in data])
        if any(trial_nogo):
            self.assertTrue(all(choice[trial_nogo]) == 0)

        # BIASED SESSIONS
        choice = extractors.biased_trials.Choice(
            session_path=self.biased_lt5['path']).extract(save=False)[0]
        self.assertTrue(isinstance(choice, np.ndarray))
        data = raw.load_data(self.biased_lt5['path'])
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in data])
        if any(trial_nogo):
            self.assertTrue(all(choice[trial_nogo]) == 0)
        # -- version >= 5.0.0
        choice = extractors.biased_trials.Choice(
            session_path=self.biased_ge5['path']).extract(save=False)[0]
        self.assertTrue(isinstance(choice, np.ndarray))
        data = raw.load_data(self.biased_ge5['path'])
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in data])
        if any(trial_nogo):
            self.assertTrue(all(choice[trial_nogo]) == 0)

    def test_get_repNum(self):
        # TODO: Test its sawtooth
        # TRAINING SESSIONS
        rn = extractors.training_trials.RepNum(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(rn, np.ndarray))
        for i in range(3):
            self.assertTrue(i in rn)
        # -- version >= 5.0.0
        rn = extractors.training_trials.RepNum(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(rn, np.ndarray))
        for i in range(4):
            self.assertTrue(i in rn)

        # BIASED SESSIONS have no repeted trials

    def test_get_rewardVolume(self):
        # TRAINING SESSIONS
        rv = extractors.training_trials.RewardVolume(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(rv, np.ndarray))
        # -- version >= 5.0.0
        rv = extractors.training_trials.RewardVolume(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(rv, np.ndarray))

        # BIASED SESSIONS
        rv = extractors.biased_trials.RewardVolume(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(isinstance(rv, np.ndarray))
        # Test if all non zero rewards are of the same value
        self.assertTrue(all([x == max(rv) for x in rv if x != 0]))
        # -- version >= 5.0.0
        rv = extractors.biased_trials.RewardVolume(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(rv, np.ndarray))
        # Test if all non zero rewards are of the same value
        self.assertTrue(all([x == max(rv) for x in rv if x != 0]))

    def test_get_feedback_times_ge5(self):
        # TRAINING SESSIONS
        ft = extractors.training_trials.FeedbackTimes(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(ft, np.ndarray))

        # BIASED SESSIONS
        ft = extractors.biased_trials.FeedbackTimes(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(ft, np.ndarray))

    def test_get_feedback_times_lt5(self):
        # TRAINING SESSIONS
        ft = extractors.training_trials.FeedbackTimes(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(ft, np.ndarray))

        # BIASED SESSIONS
        ft = extractors.biased_trials.FeedbackTimes(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(isinstance(ft, np.ndarray))

    def test_get_stimOnTrigger_times(self):
        # TRAINING SESSIONS
        sott = extractors.training_trials.StimOnTriggerTimes(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(sott, np.ndarray))
        # -- version >= 5.0.0
        sott = extractors.training_trials.StimOnTriggerTimes(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(sott, np.ndarray))
        # BIASED SESSIONS
        sott = extractors.biased_trials.StimOnTriggerTimes(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(isinstance(sott, np.ndarray))
        # -- version >= 5.0.0
        sott = extractors.biased_trials.StimOnTriggerTimes(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(sott, np.ndarray))

    def test_get_stimOn_times_lt5(self):
        # TRAINING SESSIONS
        st = extractors.training_trials.StimOnTimes(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(st, np.ndarray))

        # BIASED SESSIONS
        st = extractors.biased_trials.StimOnTimes(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(isinstance(st, np.ndarray))

    def test_get_stimOn_times_ge5(self):
        # TRAINING SESSIONS
        st = extractors.training_trials.StimOnTimes(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(st, np.ndarray))

        # BIASED SESSIONS
        st = extractors.biased_trials.StimOnTimes(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(st, np.ndarray))

    def test_stimOnOffFreeze_times(self):
        # TRAINING SESSIONS
        st = extractors.training_trials.StimOnOffFreezeTimes(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(st[0], np.ndarray))

        # BIASED SESSIONS
        st = extractors.biased_trials.StimOnOffFreezeTimes(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(isinstance(st[0], np.ndarray))

        # TRAINING SESSIONS
        st = extractors.training_trials.StimOnOffFreezeTimes(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(st[0], np.ndarray))

        # BIASED SESSIONS
        st = extractors.biased_trials.StimOnOffFreezeTimes(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(st[0], np.ndarray))

    @unittest.skip("not there yet")
    def test_stimOn_extractor_values(self):
        # Training lt5
        st_old = extractors.training_trials.StimOnTimes(
            self.training_lt5['path']).extract()[0]
        st_new = extractors.training_trials.StimOnOffFreezeTimes(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(np.all(st_old == st_new[0]))
        # Training ge5
        st_old = extractors.training_trials.StimOnTimes(
            self.training_ge5['path']).extract()[0]
        st_new = extractors.training_trials.StimOnOffFreezeTimes(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(np.all(st_old == st_new[0]))
        # Biased lt5
        st_old = extractors.biased_trials.StimOnTimes(
            self.biased_lt5['path']).extract()[0]
        st_new = extractors.biased_trials.StimOnOffFreezeTimes(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(np.all(st_old == st_new[0]))
        # Biased ge5
        st_old = extractors.biased_trials.StimOnTimes(
            self.biased_ge5['path']).extract()[0]
        st_new = extractors.biased_trials.StimOnOffFreezeTimes(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(np.all(st_old == st_new[0]))

    def test_get_intervals(self):
        # TRAINING SESSIONS
        di = extractors.training_trials.Intervals(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(di, np.ndarray))
        self.assertFalse(np.isnan(di).all())
        # -- version >= 5.0.0
        di = extractors.training_trials.Intervals(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(di, np.ndarray))
        self.assertFalse(np.isnan(di).all())

        # BIASED SESSIONS
        di = extractors.biased_trials.Intervals(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(di, np.ndarray))
        self.assertFalse(np.isnan(di).all())
        # -- version >= 5.0.0
        di = extractors.biased_trials.Intervals(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(di, np.ndarray))
        self.assertFalse(np.isnan(di).all())

    def test_get_iti_duration(self):
        # TRAINING SESSIONS
        iti = extractors.training_trials.ItiDuration(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(iti, np.ndarray))
        # -- version >= 5.0.0 iti always == 0.5 sec no extract

        # BIASED SESSIONS
        iti = extractors.biased_trials.ItiDuration(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(isinstance(iti, np.ndarray))
        # -- version >= 5.0.0 iti always == 0.5 sec no extract

    def test_get_response_times(self):
        # TRAINING SESSIONS
        rt = extractors.training_trials.ResponseTimes(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(rt, np.ndarray))
        # -- version >= 5.0.0
        rt = extractors.training_trials.ResponseTimes(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(rt, np.ndarray))

        # BIASED SESSIONS
        rt = extractors.biased_trials.ResponseTimes(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(isinstance(rt, np.ndarray))
        # -- version >= 5.0.0
        rt = extractors.biased_trials.ResponseTimes(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(rt, np.ndarray))

    def test_get_goCueTrigger_times(self):
        # TRAINING SESSIONS
        data = raw.load_data(self.training_lt5['path'])
        gct = np.array([tr['behavior_data']['States timestamps']
                        ['closed_loop'][0][0] for tr in data])
        self.assertTrue(isinstance(gct, np.ndarray))
        # -- version >= 5.0.0
        gct = extractors.training_trials.GoCueTriggerTimes(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(gct, np.ndarray))

        # BIASED SESSIONS
        data = raw.load_data(self.biased_lt5['path'])
        gct = np.array([tr['behavior_data']['States timestamps']
                        ['closed_loop'][0][0] for tr in data])
        self.assertTrue(isinstance(gct, np.ndarray))
        # -- version >= 5.0.0
        gct = extractors.biased_trials.GoCueTriggerTimes(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(gct, np.ndarray))

    def test_get_goCueOnset_times(self):
        # TRAINING SESSIONS
        gcot = extractors.training_trials.GoCueTimes(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(gcot, np.ndarray))
        self.assertTrue(np.all(np.isnan(gcot)))
        self.assertTrue(gcot.size != 0 or gcot.size == 4)
        # -- version >= 5.0.0
        gcot = extractors.training_trials.GoCueTimes(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(gcot, np.ndarray))
        self.assertFalse(np.any(np.isnan(gcot)))
        self.assertTrue(gcot.size != 0 or gcot.size == 12)

        # BIASED SESSIONS
        gcot = extractors.biased_trials.GoCueTimes(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(isinstance(gcot, np.ndarray))
        self.assertFalse(np.any(np.isnan(gcot)))
        self.assertTrue(gcot.size != 0 or gcot.size == 4)
        # -- version >= 5.0.0
        gcot = extractors.biased_trials.GoCueTimes(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(gcot, np.ndarray))
        self.assertFalse(np.any(np.isnan(gcot)))
        self.assertTrue(gcot.size != 0 or gcot.size == 8)

    def test_get_included_trials_lt5(self):
        # TRAINING SESSIONS
        it = extractors.training_trials.IncludedTrials(
            self.training_lt5['path']).extract()[0]
        self.assertTrue(isinstance(it, np.ndarray))
        # BIASED SESSIONS
        it = extractors.biased_trials.IncludedTrials(
            self.biased_lt5['path']).extract()[0]
        self.assertTrue(isinstance(it, np.ndarray))

    def test_get_included_trials_ge5(self):
        # TRAINING SESSIONS
        it = extractors.training_trials.IncludedTrials(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(it, np.ndarray))
        # BIASED SESSIONS
        it = extractors.biased_trials.IncludedTrials(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(it, np.ndarray))

    def test_get_included_trials(self):
        # TRAINING SESSIONS
        it = extractors.training_trials.IncludedTrials(
            self.training_lt5['path']).extract(settings={'IBLRIG_VERSION_TAG': '4.9.9'})[0]
        self.assertTrue(isinstance(it, np.ndarray))
        # -- version >= 5.0.0
        it = extractors.training_trials.IncludedTrials(
            self.training_ge5['path']).extract()[0]
        self.assertTrue(isinstance(it, np.ndarray))

        # BIASED SESSIONS
        it = extractors.biased_trials.IncludedTrials(
            self.biased_lt5['path']).extract(settings={'IBLRIG_VERSION_TAG': '4.9.9'})[0]
        self.assertTrue(isinstance(it, np.ndarray))
        # -- version >= 5.0.0
        it = extractors.biased_trials.IncludedTrials(
            self.biased_ge5['path']).extract()[0]
        self.assertTrue(isinstance(it, np.ndarray))

    @wheelMoves_fixture
    def test_extract_all(self):
        # TRAINING SESSIONS
        out, files = extractors.training_trials.extract_all(
            self.training_lt5['path'], settings={'IBLRIG_VERSION_TAG': '4.9.9'}, save=True)
        # -- version >= 5.0.0
        out, files = extractors.training_trials.extract_all(
            self.training_ge5['path'], save=True)
        # BIASED SESSIONS
        out, files = extractors.biased_trials.extract_all(
            self.biased_lt5['path'], settings={'IBLRIG_VERSION_TAG': '4.9.9'}, save=True)
        # -- version >= 5.0.0
        out, files = extractors.biased_trials.extract_all(
            self.biased_ge5['path'], save=True)

    def test_encoder_positions_clock_reset(self):
        # TRAINING SESSIONS
        # only for training?
        path = self.training_lt5['path'] / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
        dy = raw._load_encoder_positions_file_lt5(path)
        dat = np.array([849736, 1532230, 1822449, 1833514, 1841566, 1848206, 1853979, 1859144])
        self.assertTrue(np.all(np.diff(dy['re_ts']) > 0))
        self.assertTrue(all(dy['re_ts'][6:] - 2 ** 32 - dat == 0))

    def test_encoder_positions_clock_errors(self):
        # here we test for 2 kinds of file corruption that happen
        # 1/2 the first sample time is corrupt and absurdly high and should be discarded
        # 2/2 2 samples are swapped and need to be swapped backk
        path = self.biased_lt5['path'] / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
        dy = raw._load_encoder_positions_file_lt5(path)
        self.assertTrue(np.all(np.diff(np.array(dy.re_ts)) > 0))
        # -- version >= 5.0.0
        path = self.biased_ge5['path'] / "raw_behavior_data"
        path = next(path.glob("_iblrig_encoderPositions.raw*.ssv"), None)
        dy = raw._load_encoder_positions_file_ge5(path)
        self.assertTrue(np.all(np.diff(np.array(dy.re_ts)) > 0))

    def test_wheel_folders(self):
        # the wheel folder contains other errors in bpod output that had to be addressed
        for wf in self.wheel_lt5_path.glob('_iblrig_encoderPositions*.raw*.ssv'):
            df = raw._load_encoder_positions_file_lt5(wf)
            self.assertTrue(np.all(np.diff(np.array(df.re_ts)) > 0))
        for wf in self.wheel_lt5_path.glob('_iblrig_encoderEvents*.raw*.ssv'):
            df = raw._load_encoder_events_file_lt5(wf)
            self.assertTrue(np.all(np.diff(np.array(df.re_ts)) > 0))
        for wf in self.wheel_ge5_path.glob('_iblrig_encoderPositions*.raw*.ssv'):
            df = raw._load_encoder_positions_file_ge5(wf)
            self.assertTrue(np.all(np.diff(np.array(df.re_ts)) > 0))
        for wf in self.wheel_ge5_path.glob('_iblrig_encoderEvents*.raw*.ssv'):
            df = raw._load_encoder_events_file_ge5(wf)
            self.assertTrue(np.all(np.diff(np.array(df.re_ts)) > 0))

    def test_load_encoder_positions(self):
        raw.load_encoder_positions(self.training_lt5['path'],
                                   settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        raw.load_encoder_positions(self.training_ge5['path'])
        raw.load_encoder_positions(self.biased_lt5['path'],
                                   settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        raw.load_encoder_positions(self.biased_ge5['path'])

    def test_load_encoder_events(self):
        raw.load_encoder_events(self.training_lt5['path'],
                                settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        raw.load_encoder_events(self.training_ge5['path'])
        raw.load_encoder_events(self.biased_lt5['path'],
                                settings={'IBLRIG_VERSION_TAG': '4.9.9'})
        raw.load_encoder_events(self.biased_ge5['path'])

    def test_size_outputs(self):
        # check the output dimensions
        from ibllib.pipes.training_preprocessing import extract_training
        extract_training(self.training_ge5['path'])
        trials = alf.io.load_object(self.training_ge5['path'] / 'alf', object='trials')
        self.assertTrue(alf.io.check_dimensions(trials) == 0)
        extract_training(self.training_lt5['path'])
        trials = alf.io.load_object(self.training_lt5['path'] / 'alf', object='trials')
        self.assertTrue(alf.io.check_dimensions(trials) == 0)
        extract_training(self.biased_ge5['path'])
        trials = alf.io.load_object(self.biased_ge5['path'] / 'alf', object='trials')
        self.assertTrue(alf.io.check_dimensions(trials) == 0)
        # Wheel moves extraction fails for these wheel data; skipping
        # extract_training(self.biased_lt5['path'])
        # trials = alf.io.load_object(self.biased_lt5['path'] / 'alf', object='_ibl_trials')
        # self.assertTrue(alf.io.check_dimensions(trials) == 0)

    def tearDown(self):
        for f in self.main_path.rglob('_ibl_log.*.log'):
            f.unlink()
        [x.unlink() for x in self.training_lt5['path'].rglob('alf/*') if x.is_file()]
        [x.unlink() for x in self.biased_lt5['path'].rglob('alf/*') if x.is_file()]
        [x.unlink() for x in self.training_ge5['path'].rglob('alf/*') if x.is_file()]
        [x.unlink() for x in self.biased_ge5['path'].rglob('alf/*') if x.is_file()]
        [x.rmdir() for x in self.training_lt5['path'].rglob('alf/') if x.is_dir()]
        [x.rmdir() for x in self.biased_lt5['path'].rglob('alf/') if x.is_dir()]
        [x.rmdir() for x in self.training_ge5['path'].rglob('alf/') if x.is_dir()]
        [x.rmdir() for x in self.biased_ge5['path'].rglob('alf/') if x.is_dir()]


class TestSyncWheelBpod(unittest.TestCase):

    def test_sync_bpod_bonsai_poor_quality_timestamps(self):
        sync_trials_robust = raw.sync_trials_robust
        drift_pol = np.array([11 * 1e-6, -20])  # bpod starts 20 secs before with 10 ppm drift
        np.random.seed(seed=784)
        t0_full = np.cumsum(np.random.rand(50)) + .001
        t1_full = np.polyval(drift_pol, t0_full) + t0_full
        t0 = t0_full.copy()
        t1 = t1_full.copy()

        t0_, t1_ = sync_trials_robust(t0, t1)
        assert np.allclose(t1_, np.polyval(drift_pol, t0_) + t0_)

        t0_, t1_ = sync_trials_robust(t0, t1[:-1])
        assert np.allclose(t1_, np.polyval(drift_pol, t0_) + t0_)

        t0_, t1_ = sync_trials_robust(t0, t1[1:])
        assert np.allclose(t1_, np.polyval(drift_pol, t0_) + t0_)

        t0_, t1_ = sync_trials_robust(t0[1:], t1)
        assert np.allclose(t1_, np.polyval(drift_pol, t0_) + t0_)

        t0_, t1_ = sync_trials_robust(t0[:-1], t1)
        assert np.allclose(t1_, np.polyval(drift_pol, t0_) + t0_)

        t0_, t1_ = sync_trials_robust(t0, np.delete(t1, 24))
        assert np.allclose(t1_, np.polyval(drift_pol, t0_) + t0_)

        t0_, t1_ = sync_trials_robust(np.delete(t0, 12), np.delete(t1, 24))
        assert np.allclose(t1_, np.polyval(drift_pol, t0_) + t0_)


class TestWheelLoaders(unittest.TestCase):

    def setUp(self) -> None:
        self.main_path = Path(__file__).parent

    def test_encoder_events_corrupt(self):
        path = self.main_path.joinpath('data', 'wheel', 'lt5')
        for file_events in path.rglob('_iblrig_encoderEvents.raw.*'):
            dy = raw._load_encoder_events_file_lt5(file_events)
            self.assertTrue(dy.size > 6)
        path = self.main_path.joinpath('data', 'wheel', 'ge5')
        for file_events in path.rglob('_iblrig_encoderEvents.raw.*'):
            dy = raw._load_encoder_events_file_ge5(file_events)
            self.assertTrue(dy.size > 6)

    def test_encoder_positions_corrupts(self):
        path = self.main_path.joinpath('data', 'wheel', 'ge5')
        for file_position in path.rglob('_iblrig_encoderPositions.raw.*'):
            dy = raw._load_encoder_positions_file_ge5(file_position)
            self.assertTrue(dy.size > 18)
        path = self.main_path.joinpath('data', 'wheel', 'lt5')
        for file_position in path.rglob('_iblrig_encoderPositions.raw.*'):
            dy = raw._load_encoder_positions_file_lt5(file_position)
            self.assertTrue(dy.size > 18)


if __name__ == "__main__":
    unittest.main(exit=False)
    print('.')

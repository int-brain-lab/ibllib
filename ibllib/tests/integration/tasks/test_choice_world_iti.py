"""Tests for the extraction and QC of choice world tasks with the fixed ITI (iblrig >= v8.30.0).

From iblrig v8.30.0 the inter-trial interval (ITI), defined as the period between stimulus offset
and the start of the next trial, equals the ITI_DELAY_SECS task parameter (1 s by default). In
habituationChoiceWorld the grey screen period was moved from the start to the end of each trial.
"""

from unittest import mock

import numpy as np
from iblutil.util import Bunch
from one.api import ONE

import ibllib.pipes.behavior_tasks as btasks
from ibllib.io.extractors.ephys_fpga import FpgaTrialsHabituation
from ibllib.io.extractors.mesoscope import TimelineTrialsHabituation
from ibllib.tests import base


class ITITemplate(base.IntegrationTest):
    required_files = ['training/8.30.0/2025-09-29/*']
    task_kwargs = {'collection': 'raw_task_data_00', 'protocol_number': 0}

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB, mode='local')
        self.root = self.data_path.joinpath('training', '8.30.0', '2025-09-29')

    def check_iti(self, trials, qc, expected=1.0):
        """Check the trial ITIs and the QC ITI metric."""
        iti = trials['intervals'][1:, 0] - trials['stimOff_times'][:-1]
        np.testing.assert_allclose(iti, expected, atol=0.05)
        check = f'_{qc.namespace}_iti_delays'
        np.testing.assert_allclose(qc.metrics[check][:-1], iti - expected)
        self.assertTrue(np.all(qc.passed[check][:-1]))
        self.assertTrue(np.isnan(qc.passed[check][-1]), 'last trial should be NaN')


class TestTrainingTrialsBpod(ITITemplate):
    def test_extraction(self):
        """Test trainingChoiceWorld extraction and QC with a 1 s ITI."""
        task = btasks.ChoiceWorldTrialsBpod(
            self.root / '002', one=self.one, protocol='_iblrig_tasks_trainingChoiceWorld', **self.task_kwargs
        )
        trials, _ = task.extract_behaviour(save=False)
        trials = {**trials, **trials.pop('table')}
        trials['intervals'] = np.c_[trials.pop('intervals_0'), trials.pop('intervals_1')]
        self.assertEqual(11, len(trials['intervals']))
        # The trial ends 0.5 s after the stimulus offset, followed by ~0.5 s before the next trial
        np.testing.assert_allclose(trials['intervals'][:, 1] - trials['itiIn_times'], 0.5, atol=1e-3)
        np.testing.assert_allclose(trials['intervals'][1:, 0] - trials['intervals'][:-1, 1], 0.5, atol=0.05)
        qc = task.run_qc(trials, update=False)
        self.check_iti(trials, qc)


class TestHabituationTrialsBpod(ITITemplate):
    def test_extraction(self):
        """Test habituationChoiceWorld extraction and QC with the v8.30.0 state machine."""
        task = btasks.HabituationTrialsBpod(
            self.root / '004', one=self.one, protocol='_iblrig_tasks_habituationChoiceWorld', **self.task_kwargs
        )
        trials, _ = task.extract_behaviour(save=False)
        self.assertCountEqual(task.extractor.var_names, trials.keys())
        # Unlike previous versions, no trials are dropped
        n_trials = len(task.extractor.bpod_trials)
        self.assertTrue(all(len(v) == n_trials for v in trials.values()))
        # The stimulus is not detected on the first trial
        expected = [np.nan, 33.8578, 45.6071, 56.9582]
        np.testing.assert_array_almost_equal(expected, trials['stimOn_times'][:4], decimal=4)
        # Check the event order: start <= stim on trigger < stim on < stim center < feedback < stim off < trial end
        events = [
            'stimOnTrigger_times',
            'stimOn_times',
            'goCueTrigger_times',
            'stimCenterTrigger_times',
            'stimCenter_times',
            'feedback_times',
            'stimOffTrigger_times',
            'stimOff_times',
        ]
        times = np.c_[trials['intervals'][:, 0], *(trials[k] for k in events), trials['intervals'][:, 1]][1:, :]
        self.assertTrue(np.all(np.diff(times, axis=1) >= 0))
        np.testing.assert_array_equal(trials['intervals'][:, 0], trials['stimOnTrigger_times'])
        np.testing.assert_array_equal(trials['stimOff_times'], trials['itiIn_times'])
        np.testing.assert_allclose(trials['feedback_times'] - trials['stimCenterTrigger_times'], 0.5, atol=1e-3)
        qc = task.run_qc(trials, update=False)
        self.check_iti(trials, qc)
        _, _, outcomes = qc.compute_session_status()
        self.assertEqual('PASS', outcomes['_task_00_iti_delays'].name)


class TestHabituationTrialsDAQ(ITITemplate):
    """Test habituationChoiceWorld v8.30.0 extraction with simulated DAQ sync."""

    drift, offset = -20e-6, 100.0
    """The simulated DAQ clock drift and offset relative to Bpod."""

    chmap = {'bpod': 7, 'frame2ttl': 12, 'audio': 15}

    def setUp(self) -> None:
        super().setUp()
        self.session_path = self.root / '004'
        task = btasks.HabituationTrialsBpod(
            self.session_path, one=self.one, protocol='_iblrig_tasks_habituationChoiceWorld', **self.task_kwargs
        )
        self.bpod_trials, _ = task.extract_behaviour(save=False)
        self.bpod_extractor = task.extractor
        self.sync = self._simulate_sync(self.bpod_extractor.bpod_trials)

    def _simulate_sync(self, raw_trials):
        """Simulate the DAQ sync fronts from the raw Bpod data."""
        fronts = []  # (time, polarity, channel)
        for tr in raw_trials:
            states = tr['behavior_data']['States timestamps']
            for state in ('stim_on', 'reward', 'exit_state'):  # states with Bpod BNC1 output high
                ((t0, t1),) = states[state]
                fronts += [(t0, 1, self.chmap['bpod']), (t1, -1, self.chmap['bpod'])]
            events = tr['behavior_data']['Events timestamps']
            for event, channel in (('BNC1', 'frame2ttl'), ('BNC2', 'audio')):
                fronts += [(t, 1, self.chmap[channel]) for t in events.get(f'{event}High', [])]
                fronts += [(t, -1, self.chmap[channel]) for t in events.get(f'{event}Low', [])]
        fronts = np.array(sorted(fronts))
        return Bunch({'times': self.to_daq(fronts[:, 0]), 'polarities': fronts[:, 1], 'channels': fronts[:, 2]})

    def to_daq(self, times):
        return times * (1 + self.drift) + self.offset

    def check_trials(self, extractor):
        """Check the extracted DAQ trial events match the Bpod trial events."""
        trials, _ = extractor.extract(save=False, sync=self.sync, chmap=self.chmap, task_collection='raw_task_data_00')
        self.assertEqual('itiIn_times', extractor.sync_field)
        _, ttls = extractor.get_bpod_event_times(self.sync, self.chmap)
        n_trials = len(self.bpod_trials['intervals'])
        for k in ('trial_start', 'valve_open', 'trial_end'):
            self.assertEqual(n_trials, len(ttls[k]), f'unexpected number of {k} TTLs')
        for k, v in self.bpod_trials.items():
            if k.endswith('_times') or k == 'intervals':
                with self.subTest(attribute=k):
                    np.testing.assert_allclose(trials[k], self.to_daq(v), atol=1e-6)
        return trials

    def test_nidq(self):
        extractor = FpgaTrialsHabituation(self.session_path, bpod_trials=self.bpod_trials, bpod_extractor=self.bpod_extractor)
        self.check_trials(extractor)

    def test_timeline(self):
        timeline = Bunch({'meta': {'inputs': []}})  # no reward_valve channel
        with mock.patch('ibllib.io.extractors.mesoscope.alfio.load_object', return_value=timeline):
            extractor = TimelineTrialsHabituation(
                self.session_path, bpod_trials=self.bpod_trials, bpod_extractor=self.bpod_extractor
            )
        # NB: The simulated valve TTLs coincide with the Bpod reward state onsets
        self.check_trials(extractor)

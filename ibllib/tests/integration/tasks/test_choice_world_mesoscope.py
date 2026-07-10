"""Tests for Timeline behaviour extraction for UCL mesoscope."""
import logging
import unittest
from unittest.mock import patch, MagicMock, ANY

import numpy as np

import one.alf.io as alfio
from one.api import ONE

from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsTimeline, HabituationTrialsTimeline
from ibllib.io.extractors import mesoscope

from ibllib.tests import base

_logger = logging.getLogger('ibllib')


class TestTimelineTrials(base.IntegrationTest):
    session_path = None
    required_files = ['mesoscope/test/2023-02-17/002']
    _writable_scope = 'test'

    def setUp(self) -> None:
        super().setUp()
        self.one = ONE(**base.TEST_DB)
        self.session_path = self.data_path.joinpath(self.required_files[0])

    def test_sync(self):
        # Mocking training wheel extractor as session doesn't have Bpod rotary encoder data
        with patch('ibllib.io.extractors.training_wheel.Wheel._extract') as mock:
            mock().__getitem__.return_value = np.zeros(7)  # n trials = 7
            task = ChoiceWorldTrialsTimeline(self.session_path, sync_collection='raw_sync_data',
                                             sync_namespace='timeline', collection='raw_task_data_00')
            task.one = ONE(**base.TEST_DB, mode='local')  # Don't try updating behaviour criterion
            self.assertFalse(task.run(), 'extraction task failed')

        # Check ALF trials
        trials = alfio.load_object(self.session_path / 'alf', 'trials')
        self.assertEqual(18, len(trials.keys()))
        expected = [[9.97294005, 24.00193085],
                    [24.52629002, 28.16019116],
                    [28.6754851, 32.94392776],
                    [33.46808532, 36.9309287]]
        with self.subTest(k='intervals'):
            np.testing.assert_array_almost_equal(expected, trials['intervals'][:4, :])
        expected = [20.903, 26.033, 30.826, 34.803, 39.257, 44.131, 53.225]
        with self.subTest(k='feedback_times'):
            np.testing.assert_array_almost_equal(expected, trials['feedback_times'])
        expected = [20.811, 25.892, 30.742, 34.731, 39.091, 43.992, 53.125]
        with self.subTest(k='firstMovement_times'):
            np.testing.assert_array_almost_equal(expected, trials['firstMovement_times'])

        # Check ALF wheel
        wheel = alfio.load_object(self.session_path / 'alf', 'wheel')
        expected = [0., 0.00153398, 0.00306796, 0.00460194, 0.00613592]
        np.testing.assert_array_almost_equal(expected, wheel['position'][:5])
        expected = [20.809, 20.811, 20.812, 20.813, 20.814]
        np.testing.assert_array_almost_equal(expected, wheel['timestamps'][:5])

    @patch('ibllib.io.extractors.mesoscope.plt')
    def test_get_wheel_positions(self, plt_mock):
        """Test for TimelineTrials.get_wheel_positions in ibllib.io.extractors.mesoscope."""
        # # NB: For now we're testing individual functions before we have complete data
        timeline_trials = mesoscope.TimelineTrials(self.session_path, sync_collection='raw_sync_data')
        # Check that we can extract the wheel as it's from a counter channel, instead of raw analogue input
        wheel, moves = timeline_trials.get_wheel_positions()
        self.assertCountEqual(['timestamps', 'position'], wheel.keys())
        self.assertCountEqual(['intervals', 'peakAmplitude', 'peakVelocity_times'], moves.keys())
        self.assertEqual(4090, len(wheel['timestamps']))
        np.testing.assert_array_almost_equal([20.809, 20.811, 20.812, 20.813, 20.814], wheel['timestamps'][:5])
        np.testing.assert_array_almost_equal([0., 0.00153398, 0.00306796, 0.00460194, 0.00613592], wheel['position'][:5])
        expected = [[20.811, 21.216], [25.892, 26.251], [30.742, 31.172], [32.161, 33.208], [34.731, 36.756]]
        np.testing.assert_array_almost_equal(expected, moves['intervals'][:5, :])
        # Check input validation
        self.assertRaises(ValueError, timeline_trials.get_wheel_positions, coding='x3')
        # Test display
        plt_mock.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
        timeline_trials.bpod_trials = {'wheel_position': np.zeros_like(wheel['position']),
                                       'wheel_timestamps': wheel['timestamps']}
        timeline_trials.bpod2fpga = lambda x: x
        timeline_trials.get_wheel_positions(display=True)
        plt_mock.subplots.assert_called()
        # The second axes should be a plot of extracted wheel positions
        ax0, ax1 = plt_mock.subplots.return_value[1]
        ax1.plot.assert_called()
        np.testing.assert_array_equal(ax1.plot.call_args_list[0].args[0], wheel['timestamps'])

    @patch('ibllib.io.extractors.mesoscope.plt')
    def test_get_valve_open_times(self, plt_mock):
        """Test for TimelineTrials.get_valve_open_times in ibllib.io.extractors.mesoscope."""
        timeline_trials = mesoscope.TimelineTrials(self.session_path, sync_collection='raw_sync_data')
        # No longer supporting extraction without driver TTLs
        with self.assertLogs(mesoscope._logger, level='WARNING'):
            out = timeline_trials.get_valve_open_times()
            self.assertEqual(2, len(out))
            self.assertTrue(all(isinstance(x, np.ndarray) and x.size == 0 for x in out))

        # Test with TTLs
        ttls = np.array([[26.033, 26.099], [30.826, 30.891], [34.803, 34.868], [44.131, 44.196], [53.225, 53.29 ]])
        # Above TTLS taken from this code:
        # sync, chmap = timeline_trials.load_sync()
        # evts = timeline_trials.get_bpod_event_times(sync, chmap)
        # ttls = evts[1]['valve_open']
        intervals, open_times = timeline_trials.get_valve_open_times(driver_ttls=ttls)
        expected = np.array([[26.033, 26.101], [30.826, 30.894], [34.803, 34.871], [44.131, 44.199], [53.225, 53.293]])
        np.testing.assert_array_almost_equal(expected, intervals)
        np.testing.assert_array_almost_equal(expected[:, 0], open_times)
        # Test display
        plt_mock.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
        intervals, open_times = timeline_trials.get_valve_open_times(display=True, driver_ttls=ttls)
        plt_mock.subplots.assert_called()
        # The second axes should be a plot of expected valve open times
        ax0, ax1 = plt_mock.subplots.return_value[1]
        ax1.plot.assert_called()
        np.testing.assert_array_equal(ax1.plot.call_args_list[1].args[0], open_times)

    @patch('ibllib.io.extractors.mesoscope.plt')
    def test_plot_timeline(self, plt_mock):
        """Test for ibllib.io.extractors.mesoscope.plot_timeline."""
        ax = MagicMock()
        plt_mock.subplots.return_value = (MagicMock(), [ax] * 19)
        timeline = alfio.load_object(self.session_path / 'raw_sync_data', 'DAQdata')
        fig, axes = mesoscope.plot_timeline(timeline)
        plt_mock.subplots.assert_called_with(19, 1, sharex=True)
        self.assertIs(ax, axes[0], 'failed to return figure axes')
        axes[0].set_ylabel.assert_called_with('syncEcho', rotation=ANY, fontsize=ANY)
        self.assertEqual(19, axes[0].set_ylabel.call_count)
        (x, y), _ = axes[0].plot.call_args
        np.testing.assert_array_equal(timeline['timestamps'], x)
        np.testing.assert_array_equal(timeline['raw'][:, 18], y)

        # Test with raw=False and channels
        ax.reset_mock(), plt_mock.reset_mock()
        channels = ['audio', 'bpod']
        fig, axes = mesoscope.plot_timeline(timeline, channels=channels, raw=False)
        self.assertEqual(2, axes[0].set_ylabel.call_count)
        axes[0].set_ylabel.assert_called_with('bpod', rotation=ANY, fontsize=ANY)
        ylabels = [x[0][0] for x in axes[0].set_ylabel.call_args_list]
        self.assertCountEqual(channels, ylabels)

        (x, y), _ = axes[0].plot.call_args
        self.assertEqual(56, len(x))
        self.assertCountEqual({-1, 1}, np.unique(y))

    def test_timeline2sync(self):
        """Test for ibllib.io.raw_daq_loaders.load_timeline_sync_and_chmap."""
        sync, chmap = mesoscope.load_timeline_sync_and_chmap(self.session_path / 'raw_sync_data', save=False)
        self.assertIsInstance(sync, dict)
        self.assertCountEqual(('times', 'channels', 'polarities'), sync.keys())
        expected = {
            'neural_frames': 3,
            'volume_counter': 4,
            'bpod': 10,
            'frame2ttl': 12,
            'left_camera': 13,
            'right_camera': 14,
            'belly_camera': 15,
            'audio': 16,
            'rotary_encoder': 17}
        self.assertDictEqual(expected, chmap)


class TestTimelineTrialsHabituation(base.IntegrationTest):
    """Test for HabituationTrialsTimeline task."""
    session_path = None
    required_files = ['mesoscope/SP065/2024-10-07/001']

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        self.session_path = self.data_path.joinpath(self.required_files[0])

    def test_extraction(self):
        """Test habituation data is correct.

        NB: In this session the stimulus on the first is indeed not displayed or detected, so we expect a NaN.
        """
        sync_kwargs = {'sync': 'nidq', 'sync_collection': 'raw_sync_data', 'sync_ext': 'npy', 'sync_namespace': 'timeline'}
        task_kwargs = {'protocol': '_iblrig_tasks_habituationChoiceWorld', 'collection': 'raw_task_data_00', 'protocol_number': 0}
        task = HabituationTrialsTimeline(self.session_path, one=self.one, **sync_kwargs, **task_kwargs)
        data, files = task.extract_behaviour(save=True)
        # Check expected output datasets
        task.get_signatures()
        self.assertEqual(12, len(files))
        self.assertEqual(12, len(task.output_files))
        self.assertEqual(17, len(data))
        for f in task.output_files:
            self.assertTrue(f.find_files(task.session_path)[0], f'File {f.glob_pattern} not found')

        # Check the data
        trials = alfio.AlfBunch(data)
        self.assertEqual(0, trials.check_dimensions)
        np.testing.assert_array_almost_equal([np.nan, 22.97, 37.737, 52.27, 63.136], trials.stimOn_times[:5])
        np.testing.assert_array_almost_equal([np.nan, 35.703, 50.254, 61.12, 70.387], trials.stimCenter_times[:5])
        np.all(trials.intervals[:, 1] >= trials.stimOff_times)
        self.assertTrue(np.greater_equal(trials.intervals[:, 1], trials.stimOff_times).all())
        for k in ('stimOn', 'stimCenter', 'stimOff'):
            # Check all non-NaN stim times are greater than the trigger times
            valid = ~np.isnan(trials[f'{k}_times'])
            correct = np.greater(
                trials[f'{k}_times'], trials[f'{k}Trigger_times'], where=valid, out=np.ones(valid.shape, dtype=bool))
            self.assertTrue(correct.all(), f'{sum(~correct)}/{len(correct)} {k} times are invalid')


if __name__ == '__main__':
    unittest.main()

"""Tests for ibllib.pipes.mesoscope_tasks module."""
import sys
import logging
import shutil
import tempfile
import unittest.mock
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY
import tarfile
from itertools import chain
from uuid import UUID

import numpy as np
import pandas as pd
import sparse

import one.alf.io as alfio
from one.alf.path import get_session_path
from one.api import ONE

from ibllib.pipes.mesoscope_tasks import (
    MesoscopeSync, MesoscopeFOV, MesoscopeRegisterSnapshots,
    MesoscopePreprocess, MesoscopeCompress, Provenance
)
from iblatlas.atlas import AllenAtlas
from ibllib.pipes.behavior_tasks import ChoiceWorldTrialsTimeline, HabituationTrialsTimeline
from ibllib.io.extractors import mesoscope
from ibllib.io.raw_daq_loaders import load_timeline_sync_and_chmap

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
        expected = [20.903, 26.056, 30.847, 34.824, 39.257, 44.153, 53.247]
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
        expected = [[20.811, 21.216], [25.892, 26.251], [30.742, 31.173], [32.161, 33.208], [34.731, 36.756]]
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
        expected = [[26.056, 26.099],
                    [30.847, 30.891],
                    [34.824, 34.868],
                    [44.153, 44.197],
                    [53.247, 53.29],
                    [66.295, np.nan]]
        np.testing.assert_array_almost_equal(expected, timeline_trials.get_valve_open_times())
        # Test display
        plt_mock.subplots.return_value = (MagicMock(), (MagicMock(), MagicMock()))
        open_times = timeline_trials.get_valve_open_times(display=True)
        plt_mock.subplots.assert_called()
        # The second axes should be a plot of expected valve open times
        ax0, ax1 = plt_mock.subplots.return_value[1]
        ax1.plot.assert_called()
        ax1.twinx.assert_called()
        ax2 = ax1.twinx()
        np.testing.assert_array_equal(ax2.plot.call_args_list[1].args[0], open_times[:, 1])

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


class TestMesoscopeFOV(base.IntegrationTest):
    session_path = None

    def setUp(self) -> None:
        self.one = ONE(**base.TEST_DB)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.session_path = Path(tmpdir.name, 'subject', '2020-01-01', '001')
        self.session_path.joinpath('alf').mkdir(parents=True)
        # Make some toy datasets
        self.n_pixels = 512  # Number of pixels xy pixels in each FOV
        self.n_fov = 2  # Number of fields of view
        self.n_roi = 128  # Number of ROIs (will be multiplied by FOV number)
        self.expected_roi_mlapdv = {}  # Save the expected extracted ROI MLAPDV coordinates
        self.offset = 5.  # Offset between pixel number and MLAPDV coordinate
        self.mean_img_mlapdv = dict.fromkeys(range(self.n_fov))
        self.mean_img_ids = dict.fromkeys(range(self.n_fov))
        for i in range(self.n_fov):
            (alf_path := self.session_path.joinpath('alf', f'FOV_{i:02}')).mkdir()
            # Mean image MLAPDV coordinates
            ml = np.tile(np.arange(self.n_pixels), (self.n_pixels, 1)).astype(float) + self.offset
            self.mean_img_mlapdv[i] = np.dstack([ml, ml.T, np.zeros_like(ml)])

            # Mean image brain location IDs (a grid of 32x32 brain locations)
            n_tiles = 32
            tile_sz = int(self.n_pixels / n_tiles)
            x = np.repeat(np.arange(tile_sz), n_tiles)
            y = np.repeat(np.r_[0, (2 ** np.arange(tile_sz) * tile_sz)[:-1]], n_tiles)
            self.mean_img_ids[i] = x + y[..., None]

            # mpciROIs.stackPos (evenly spaced along the diagonal)
            n_roi = self.n_roi * (i + 1)  # 2nd FOV has twice as many as first
            v = np.linspace(0, self.n_pixels - 1, n_roi).astype(int)
            roi_mlapdv = np.vstack([v, v, np.zeros_like(v)]).T
            self.expected_roi_mlapdv[i] = np.c_[roi_mlapdv[:, :2] + self.offset, roi_mlapdv[:, 2]]
            np.save(alf_path / 'mpciROIs.stackPos.npy', roi_mlapdv)
        # For now the meta only contains number of FOVs
        alf_path = self.session_path.joinpath('raw_imaging_data')
        alf_path.mkdir()
        with open(alf_path / '_ibl_rawImagingData.meta.json', 'w') as fp:
            fp.write('{"nFrames": 2000, "FOV":[%s]}' % ','.join(['{}'] * self.n_fov))

    def test_mesoscope_fov(self):
        """Test for MesoscopeFOV._run and MesoscopeFOV.roi_mlapdv methods.

        This stubs both register_fov and project_mlapdv, which are tested separately.
        """
        # Test generation of mpciROI datasets
        task = MesoscopeFOV(self.session_path, device_collection='raw_imaging_data', one=self.one)
        mean_img_map = (self.mean_img_mlapdv, self.mean_img_ids)
        with patch.object(task, 'register_fov') as mock_obj, \
                patch.object(task, 'project_mlapdv', return_value=mean_img_map):
            self.assertEqual(0, task.run())
            mock_obj.assert_called_once_with(unittest.mock.ANY, 'estimate')
        self.assertEqual(self.n_fov * 4 + 1, len(task.outputs))  # + 1 for modified meta file
        # Mean image brain locations should be int
        file = next(f for f in task.outputs if 'mpciMeanImage.brainLocationIds_ccf_2017_estimate' in f.name)
        self.assertIs(np.load(file).dtype, np.dtype('int'))
        # Check ROI MLAPDV and brain locations
        rois = alfio.load_object(self.session_path / 'alf' / 'FOV_00', 'mpciROIs')
        expected = {'brainLocationIds_ccf_2017_estimate', 'mlapdv_estimate', 'stackPos'}
        self.assertCountEqual(expected, rois.keys())
        expected = self.expected_roi_mlapdv[0]
        np.testing.assert_array_equal(expected, rois['mlapdv_estimate'])
        expected = np.repeat(np.array([0, 17, 34, 67]), 8)
        self.assertIs(rois['brainLocationIds_ccf_2017_estimate'].dtype, np.dtype(int))
        np.testing.assert_array_equal(expected, rois['brainLocationIds_ccf_2017_estimate'][:32])

        # Test that we preferentially use the final coordinates
        # Copy data from another FOV and use as final
        for file in self.session_path.joinpath('alf', 'FOV_01').glob('mpciMeanImage.*'):
            file = file.replace(file.with_name(file.name.replace('_estimate', '')))
            shutil.copy(file, self.session_path.joinpath('alf', 'FOV_00', file.name))

        task = MesoscopeFOV(self.session_path, device_collection='raw_imaging_data', one=self.one)
        with patch.object(task, 'register_fov') as mock_obj, \
                patch.object(task, 'project_mlapdv', return_value=mean_img_map):
            self.assertEqual(0, task.run(provenance=Provenance.HISTOLOGY))
            mock_obj.assert_called_once_with(unittest.mock.ANY, None)
        self.assertEqual((self.n_fov * 4) + 1, len(task.outputs))  # + 1 for modified meta file
        self.assertFalse(any('_estimate' in x.name for x in task.outputs))
        rois = alfio.load_object(self.session_path / 'alf' / 'FOV_00', 'mpciROIs')
        expected = {'brainLocationIds_ccf_2017', 'mlapdv', 'stackPos'}
        self.assertTrue(expected <= set(rois.keys()))

        # Check behaviour when there are incomplete datasets
        self.session_path.joinpath('alf', 'FOV_00', 'mpciROIs.stackPos.npy').unlink()
        self.assertRaises(FileNotFoundError, task.roi_mlapdv, self.n_fov)


class TestProjectFOV(base.IntegrationTest):
    """Test MesoscopeFOV.project_mlapdv method."""
    session_path = None

    def setUp(self) -> None:
        # Load fixtures and create simple meta map
        self.session_path = Path('subject', '2020-01-01', '001')
        self.n_pixels = 64  # Number of pixels xy pixels in each FOV
        self.n_fov = 2  # Number of fields of view

        self.atlas = AllenAtlas(res_um=50)  # Use low res atlas for speed
        self.one = ONE(**base.TEST_DB, mode='local')

        # Create a toy meta file
        self.meta = {'centerMM': {'ML': 2.6, 'AP': -1.9}}
        MM = {'topLeft': [2.307, -1.607], 'topRight': [2.892, -1.607],
              'bottomLeft': [2.30, -2.193], 'bottomRight': [2.893, -2.193]}
        self.meta['FOV'] = [{'nXnYnZ': [self.n_pixels, self.n_pixels, 1], 'MM': MM}] * self.n_fov

    def test_project_mlapdv(self):
        """Test the full MesoscopeFOV.project_mlapdv method."""
        # Test generation of mpciROI datasets
        task = MesoscopeFOV(self.session_path, device_collection='raw_imaging_data', one=self.one)
        mlapdv, ids = task.project_mlapdv(self.meta, self.atlas)

        # Check MLAPDV coordinates
        self.assertCountEqual(mlapdv.keys(), range(self.n_fov))
        self.assertEqual(mlapdv[0].shape, (self.n_pixels, self.n_pixels, 3))
        # NB: Both FOVs will have the same values as the corner coords were duplicated
        expected = [
            [[2309.19916861, -1601.44040887, -231.35034825],
             [2317.83114255, -1601.89273938, -234.74282709],
             [2326.4631165, -1602.34506989, -238.13530593]],
            [[2309.09588003, -1610.65498922, -230.0804221],
             [2317.72972769, -1611.10741792, -233.47363734],
             [2326.36357535, -1611.55984662, -236.86685258]],
            [[2308.99259145, -1619.86956957, -228.81049596],
             [2317.62831283, -1620.32209646, -232.20444759],
             [2326.26403421, -1620.77462334, -235.59839922]]
        ]
        np.testing.assert_array_almost_equal(mlapdv[0][:3, :3, :], expected)

        # Check brain location IDs
        expected = [[1006, 981, 981],
                    [312782550, 981, 981],
                    [312782550, 981, 981]]
        np.testing.assert_array_almost_equal(ids[0][:3, 49:52], expected)
        self.assertCountEqual(ids.keys(), range(self.n_fov))
        self.assertEqual(ids[0].shape, (self.n_pixels, self.n_pixels))

        # Check meta map was modified
        FOV_00 = self.meta['FOV'][0]
        self.assertTrue(set(FOV_00.keys()) >= {'MLAPDV', 'brainLocationIds'})
        expected = {'topLeft': 312782550, 'topRight': 981, 'bottomLeft': 312782550,
                    'bottomRight': 312782604, 'center': 312782550}
        self.assertDictEqual(FOV_00['brainLocationIds'], expected)
        expected = [2575.3890558071657, -1901.209002390902, -297.8571573244117]
        np.testing.assert_array_almost_equal(FOV_00['MLAPDV']['center'], expected)

        # Test behaviour when outside of the brain (also remove one of the FOVs for speed)
        FOV_00 = self.meta['FOV'].pop()
        for k in FOV_00['MM']:
            FOV_00['MM'][k] = np.array(FOV_00['MM'][k]) + 10
        with self.assertLogs('ibllib.pipes.mesoscope_tasks', 'WARNING'):
            mlapdv, ids = task.project_mlapdv(self.meta, self.atlas)
        self.assertTrue(np.all(np.isnan(mlapdv[0])))
        np.testing.assert_array_equal(ids[0], np.zeros((self.n_pixels, self.n_pixels), dtype=int))


class TestMesoscopeSync(base.IntegrationTest):
    # session_path_0 = None  # A single imaging bout
    # session_path_1 = None  # Multiple imaging bouts
    # session_path_2 = None  # Multiple depths
    required_files = ['mesoscope/test/2023-02-17/002', 'mesoscope/test/2023-03-03/002', 'mesoscope/SP061/2025-02-26/001']
    _writable_scope = 'class'

    def setUp(self) -> None:
        super().setUp()
        self.one = ONE(**base.TEST_DB)
        self.session_path_0 = self.data_path.joinpath('mesoscope', 'test', '2023-02-17', '002')
        self.session_path_1 = self.data_path.joinpath('mesoscope', 'test', '2023-03-03', '002')
        self.session_path_2 = self.data_path.joinpath('mesoscope', 'SP061', '2025-02-26', '001')

    def test_single_depth(self):
        """Test for MesoscopeSync with single depth, single bout."""
        task = MesoscopeSync(self.session_path_0, device_collection='raw_imaging_data', one=self.one)
        status = task.run()
        assert status == 0

        # Check output
        nFOVs = 9
        alf_path = self.session_path_0.joinpath('alf')
        FOV_folders = sorted(filter(Path.is_dir, alf_path.rglob('FOV*')))
        self.assertEqual(nFOVs, len(FOV_folders))
        FOV_times = sorted(alf_path.rglob('mpci.times.npy'))
        self.assertEqual(nFOVs, len(FOV_times))
        expected = [1.106, 1.304, 1.503, 1.701, 1.899]
        np.testing.assert_array_almost_equal(np.load(FOV_times[0])[:5], expected)
        FOV_shifts = sorted(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nFOVs, len(FOV_shifts))
        expected = [0., 4.157940e-05, 8.315880e-05, 1.247382e-04, 1.663176e-04]
        np.testing.assert_array_almost_equal(np.load(FOV_shifts[0])[:5], expected)

        # Test what happens when there are more frame TTLs than timestamps in the header file
        extractor = mesoscope.MesoscopeSyncTimeline(self.session_path_0, nFOVs)
        n_frames = 336
        sync = {'times': np.arange(n_frames + 5), 'channels': np.zeros(n_frames + 5)}
        # For the purposes of this test these two channels can be the same
        # (the values would be identical for single plane data anyway)
        chmap = {'neural_frames': 0, 'volume_counter': 0}
        with self.assertLogs(mesoscope.__name__) as log:
            out, _ = extractor.extract(sync=sync, chmap=chmap)
            self.assertEqual('WARNING', log.records[0].levelname, 'failed to log warning')
            self.assertIn('Dropping last 5 frame times', log.output[-1])
        self.assertEqual({n_frames}, set(map(len, out[:nFOVs])), 'failed to drop timestamps')

    def test_multiple_bouts(self):
        """Test for MesoscopeSync with multiple imaging bouts."""
        task = MesoscopeSync(self.session_path_1, device_collection='raw_imaging_data*', one=self.one)
        status = task.run()
        assert status == 0

        # Check output
        nROIs = 6
        alf_path = self.session_path_1.joinpath('alf')
        ROI_folders = list(filter(Path.is_dir, alf_path.rglob('FOV*')))
        self.assertEqual(nROIs, len(ROI_folders))
        ROI_times = sorted(alf_path.rglob('mpci.times.npy'))
        self.assertEqual(nROIs, len(ROI_times))
        expected = [1.0075, 1.154, 1.3, 1.446, 1.5925]
        np.testing.assert_array_almost_equal(np.load(ROI_times[0])[:5], expected)
        ROI_shifts = sorted(alf_path.rglob('mpciStack.timeshift.npy'))
        self.assertEqual(nROIs, len(ROI_shifts))
        expected = [0., 4.157550e-05, 8.315100e-05, 1.247265e-04, 1.663020e-04]
        np.testing.assert_array_almost_equal(np.load(ROI_shifts[0])[:5], expected)

    def test_multi_depth(self):
        """Test for MesoscopeSync with multiple depths."""
        sync_path = self.session_path_2 / 'raw_sync_data'
        events = alfio.load_object(sync_path, 'softwareEvents').get('log')
        sync, chmap = load_timeline_sync_and_chmap(sync_path)
        collections = [f'raw_imaging_data_{i:02d}' for i in range(2)]
        nFOVs = 8
        mesosync = mesoscope.MesoscopeSyncTimeline(self.session_path_2, nFOVs)
        kwargs = dict(save=False, sync=sync, chmap=chmap, device_collection=collections, events=events)
        out, _ = mesosync.extract(use_volume_counter=False, **kwargs)

        # Check output
        # Times and line shifts for each FOV
        self.assertEqual(nFOVs * 2, len(out))
        mpci_times, mpciStack_timeshift = out[:nFOVs], out[nFOVs:]
        self.assertEqual({512}, set(map(len, mpciStack_timeshift)))
        self.assertEqual({17586}, set(map(len, mpci_times)))
        expected = [3624.5774065, 3624.7824065, 3624.9874065, 3625.1924065, 3625.3974065]
        np.testing.assert_array_almost_equal(mpci_times[5][-5:], expected)
        expected = [0.0211162, 0.02115785, 0.0211995, 0.02124115, 0.0212828]
        np.testing.assert_array_almost_equal(mpciStack_timeshift[0][-5:], expected)
        for shifts in mpciStack_timeshift[1:]:
            np.testing.assert_array_equal(mpciStack_timeshift[0], shifts)

        # Check extraction when using the volume counter instead of neural frames
        out, _ = mesosync.extract(use_volume_counter=True, **kwargs)
        mpci_times_volume, mpciStack_timeshift_volume = out[:nFOVs], out[nFOVs:]
        np.testing.assert_array_equal(mpciStack_timeshift[0], mpciStack_timeshift_volume[0])
        # The neural frame times should be close but not identical to the volume counter times
        for i, (neural_times, volume_times) in enumerate(zip(mpci_times, mpci_times_volume)):
            with self.subTest(FOV=i):
                np.testing.assert_array_almost_equal(neural_times, volume_times, decimal=3)

    @patch('ibllib.io.extractors.mesoscope.plt')
    def test_get_bout_edges(self, plt_mock):
        """Test for ibllib.io.extractors.mesoscope.MesoscopeSyncTimeline.get_bout_edges.

        This tests detection with and without the _ibl_softwareEvents.log.htsv file.
        """
        sync, chmap = load_timeline_sync_and_chmap(self.session_path_1 / 'raw_sync_data')
        extractor = mesoscope.MesoscopeSyncTimeline(self.session_path_1, 6)
        frame_times = sync['times'][sync['channels'] == chmap['neural_frames']]
        udp_events = self.session_path_1.joinpath('raw_sync_data', '_ibl_softwareEvents.log.htsv')
        events = pd.read_csv(udp_events, delimiter='\t')
        collections = ['raw_imaging_data_00', 'raw_imaging_data_01']
        bouts = extractor.get_bout_edges(frame_times, collections, events)
        np.testing.assert_array_equal(bouts, [[1.0075, 57.7175], [89.142, 132.5525]])

        # Test works with no end times
        bouts2 = extractor.get_bout_edges(frame_times, collections, events.drop([2, 4, 5]))
        np.testing.assert_array_equal(bouts, bouts2)

        # Test works with no events
        np.testing.assert_array_equal(bouts, extractor.get_bout_edges(frame_times, collections))

        # Test display
        plt_mock.subplots.return_value = (MagicMock(), MagicMock())
        extractor.get_bout_edges(frame_times, collections, events.drop([2, 4, 5]), display=True)
        plt_mock.subplots.assert_called()
        # Check plotted bout starts equal returned values
        ax = plt_mock.subplots.return_value[1]
        ax.plot.assert_called()
        plot_args = ax.plot.call_args_list[0]
        self.assertEqual('bout start', plot_args.kwargs['label'])
        bout_starts = np.unique(plot_args.args[0])
        np.testing.assert_array_equal(bout_starts[~np.isnan(bout_starts)], bouts2[:, 0])

        # Check validation
        collections.append('raw_imaging_data_02')
        self.assertRaises(ValueError, extractor.get_bout_edges, frame_times, collections, events)


class TestMesoscopeRegisterSnapshots(base.IntegrationTest):
    session_path = None
    one = None
    required_files = ['mesoscope/test/2023-03-03/002']
    reference_files = ['referenceImage.raw.tif', 'referenceImage.stack.tif', 'referenceImage.meta.json']

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.one = ONE(**base.TEST_DB)
        cls.session_path = cls.data_path.joinpath(cls.required_files[0])
        # Create some reference images to register
        for i in range(2):
            for file in cls.reference_files:
                p = cls.session_path.joinpath(f'raw_imaging_data_{i:02}', 'reference', file)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()

    def test_register_snapshots(self):
        """Test for MesoscopeRegisterSnapshots.

        NB: More thorough tests of register_snapshots exist in
          ibllib.tests.test_base_tasks.TestRegisterRawDataTask.test_register_snapshots
          ibllib.tests.test_pipes.TestRegisterRawDataTask.test_rename_files
        """
        task = MesoscopeRegisterSnapshots(self.session_path, one=self.one)
        eid = self.one.search()[0]
        with patch.object(self.one, 'path2eid', return_value=eid), \
                patch.object(task, 'register_snapshots') as reg_mock:
            status = task.run()
            reg_mock.assert_called_once_with(collection=['raw_imaging_data_??', ''])
        self.assertEqual(0, status)

    def test_get_signature(self):
        task = MesoscopeRegisterSnapshots(self.session_path, one=self.one)
        task.get_signatures()
        N = 2  # Number of raw_imaging_data collections
        n_input_files = len(list(chain.from_iterable(x.glob_pattern for x in task.input_files)))
        self.assertEqual(len(task.signature['input_files']) * N, n_input_files)
        n_output_files = len(list(chain.from_iterable(x.glob_pattern for x in task.output_files)))
        self.assertEqual(len(task.signature['output_files']) * N, n_output_files)


class TestMesoscopePreprocessRename(base.IntegrationTest):
    session_path = None
    required_files = ['mesoscope/SP037/2023-03-23/002']
    _writable_scope = 'test'

    """Test for MesoscopePreprocess task."""
    def setUp(self) -> None:
        self.session_path = self.data_path.joinpath(self.required_files[0])
        self.alf_path = self.session_path.joinpath('suite2p', 'plane2')
        self.rename_dict = {
            'F.npy': 'mpci.ROIActivityF.npy',
            'spks.npy': 'mpci.ROIActivityDeconvolved.npy',
            'Fneu.npy': 'mpci.ROINeuropilActivityF.npy'}
        # Copy files to another temp dir
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.suite2pdir = Path(self._tempdir.name).joinpath(*self.alf_path.parts[-6:])
        shutil.copytree(self.alf_path, self.suite2pdir)
        # Create a 'combined' folder which suite2p may create but should be ignored
        self.suite2pdir.joinpath('combined').mkdir()
        self.one = ONE(**base.TEST_DB)

    def test_rename_outputs(self):
        """Test MesoscopePreprocess._rename_outputs method."""
        session_path = get_session_path(self.suite2pdir)
        task = MesoscopePreprocess(session_path, one=self.one)
        files = task._rename_outputs(self.suite2pdir.parent, None, None)
        self.assertTrue(all(map(Path.exists, files)))
        self.assertTrue(self.suite2pdir.exists())
        # Check that other files were removed (note: there's no bin file in this test)
        self.assertEqual({'ops.npy'}, set(x.name for x in self.suite2pdir.rglob('*.*')))
        self.assertTrue((compressed := files[0].with_name('_suite2p_ROIData.raw.zip')).exists())
        self.assertIn(compressed, files)
        # Check files saved transposed
        for old, new in self.rename_dict.items():
            expected = np.load(self.alf_path / old).T
            np.testing.assert_array_equal(expected, np.load(files[0].with_name(new)))
        # Check frame QC not saved
        self.assertFalse(any('mpciFrameQC' in f.name for f in files))
        # Check sparse mask files
        sparse_files = sorted(f for f in files if f.suffix == '.sparse_npz')
        self.assertEqual(2, len(sparse_files))
        arr = sparse.load_npz(sparse_files[0])
        self.assertEqual((222, 512, 512), arr.shape)
        # Check first 10 non-zero elements of the first ROI
        mask = arr[0].todense()
        expected = [1.9042398, 2.0305383, 3.5443015, 4.247522, 3.14291, 2.286991,
                    3.8462281, 3.553623, 2.456772, 3.4159436]
        np.testing.assert_array_almost_equal(expected, mask[np.nonzero(mask)][:10])
        # Check ROI UUIDs were generated
        self.assertTrue((uuids_file := files[0].with_name('mpciROIs.uuids.csv')).exists())
        try:
            uuids = pd.read_csv(uuids_file).squeeze().apply(UUID)
        except ValueError as ex:
            self.assertFalse(True, f'failed to load and parse mpciROIs.uuids.csv: {ex}')
        expected_rois = 222
        self.assertEqual(uuids.size, expected_rois)
        self.assertEqual(uuids.nunique(), expected_rois)

    def test_rename_with_qc(self):
        """Test MesoscopePreprocess._rename_outputs method with frame QC input.

        Also tests behaviour if FOV folder(s) already exist, and that bin files are moved.
        """
        # Create an old FOV folder for it to remove
        session_path = get_session_path(self.suite2pdir)
        (fov_folder := session_path.joinpath('alf', 'FOV_02')).mkdir(parents=True)
        for name in self.rename_dict.values():
            fov_folder.joinpath(name).touch()
        # Should not delete other files from this folder
        fov_folder.joinpath('mpci.times.npy').touch()

        # Create binary data files
        self.suite2pdir.joinpath('data.bin').touch()
        self.suite2pdir.joinpath('data_raw.bin').touch()

        task = MesoscopePreprocess(session_path, one=self.one)

        frameQC_names = pd.DataFrame([(0, 'ok'), (1, 'foo')], columns=['qc_values', 'qc_labels'])
        with self.assertLogs('ibllib.pipes.mesoscope_tasks', 'DEBUG') as log:
            files = task._rename_outputs(self.suite2pdir.parent, frameQC_names, np.zeros(15))
        # Check old output files removed
        log_messages = (x.getMessage() for x in log.records)
        log_messages = [x for x in log_messages if x.startswith('Removing old file')]
        self.assertEqual(len(self.rename_dict), len(log_messages))
        self.assertNotIn('mpci.times.npy', ' '.join(log_messages))
        self.assertTrue(fov_folder.joinpath('mpci.times.npy').exists())
        # Check frameQC is saved
        self.assertIn(files[0].with_name('mpciFrameQC.names.tsv'), files)
        self.assertIn(files[0].with_name('mpci.mpciFrameQC.npy'), files)

        # bin files should be kept, the motion registered one should be renamed
        expected = {'ops.npy', 'imaging.frames_motionRegistered.bin', 'data_raw.bin'}
        self.assertEqual(expected, set(x.name for x in self.suite2pdir.rglob('*.*')))


class TestMesoscopePreprocess(base.IntegrationTest):
    session_path = None
    required_files = ['mesoscope/SP053/2024-02-07/001', 'mesoscope/SP037/2023-03-23/002/suite2p/plane2/ops.npy']

    """Test for MesoscopePreprocess task."""
    def setUp(self) -> None:
        self.session_path = self.default_data_root().joinpath('mesoscope', 'SP053', '2024-02-07', '001')
        # Copy files to temp dir
        # NB: suite2p dir now in session_path, not alf folder. This shouldn't affect these tests
        self.one = ONE(**base.TEST_DB)
        # Mock suite2p
        self.suite2p_mock = MagicMock()
        sys.modules['suite2p'] = self.suite2p_mock
        sys.modules['suite2p.io'] = self.suite2p_mock.io

    @patch.object(MesoscopePreprocess, 'image_motion_registration', autospec=True)
    @patch.object(MesoscopePreprocess, 'roi_detection', autospec=True)
    def test_run(self, roi_detection_mock, img_reg_mock):
        """Test MesoscopePreprocess._run method."""
        task = MesoscopePreprocess(self.session_path, one=self.one)

        # first create some raw bin data and assert that these are used instead of calling bin_per_plane
        n_planes = 2
        for i in range(n_planes):
            loc = self.session_path.joinpath('suite2p', f'plane{i}')
            loc.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.data_path.joinpath(self.required_files[1]), loc.joinpath('ops.npy'))
            loc.joinpath('imaging.frames_motionRegistered.bin').touch()

        task.get_signatures()
        files = task._run(rename_files=False)
        bad_frames_file = self.session_path.joinpath('raw_imaging_data_00').joinpath('bad_frames.npy')
        self.assertTrue(bad_frames_file.exists())
        bad_frames = np.load(bad_frames_file)
        expected = [2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237]
        np.testing.assert_array_equal(expected, bad_frames)
        self.suite2p_mock.io.mesoscan_to_binary.assert_not_called()
        self.assertEqual(n_planes, roi_detection_mock.call_count)
        self.assertEqual(n_planes, img_reg_mock.call_count)
        # Check that bin files were renamed
        self.assertFalse(self.session_path.joinpath('raw_bin_files').exists())
        expected = ['suite2p/plane0/data.bin', 'suite2p/plane0/ops.npy', 'suite2p/plane1/data.bin', 'suite2p/plane1/ops.npy']
        self.assertCountEqual(expected, [x.relative_to(self.session_path).as_posix() for x in files])

        # Test extraction of binary files and consolidation of QC
        shutil.rmtree(files[0].parents[1])
        self.suite2p_mock.io.mesoscan_to_binary.side_effect = self._create_planes
        # NB: rename_outputs tested elsewhere, here we mock it and check the inputs only
        with patch.object(task, 'get_default_tau', return_value=1.5), \
             patch.object(task, '_rename_outputs') as rename_outputs_mock:
            _ = task._run(rename_files=True)
        self.suite2p_mock.io.mesoscan_to_binary.assert_called()
        rename_outputs_mock.assert_called_once()
        (_, frame_qc_names, frame_qc), _ = rename_outputs_mock.call_args
        self.assertCountEqual(['ok', 'PMT off', 'galvos fault', 'high signal'], frame_qc_names.qc_labels)
        self.assertEqual(16328, frame_qc.size)
        np.testing.assert_array_equal(np.unique(frame_qc), [0, 1])
        expected = [2226, 2227, 2228, 2229, 2230, 2231, 2232, 2233, 2234, 2235, 2236, 2237]
        np.testing.assert_array_equal(np.where(frame_qc)[0], expected)

    @staticmethod
    def _create_planes(ops):
        """Save the ops files to the suite2p folder structure."""
        p = Path(ops['save_path0'], ops['save_folder'])
        p.mkdir(parents=True, exist_ok=True)
        for i in range(ops['nplanes']):
            p.joinpath(f'plane{i}').mkdir()
            np.save(p.joinpath(f'plane{i}', 'ops.npy'), ops)
        return ops


class TestMesoscopeCompress(base.IntegrationTest):
    """Test for MesoscopeCompress task."""

    def setUp(self) -> None:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)

        self.alf_path = Path(tempdir.name, 'test', '2023-03-03', '002', 'raw_imaging_data_00')
        self.alf_path.mkdir(parents=True)
        for i in range(2):
            with open(str(self.alf_path / f'2023-03-03_2_test_2P_00001_{i:05}.tif'), 'wb') as fp:
                np.save(fp, np.zeros((512, 512, 2), dtype=np.int16))

        # Touch some unnecessary files
        for name in ('_ibl_rawImagingData.meta.json', '2023-03-03_2_test_2P_00001_00001.mat'):
            self.alf_path.joinpath(name).touch()

        self.one = ONE(**base.TEST_DB)

    def test_compress(self):
        task = MesoscopeCompress(self.alf_path.parent, one=self.one)

        # Check fails if compressed file too small
        self.assertEqual(-1, task.run(remove_uncompressed=True))
        self.assertIn('Compressed file < 1KB', task.log)

        # Shouldn't unlink files if compression failed
        tif_files = list(self.alf_path.glob('*.tif'))
        self.assertEqual(2, len(tif_files), 'deleted tif files after failed compression')

        self.alf_path.joinpath('imaging.frames.tar.bz2').unlink()
        # With a mocked file size the task should complete
        status = task.run(verify_min_size=False, remove_uncompressed=True)
        self.assertFalse(status, 'compression task failed')

        self.assertTrue(self.alf_path.joinpath('imaging.frames.tar.bz2').exists())
        # Should delete the tifs after compression
        self.assertFalse(any(x.exists() for x in tif_files), 'failed to remove tifs')
        tfile = tarfile.open(self.alf_path.joinpath('imaging.frames.tar.bz2'))
        self.assertEqual(set(tfile.getnames()), set(x.name for x in tif_files))

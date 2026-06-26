"""Test passiveChoiceWorld extraction.

Basic passive extraction is tested using ephys/passive_extraction/SWC_054/2020-10-10/001.
Chained passive protocol extraction is tested with ephys/passive_extraction/ZFM-05496/2022-12-08/001.
Currently ZFM-05496/2022-12-08/001/raw_task_data_00 is not used.
"""
import unittest
import logging
import shutil

import pandas as pd
from pathlib import Path
import numpy as np
from packaging import version

from ibllib.io.extractors import ephys_passive
from ibllib.io.extractors import ephys_fpga
from ibllib.pipes.behavior_tasks import PassiveTaskNidq
from ibllib.tests import base

log = logging.getLogger('ibllib')


class TestLoadFixtures(base.IntegrationTest):

    def _check(self, task_replay, expected_sequence, n_repeat=1, n_removed=0):

        self.assertTrue(len(task_replay) == (300 * n_repeat) - n_removed)
        self.assertTrue(len(task_replay[task_replay['stim_type'] == 'G']) == (180 * n_repeat) - n_removed)
        self.assertTrue(len(task_replay[task_replay['stim_type'] == 'V']) == 40 * n_repeat)
        self.assertTrue(len(task_replay[task_replay['stim_type'] == 'T']) == 40 * n_repeat)
        self.assertTrue(len(task_replay[task_replay['stim_type'] == 'N']) == 40 * n_repeat)
        stims = task_replay['stim_type'].values
        self.assertListEqual(list(stims[0:10]), expected_sequence)

    def test_load_v7_fixtures(self):
        """ Test loading of iblrig v7 or less fixtures """
        settings = {
            'IBLRIG_VERSION': '7.6.0',
            'PRELOADED_SESSION_NUM': 3,
        }

        task_replay = ephys_passive._load_v7_fixture_df(settings)
        self._check(task_replay, ['G', 'N', 'T', 'G', 'G', 'G', 'G', 'N', 'G', 'T'])

    def test_load_v8_fixtures(self):
        """ Test loading of iblrig v8 fixtures """
        settings = {
            'IBLRIG_VERSION': '8.0.0',
            'SESSION_TEMPLATE_ID': 2,
        }

        task_replay = ephys_passive._load_v8_fixture_df(settings)
        self._check(task_replay, ['G', 'N', 'G', 'T', 'G', 'T', 'V', 'G', 'N', 'N'])

    def test_load_v8_fixtures_repeated(self):
        """ Test loading of iblrig v8 fixtures with repeated stimuli """
        settings = {
            'IBLRIG_VERSION': '8.2.0',
            'SESSION_TEMPLATE_ID': 4,
            'NUM_STIM_PRESENTATIONS': 900
        }

        task_replay = ephys_passive._load_v8_fixture_df(settings)
        self._check(task_replay, ['G', 'N', 'G', 'V', 'G', 'N', 'G', 'T', 'G', 'G'], n_repeat=3)

    def test_load_task_replay_fixtures(self):
        """
        Test the main function that selects between v7 and v8 loading

        For iblrig less than v8.2.9 it also removes the first Gabor stimulus and shifts the stimuli by one
        """

        sess_path = Path('tmp')
        collection = 'raw_task_data_00'

        # Check version less that less than 8.2.9, gabor are shifted and first gabor removed
        settings = {
            'IBLRIG_VERSION': '7.6.0',
            'PRELOADED_SESSION_NUM': 3,
        }
        task_replay_unshifted = ephys_passive._load_v7_fixture_df(settings)
        task_replay_shifted = ephys_passive.load_task_replay_fixtures(sess_path, collection, settings=settings)

        stims = task_replay_unshifted['stim_type'].values
        gabors = task_replay_unshifted[task_replay_unshifted['stim_type'] == 'G']
        stims_shifted = task_replay_shifted['stim_type'].values
        gabors_shifted = task_replay_shifted[task_replay_shifted['stim_type'] == 'G']

        self.assertListEqual(list(stims[0:5]), ['G', 'N', 'T', 'G', 'G'])
        self.assertListEqual(list(stims_shifted[0:5]), ['N', 'T', 'G', 'G', 'G'])
        self.assertEqual(len(stims) - 1, len(stims_shifted))
        np.testing.assert_array_equal(gabors['contrast'].values[:-1], gabors_shifted['contrast'].values)


class TestExtractPassivePeriods(base.IntegrationTest):
    def setUp(self):
        super().setUp()
        self.root_folder = self.data_path.joinpath('ephys', 'passive_extraction')
        self.session_path = self.root_folder.joinpath('SWC_054', '2020-10-10', '001')
        if not self.root_folder.exists():
            log.error(f'{self.root_folder} does not exist')

        self.sync, self.sync_map = ephys_fpga.get_sync_and_chn_map(self.session_path, 'raw_ephys_data')

    def test_get_spacers_good(self):
        """ Test that spacers are correctly identified in good data"""

        fttl = ephys_fpga.get_sync_fronts(self.sync, self.sync_map["frame2ttl"])
        ttl_signal = fttl['times']
        _, spacer_start, spacer_end = ephys_passive._get_spacer_times(ttl_signal, 0, self.sync['times'][-1])

        # Check that 3 spacers are found
        self.assertEqual(len(spacer_start), 3)

    def test_get_spacers_missing(self):
        """ Test that spacers missing raise value error"""
        fttl = ephys_fpga.get_sync_fronts(self.sync, self.sync_map["frame2ttl"])
        ttl_signal = fttl['times']
        idx = np.bitwise_and(ttl_signal > 4536, ttl_signal < 4544)
        ttl_signal = ttl_signal[~idx]
        with self.assertRaises(ValueError):
            _, spacer_start, spacer_end = ephys_passive._get_spacer_times(ttl_signal, 0, self.sync['times'][-1])


class TestExtractRFMapping(base.IntegrationTest):
    def setUp(self):
        super().setUp()
        self.root_folder = self.data_path.joinpath('ephys', 'passive_extraction')
        self.session_path = self.root_folder.joinpath('SWC_054', '2020-10-10', '001')
        if not self.root_folder.exists():
            log.error(f'{self.root_folder} does not exist')

        self.sync, self.sync_map = ephys_fpga.get_sync_and_chn_map(self.session_path, 'raw_ephys_data')

    def test_extract_rf_mapping(self):
        rf_times = ephys_passive.extract_rfmapping(self.session_path, sync_collection='raw_ephys_data',
                                                   task_collection='raw_passive_data', sync=self.sync,
                                                   sync_map=self.sync_map)

        self.assertEqual(len(rf_times), 17999)


class TestExtractTaskReplayComponents(base.IntegrationTest):
    def setUp(self):
        super().setUp()
        self.root_folder = self.data_path.joinpath('ephys', 'passive_extraction')
        self.session_path = self.root_folder.joinpath('SWC_054', '2020-10-10', '001')
        if not self.root_folder.exists():
            log.error(f'{self.root_folder} does not exist')

        self.sync, self.sync_map = ephys_fpga.get_sync_and_chn_map(self.session_path, 'raw_ephys_data')
        passivePeriods_df = ephys_passive.extract_passive_periods(
            self.session_path, 'raw_ephys_data', sync=self.sync, sync_map=self.sync_map)
        self.treplay = passivePeriods_df.taskReplay.values
        self.task_replay = ephys_passive.load_task_replay_fixtures(self.session_path, 'raw_passive_data')

    def test_extract_gabor(self):
        """ Test extraction of gabor stimuli on good data"""

        fttl = ephys_fpga.get_sync_fronts(self.sync, self.sync_map['frame2ttl'],
                                          tmin=self.treplay[0], tmax=self.treplay[-1])
        gabor_df = ephys_passive._extract_passive_gabor(fttl, self.task_replay)

        self.assertEqual(len(gabor_df), 179)
        expected_cols = ["stim_type", "start", "stop", "position", "contrast", "phase"]
        for col in expected_cols:
            self.assertIn(col, gabor_df.columns)

    def test_extract_audio(self):
        """ Test extraction of audio stimuli on good data"""

        audio = ephys_fpga.get_sync_fronts(self.sync, self.sync_map['audio'],
                                           tmin=self.treplay[0], tmax=self.treplay[-1])
        tone_df, noise_df = ephys_passive._extract_passive_audio(audio, self.task_replay, version.Version('8.0.0'))

        self.assertEqual(len(noise_df), 40)
        self.assertEqual(len(tone_df), 40)
        self.assertTrue(all(noise_df['stim_type'] == 'N'))
        self.assertTrue(all(tone_df['stim_type'] == 'T'))
        expected_cols = ["stim_type", "start", "stop"]
        for col in expected_cols:
            self.assertIn(col, noise_df.columns)
            self.assertIn(col, tone_df.columns)

    def test_extract_valve(self):
        """ Test extraction of valve stimuli on good data"""

        bpod = ephys_fpga.get_sync_fronts(self.sync, self.sync_map['bpod'],
                                          tmin=self.treplay[0], tmax=self.treplay[-1])
        valve_df = ephys_passive._extract_passive_valve(bpod, self.task_replay)

        self.assertEqual(len(valve_df), 40)
        expected_cols = ["stim_type", "start", "stop"]
        for col in expected_cols:
            self.assertIn(col, valve_df.columns)


class TestExtractTaskReplay(base.IntegrationTest):

    def setUp(self):
        super().setUp()
        self.root_folder = self.data_path.joinpath('ephys', 'passive_extraction')
        self.session_path = self.root_folder.joinpath('SWC_054', '2020-10-10', '001')
        if not self.root_folder.exists():
            log.error(f'{self.root_folder} does not exist')

        self.sync, self.sync_map = ephys_fpga.get_sync_and_chn_map(self.session_path, 'raw_ephys_data')
        passivePeriods_df = ephys_passive.extract_passive_periods(
            self.session_path, 'raw_ephys_data', sync=self.sync, sync_map=self.sync_map)
        self.treplay = passivePeriods_df.taskReplay.values
        self.task_replay = ephys_passive.load_task_replay_fixtures(self.session_path, 'raw_passive_data')

    def test_extract_replay(self):
        """ Test extraction of full task replay stimuli on good data"""

        gabor_df, stim_df = ephys_passive.extract_task_replay(
            self.session_path, sync_collection='raw_ephys_Data', task_collection='raw_passive_data',
            sync=self.sync, sync_map=self.sync_map, treplay=self.treplay)

        self.assertEqual(len(gabor_df), 179)
        expected_cols = ['start', 'stop', 'position', 'contrast', 'phase']
        for col in expected_cols:
            self.assertIn(col, gabor_df.columns)

        self.assertEqual(len(stim_df), 40)
        expected_cols = ['valveOn', 'valveOff', 'toneOn', 'toneOff', 'noiseOn', 'noiseOff']
        for col in expected_cols:
            self.assertIn(col, stim_df.columns)


class TestExtractReplayNoiseMissing(base.IntegrationTest):
    def setUp(self):
        super().setUp()
        self.root_folder = self.data_path.joinpath('ephys', 'passive_extraction')
        self.session_path = self.root_folder.joinpath('KM_020', '2024-08-07', '001')
        if not self.root_folder.exists():
            log.error(f'{self.root_folder} does not exist')

        self.sync, self.sync_map = ephys_fpga.get_sync_and_chn_map(self.session_path, 'raw_ephys_data')
        passivePeriods_df = ephys_passive.extract_passive_periods(
            self.session_path, 'raw_ephys_data', sync=self.sync, sync_map=self.sync_map)
        self.treplay = passivePeriods_df.taskReplay.values
        self.task_replay = ephys_passive.load_task_replay_fixtures(self.session_path, 'raw_task_data_01')

    def test_extract_replay_noise_missing(self):
        """ Test extraction of full task replay stimuli for specific case where no noise stimuli were presented,
        also Gabor fails"""

        gabor_df, stim_df = ephys_passive.extract_task_replay(
            self.session_path, sync_collection='raw_ephys_Data', task_collection='raw_task_data_01',
            sync=self.sync, sync_map=self.sync_map, treplay=self.treplay)

        self.assertIsNone(gabor_df)

        self.assertEqual(len(stim_df), 80)
        expected_cols = ['valveOn', 'valveOff', 'toneOn', 'toneOff', 'noiseOn', 'noiseOff']
        for col in expected_cols:
            self.assertIn(col, stim_df.columns)

        self.assertTrue(all(pd.isna(stim_df['noiseOn'])))
        self.assertEqual(pd.isna(stim_df['toneOn']).sum(), 0)
        self.assertEqual(pd.isna(stim_df['valveOn']).sum(), 40)


class TestEphysPassiveExtraction(base.IntegrationTest):

    required_files = ['ephys/passive_extraction/SWC_054/2020-10-10/001']
    _writable_scope = 'test'

    def setUp(self) -> None:
        super().setUp()
        self.root_folder = self.data_path.joinpath('ephys', 'passive_extraction')
        self.session_path = self.root_folder.joinpath('SWC_054', '2020-10-10', '001')
        if not self.root_folder.exists():
            log.error(f'{self.root_folder} does not exist')

    def test_task_extraction(self):
        ext = ephys_passive.PassiveChoiceWorld(self.session_path)
        data, paths = ext.extract()
        self.assertTrue(len(data) == 4)
        self.assertTrue(paths is None)

    def test_task_extraction_files(self):
        ext = ephys_passive.PassiveChoiceWorld(self.session_path)
        data, paths = ext.extract(save=True)
        path_names = [x.name for x in paths]
        expected = [
            '_ibl_passivePeriods.intervalsTable.csv',
            '_ibl_passiveRFM.times.npy',
            '_ibl_passiveGabor.table.csv',
            '_ibl_passiveStims.table.csv',
        ]
        self.assertTrue(all([x in path_names for x in expected]))

    def tearDown(self):
        # remove alf folder
        shutil.rmtree(self.session_path.joinpath('alf'), ignore_errors=True)


class TestChainedPassiveExtraction(base.IntegrationTest):
    """Test for the dynamic pipeline extraction of two chained passive protocols.

     Employs the ibllib.pipes.behavior_tasks.PassiveTaskNidq class.
     """

    required_files = ['ephys/passive_extraction/ZFM-05496/2022-12-08/001']
    _writable_scope = 'test'

    def setUp(self) -> None:
        super().setUp()
        self.root_folder = self.data_path.joinpath('ephys', 'passive_extraction')
        self.session_path = self.root_folder.joinpath('ZFM-05496', '2022-12-08', '001')
        if not self.session_path.exists():
            log.error(f'{self.root_folder} does not exist')

    def test_chained_passive_task_extraction(self):
        kwargs = {
            'collection': 'raw_task_data_01',
            'protocol': 'passiveChoiceWorld',
            'protocol_number': 1,
            'sync': 'nidq',
            'sync_collection': 'raw_ephys_data',
            'sync_ext': 'bin',
            'sync_namespace': 'spikeglx'
        }
        task = PassiveTaskNidq(self.session_path, location='local', **kwargs)
        self.assertEqual(0, task.run())
        self.assertEqual('alf/task_01', task.output_collection)
        self.assertEqual(4, len(task.output_files))
        out_path = self.session_path.joinpath(task.output_collection)
        # FIXME assert_expected_outputs will be called by Task.run in the future and this won't be necessary
        for dset in task.output_files:
            with self.subTest(dset):
                ok, files, _ = dset.find_files(self.session_path)
                self.assertTrue(ok)
        df = pd.read_csv(out_path / '_ibl_passivePeriods.intervalsTable.csv', index_col=0)
        expected = [3119.84190444, 4100.975761]
        np.testing.assert_array_almost_equal(df['passiveProtocol'].values, expected)

    def tearDown(self):
        # remove alf folder
        shutil.rmtree(self.session_path.joinpath('alf'), ignore_errors=True)


if __name__ == '__main__':
    unittest.main(exit=False)

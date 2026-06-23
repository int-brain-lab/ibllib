import unittest
import logging
import time
import tempfile
import shutil
from pathlib import Path
import json

import numpy as np
import numpy.testing

import ibllib.io.video as vidio
from one.api import ONE
from ci.tests import base
from ibllib.io.raw_daq_loaders import load_channels_tdms, correct_counter_discontinuities
from ibllib.io.raw_data_loaders import patch_settings, load_settings


class TestVideoIO(base.IntegrationTest):

    required_files = ['ephys/choice_world_init/KS022/2019-12-10/001/raw_video_data/_iblrig_leftCamera.raw.mp4']

    def setUp(self) -> None:
        root = self.data_path  # Path to integration data
        self.video_path = root.joinpath('ephys', 'choice_world_init', 'KS022', '2019-12-10',
                                        '001', 'raw_video_data', '_iblrig_leftCamera.raw.mp4')
        self.log = logging.getLogger('ibllib')

    def test_get_video_frame(self):
        n = 50  # Frame number to fetch
        frame = vidio.get_video_frame(self.video_path, n)
        expected_shape = (1024, 1280, 3)
        self.assertEqual(frame.shape, expected_shape)
        expected = np.array([[156, 222, 157, 75, 36, 15, 19, 20, 23]], dtype=np.uint8)
        np.testing.assert_array_equal(frame[:1, :9, 0], expected)

    def test_get_video_frames_preload(self):
        n = range(100, 103)  # Frame numbers to fetch

        # Test loading sequential frames without slice
        frames = vidio.get_video_frames_preload(self.video_path, n)
        expected_shape = (len(n), 1024, 1280, 3)
        self.assertEqual(frames.shape, expected_shape)
        self.assertEqual(frames.dtype, np.dtype(np.uint8))

        # Test loading frames with slice
        expected = np.array([[173, 133, 173, 216, 0],
                             [182, 133, 22, 241, 19],
                             [170, 152, 97, 48, 25]], dtype=np.uint8)
        frames = vidio.get_video_frames_preload(self.video_path, n, mask=np.s_[0, :5, 0])
        self.assertTrue(np.all(frames == expected))
        expected_shape = (len(n), 5)
        self.assertEqual(frames.shape, expected_shape)

        # Test loading frames as list
        frames = vidio.get_video_frames_preload(self.video_path, n, as_list=True)
        self.assertIsInstance(frames, list)
        self.assertEqual(frames[0].shape, (1024, 1280, 3))
        self.assertEqual(frames[0].dtype, np.dtype(np.uint8))
        self.assertEqual(len(frames), 3)

        # Test applying function
        frames = vidio.get_video_frames_preload(self.video_path, n,
                                                func=lambda x: np.mean(x, axis=2))
        expected_shape = (len(n), 1024, 1280)
        self.assertEqual(frames.shape, expected_shape)

    def test_get_video_frames_preload_perf(self):
        # Fetch x frames every 100 frames for y hundred frames total
        x = 5
        y = 3
        n = np.tile(np.arange(x * 10), (y, 1))
        n += np.arange(1, y * 100, 100).reshape(y, -1) - 1
        # Test loading sequential frames without slice
        t0 = time.time()
        vidio.get_video_frames_preload(self.video_path, n.flatten())
        elapsed = time.time() - t0
        self.log.info(f'fetching {n.size} frames with {y - 1} incontiguities took {elapsed:.2f}s')
        # self.assertLess(elapsed, 10, 'fetching frames took too long')

    def test_get_video_meta(self):
        # Check with local video path
        meta = vidio.get_video_meta(self.video_path)
        expected = {
            'length': 158377,
            'fps': 60,
            'width': 1280,
            'height': 1024,
            'size': 4257349100
        }
        self.assertTrue(expected.items() <= meta.items())
        self.assertEqual(meta.duration.total_seconds(), 2639.616667)

        # Check with remote path
        one = ONE(**base.TEST_DB)
        dset = one.alyx.rest('datasets', 'list', session='a41019f8-e430-473b-b22c-bb740dbecc33',
                             name='_iblrig_leftCamera.raw.mp4', exists=True)[0]
        video_url = next(fr['data_url'] for fr in dset['file_records'] if fr['data_url'])
        meta = vidio.get_video_meta(video_url, one=one)
        self.assertTrue(set(expected.keys()).issubset(meta.keys()))


class Read_DAQ_tdms(base.IntegrationTest):

    required_files = ['io/tdms_reader/20210421_daqami_analog.tdms', 'io/tdms_reader/20221102_daqami_digital.tdms']

    def setUp(self) -> None:
        root = self.data_path  # Path to integration data
        self.file_tdms_analog = root.joinpath('io/tdms_reader/20210421_daqami_analog.tdms')
        self.file_tdms_digital = root.joinpath('io/tdms_reader/20221102_daqami_digital.tdms')

    def test_read_tdms_analog_only(self):
        data, fs = load_channels_tdms(self.file_tdms_analog)
        self.assertEqual(set(data.keys()), set(f'AI{i}' for i in range(8)))
        self.assertEqual(fs, 1000)
        chmap = {'titi': 'AI0', 'tata': 'AI1'}
        dch, fs = load_channels_tdms(self.file_tdms_analog, chmap=chmap)
        np.testing.assert_array_equal(dch['tata'], data['AI1'])
        self.assertEqual(fs, 1000)

    def test_read_tdms_digital_only(self):
        data, _ = load_channels_tdms(self.file_tdms_digital)
        self.assertEqual(set(data.keys()), set(f'DI{i}' for i in range(2)))
        self.assertEqual(set(data[k].size for k in data.keys()), {2540244})
        chmap = {'bpod': 'DI0', 'frame2ttl': 'DI1'}
        data2, _ = load_channels_tdms(self.file_tdms_digital, chmap=chmap)
        np.testing.assert_equal(data2['bpod'], data['DI0'])
        np.testing.assert_equal(data2['frame2ttl'], data['DI1'])


class TestPatchSettings(base.IntegrationTest):
    required_files = ['Subjects_init/ZM_1085/2019-02-12/003/raw_task_data_00']

    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        src_path = self.data_path.joinpath('Subjects_init', 'ZM_1085', '2019-02-12', '003', 'raw_task_data_00')
        path = shutil.copytree(src_path, Path(self.tempdir.name).joinpath(*src_path.parts[-4:]))
        self.settings_file = next(path.glob('*Settings.raw*'))
        self.addCleanup(self.tempdir.cleanup)

    def test_patch_settings(self):
        """Test for ibllib.io.raw_data_loaders.patch_settings"""
        session_path = self.settings_file.parents[1]
        collection = self.settings_file.parts[-2]
        self.assertRaises(IOError, patch_settings, session_path, 'foobar')
        # # Get the old subject name
        settings = load_settings(session_path, collection)
        new_data = dict(subject='SUB_00', date='2020-01-01', number=9, new_collection='raw_task_data_04')
        new_settings = patch_settings(session_path, collection, **new_data)

        # Check output
        actual = (
            new_settings['SUBJECT_NAME'],
            new_settings['SESSION_DATE'],
            int(new_settings['SESSION_NUMBER']),
            new_settings['SESSION_RAW_DATA_FOLDER'].split('\\')[-1])
        self.assertCountEqual(new_data.values(), actual)
        self.assertNotIn('PYBPOD_SUBJECT_EXTRA', new_settings)

        # Checkout saved file
        with open(self.settings_file, 'r') as fp:
            new_raw_settings = fp.readlines()
        new_path = '\\\\'.join(map(lambda x: str(x).zfill(3), list(new_data.values())[:-1]))
        self.assertEqual(7, sum(new_path in ln for ln in new_raw_settings))
        old_path = '\\\\'.join([settings['SUBJECT_NAME'], settings['SESSION_DATE'], settings['SESSION_NUMBER']])
        self.assertFalse(any(old_path in ln for ln in new_raw_settings))

        # Test with v8-style settings
        settings.pop('SESSION_RAW_DATA_FOLDER')
        settings['SESSION_END_TIME'] = settings['SESSION_DATE'] + 'T11:15:30.064893'
        settings['SESSION_START_TIME'] = settings['SESSION_DATETIME']
        with open(self.settings_file, 'w') as fp:
            json.dump(settings, fp, indent=' ')
        new_data.update(new_collection='raw_task_data_03')
        with self.assertLogs('ibllib.io.raw_data_loaders', level=20):
            new_settings = patch_settings(session_path, collection, **new_data)
        self.assertTrue(new_settings['SESSION_END_TIME'].startswith(new_data['date']))
        self.assertTrue(new_settings['SESSION_START_TIME'].startswith(new_data['date']))


class TestDAQDiscontinuities(base.IntegrationTest):
    required_files = ['mesoscope/2023-03-03_1_SP035-re_pos.npy']

    def setUp(self) -> None:
        src_path = self.data_path.joinpath('mesoscope', '2023-03-03_1_SP035-re_pos.npy')
        self.re_pos = np.load(src_path)

    def test_correct_counter_discontinuities(self):
        """Test for ibllib.io.raw_daq_loaders.correct_counter_discontinuities"""
        expected = self.re_pos.copy()
        expected[10:] -= 4294967295
        np.testing.assert_array_equal(expected, correct_counter_discontinuities(self.re_pos))


if __name__ == '__main__':
    unittest.main()

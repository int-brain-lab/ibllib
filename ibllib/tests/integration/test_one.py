"""Tests for the VideoStreamer and oneibl.registration module.

Tests include registering chained protocols.
"""
from datetime import datetime
from tempfile import TemporaryDirectory
from pathlib import Path
import shutil
import string
import random
from unittest import mock

import numpy as np
from ibllib.io.video import VideoStreamer
import ibllib.oneibl.registration as reg
from one.api import ONE
import one.alf.exceptions as alferr

from ci.tests.base import IntegrationTest, TEST_DB


class TestVideoStreamer(IntegrationTest):
    def setUp(self) -> None:
        self.one = ONE(**TEST_DB)
        self.eid, = self.one.search(subject='ZM_1743', number=1, date_range='2019-06-14')

    def test_video_streamer(self):
        dset = self.one.alyx.rest('datasets', 'list',
                                  session=self.eid, name='_iblrig_leftCamera.raw.mp4')[0]
        url = next(fr['data_url'] for fr in dset['file_records'] if fr['data_url'])
        frame_id = 5
        vs = VideoStreamer(url)
        f, im = vs.get_frame(frame_id)
        assert f
        assert vs.total_frames == 144120
        # Test with data set dict
        vs = VideoStreamer(dset)
        f, im2 = vs.get_frame(frame_id)
        assert np.all(im == im2)


class TestRegistrationUtils(IntegrationTest):
    """Tests for ibllib.oneibl.registration functions"""
    session_path = None
    required_files = ['Subjects_init/ZM_1085/2019-02-12/003']

    @classmethod
    def setUpClass(cls) -> None:
        root = IntegrationTest.default_data_root()
        cls.session_path = root.joinpath('Subjects_init', 'ZM_1085', '2019-02-12', '003')
        bpod = [reg.raw.load_bpod(cls.session_path, f'raw_task_data_{x:02}') for x in range(2)]
        cls.settings, cls.data = list(zip(*bpod))

    def test_get_session_times(self):
        # Test with two protocols
        start, end = reg._get_session_times(self.session_path, self.settings, self.data)
        self.assertEqual(datetime(2019, 2, 12, 10, 14, 20, 47259), start)
        self.assertEqual(datetime(2019, 2, 12, 12, 27, 49, 741060), end)
        # Test with a single protocol
        start, end = reg._get_session_times(self.session_path, self.settings[1], self.data[1])
        self.assertEqual(datetime(2019, 2, 12, 11, 22, 23, 47259), start)
        self.assertEqual(datetime(2019, 2, 12, 12, 27, 49, 741060), end)

    def test_get_session_performance(self):
        # Test with two protocols
        n_trials, n_correct = reg._get_session_performance(self.settings, self.data)
        self.assertEqual(1838, n_trials)
        self.assertEqual(1578, n_correct)
        # Test with a single protocol
        n_trials, n_correct = reg._get_session_performance(self.settings[0], self.data[0])
        self.assertEqual(919, n_trials)
        self.assertEqual(789, n_correct)
        # Test with a empty session data
        n_trials, n_correct = reg._get_session_performance(self.settings, [self.data[0], None])
        self.assertEqual(919, n_trials)
        self.assertEqual(789, n_correct)
        # Test with habituation choice world
        settings = [self.settings[0].copy(), self.settings[1]]
        settings[0]['PYBPOD_PROTOCOL'] = 'habituationChoiceWorld'
        n_trials, n_correct = reg._get_session_performance(settings, self.data)
        self.assertEqual(1838, n_trials)
        self.assertEqual(789, n_correct)


class TestRegistration(IntegrationTest):
    """Tests for ibllib.oneibl.registration"""
    source_path = None
    required_files = ['Subjects_init/ZM_1085/2019-02-12/003']

    @classmethod
    def setUpClass(cls) -> None:
        root = IntegrationTest.default_data_root()
        cls.source_path = root.joinpath('Subjects_init', 'ZM_1085', '2019-02-12', '003')
        cls.subject = ''.join(random.choices(string.ascii_letters, k=10))

    def setUp(self) -> None:
        self.tempdir = TemporaryDirectory()
        self.session_path = Path(self.tempdir.name).joinpath(
            'Subjects_init', self.subject, *self.source_path.parts[-2:])
        shutil.copytree(self.source_path, self.session_path)
        one = ONE(cache_rest=None, **TEST_DB)
        self.client = reg.IBLRegistrationClient(one)
        one.alyx.rest('subjects', 'create', data={'lab': 'mainenlab', 'nickname': self.subject})
        self._patch_settings()

    def _patch_settings(self):
        """Change the subject name in all of the settings files"""
        for file in self.session_path.rglob('*_iblrig_taskSettings.raw.json'):
            with open(file, 'r') as f:
                # read a list of lines into data
                data = f.readlines()
            for i in range(len(data)):
                data[i] = data[i].replace('ZM_1085', self.subject)
            with open(file, 'w') as f:
                # write everything back
                f.writelines(data)

    def test_create_sessions(self):
        assert self.session_path.joinpath('create_me.flag').exists()
        (session_path,), (info,) = self.client.create_sessions(self.tempdir.name)
        self.assertFalse(session_path.joinpath('create_me.flag').exists())
        expected = {
            'users': ['ines'],
            'task_protocol': '_iblrig_tasks_biasedChoiceWorld3.7.0/_iblrig_tasks_biasedChoiceWorld3.7.0',
            'start_time': '2019-02-12T10:14:20.047259', 'end_time': '2019-02-12T12:27:49.741060',
            'n_correct_trials': 1578, 'n_trials': 1838
        }
        subset = {k: v for k, v in info.items() if k in expected}
        self.assertDictEqual(subset, expected)
        self.assertEqual(4, info['json'].get('POOP_COUNT'))
        # Check weights
        subj = self.client.one.alyx.rest('subjects', 'read', id=self.subject)
        self.assertEqual(2, len(subj['weighings']))
        actual = [x['date_time'] for x in subj['weighings']]
        self.assertCountEqual(['2019-02-12T10:14:20.047259', '2019-02-12T11:22:23.047259'], actual)
        self.assertCountEqual({x['weight'] for x in subj['weighings']}, [22.35])

        # Check water restrictions
        self.assertEqual(2, len(subj['water_administrations']))
        self.assertCountEqual({x['water_administered'] for x in subj['water_administrations']}, [2.367])
        self.assertCountEqual({x['session'] for x in subj['water_administrations']}, [info['id']])
        actual = [x['date_time'] for x in subj['water_administrations']]
        # sort the returned data
        actual.sort()
        self.assertEqual(['2019-02-12T11:19:46.741060', '2019-02-12T12:27:49.741060'], actual)

    def test_register_raw(self):
        # Touch a non-raw file
        alf_file = self.session_path.joinpath('alf', '_ibl_trials.intervals.npy')
        alf_file.parent.mkdir(), alf_file.touch()
        with self.assertRaises(alferr.ALFError):
            reg.register_session_raw_data(self.session_path, self.client.one)

        # Create session
        data = {k: v for k, v in zip(('subject', 'start_time', 'number'), self.session_path.parts[-3:])}
        data['start_time'] = self.client.ensure_ISO8601(data['start_time'])
        data['type'] = 'Base'
        data['users'] = [self.client.one.alyx.user]
        self.client.one.alyx.rest('sessions', 'create', data=data)
        globus_id = '2dc8ccc6-2f8e-11e9-9351-0e3d676669f4'  # 'mainen_lab_SR'
        with mock.patch('ibllib.oneibl.registration.get_local_endpoint_id', return_value=globus_id):
            files, recs = reg.register_session_raw_data(self.session_path, self.client.one)
        self.assertEqual(4, len(files))
        self.assertFalse(alf_file in files)
        self.assertEqual(2, len(recs[0]['file_records']))
        # self.assertEqual(ibllib_version, recs[0]['version'])  # version not exposed by endpoint
        self.assertFalse(any(r['exists'] for r in recs[0]['file_records']))
        expected = ['flatiron_mainenlab', 'mainen_lab_SR']
        self.assertCountEqual(expected, (r['data_repository'] for r in recs[0]['file_records']))

    def tearDown(self) -> None:
        one = self.client.one
        # Note: datasets, weights, water admins and sessions deleted in cascade
        one.alyx.rest('subjects', 'delete', id=self.subject)
        self.tempdir.cleanup()

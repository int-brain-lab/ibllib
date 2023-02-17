import unittest
import tempfile
from unittest import mock
from pathlib import PurePosixPath, Path
import json
import datetime
import random
import string

from requests import HTTPError
import numpy as np
from one.api import ONE
import iblutil.io.params as iopar

from ibllib.oneibl import patcher, registration
import ibllib.io.extractors.base
from ibllib import __version__
from ibllib.tests import TEST_DB
from ibllib.io import session_params


class TestFTPPatcher(unittest.TestCase):
    def setUp(self) -> None:
        self.one = ONE(**TEST_DB)

    def reset_params(self):
        """Remove the FTP parameters from the AlyxClient"""
        par = iopar.as_dict(self.one.alyx._par)
        self.one.alyx._par = iopar.from_dict({k: v for k, v in par.items()
                                              if not k.startswith('FTP')})

    @mock.patch('ftplib.FTP_TLS')
    def test_setup(self, _):
        self.reset_params()
        # Test silent setup (one instance is in silent mode)
        patcher.FTPPatcher(one=self.one)
        keys = ('FTP_DATA_SERVER', 'FTP_DATA_SERVER_LOGIN', 'FTP_DATA_SERVER_PWD')
        self.assertTrue(all(k in self.one.alyx._par.as_dict() for k in keys))
        # Silent mode off
        self.reset_params()
        self.one.alyx.silent = False
        with mock.patch('builtins.input', new=self.mock_input),\
                mock.patch('ibllib.oneibl.patcher.getpass', return_value='foobar'):
            patcher.FTPPatcher(one=self.one)
        self.assertEqual(self.one.alyx._par.FTP_DATA_SERVER_LOGIN, 'usr')
        self.assertEqual(self.one.alyx._par.FTP_DATA_SERVER_PWD, 'foobar')

    @staticmethod
    def mock_input(prompt):
        FTP_pars = {
            'FTP_DATA_SERVER': 'ftp://server.net',
            'FTP_DATA_SERVER_LOGIN': 'usr'}
        return FTP_pars[next(k for k in FTP_pars.keys() if k in prompt.replace(',', '').split())]


class TestAlyx2Path(unittest.TestCase):
    dset = {
        'url': 'https://alyx.internationalbrainlab.org/'
               'datasets/00059298-1b33-429c-a802-fa51bb662d72',
        'name': 'channels.localCoordinates.npy',
        'collection': 'alf/probe00',
        'session': ('https://alyx.internationalbrainlab.org/'
                    'sessions/7cffad38-0f22-4546-92b5-fd6d2e8b2be9'),
        'file_records': [
            {'id': 'c9ae1b6e-03a6-41c9-9e1b-4a7f9b5cfdbf', 'data_repository': 'ibl_floferlab_SR',
             'data_repository_path': '/mnt/s0/Data/Subjects/',
             'relative_path': 'SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy',
             'data_url': None, 'exists': True},
            {'id': 'f434a638-bc61-4695-884e-70fd1e521d60', 'data_repository': 'flatiron_hoferlab',
             'data_repository_path': '/hoferlab/Subjects/',
             'relative_path': 'SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy',
             'data_url': (
                 'https://ibl.flatironinstitute.org/hoferlab/Subjects/SWC_014/2019-12-11/001/'
                 'alf/probe00/channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy'),
             'exists': True}],
    }

    def test_dsets_2_path(self):
        self.assertEqual(len(patcher.globus_path_from_dataset([self.dset] * 3)), 3)
        sdsc_path = ('/mnt/ibl/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/'
                     'channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy')
        globus_path_sdsc = ('/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/'
                            'channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy')
        globus_path_sr = ('/mnt/s0/Data/Subjects/SWC_014/2019-12-11/001/alf/probe00/'
                          'channels.localCoordinates.npy')

        # Test sdsc_path_from_dataset
        testable = patcher.sdsc_path_from_dataset(self.dset)
        self.assertEqual(str(testable), sdsc_path)
        self.assertIsInstance(testable, PurePosixPath)

        # Test sdsc_globus_path_from_dataset
        testable = patcher.sdsc_globus_path_from_dataset(self.dset)
        self.assertEqual(str(testable), globus_path_sdsc)
        self.assertIsInstance(testable, PurePosixPath)

        # Test globus_path_from_dataset
        testable = patcher.globus_path_from_dataset(self.dset, repository='ibl_floferlab_SR')
        self.assertEqual(str(testable), globus_path_sr)
        self.assertIsInstance(testable, PurePosixPath)


SUBJECT = 'clns0730'
USER = 'test_user'

md5_0 = 'add2ab27dbf8428f8140-0870d5080c7f'
r = {'created_by': 'olivier',
     'path': f'{SUBJECT}/2018-08-24/002',
     'filenames': ["raw_behavior_data/_iblrig_encoderTrialInfo.raw.ssv"],
     'hashes': [md5_0],
     'filesizes': [1234],
     'versions': [__version__]}

MOCK_SESSION_SETTINGS = {
    'SESSION_DATE': '2018-04-01',
    'SESSION_DATETIME': '2018-04-01T12:48:26.795526',
    'PYBPOD_CREATOR': [USER, 'f092c2d5-c98a-45a1-be7c-df05f129a93c', 'local'],
    'SESSION_NUMBER': '002',
    'SUBJECT_NAME': SUBJECT,
    'PYBPOD_BOARD': '_iblrig_mainenlab_behavior_1',
    'PYBPOD_PROTOCOL': '_iblrig_tasks_ephysChoiceWorld',
    'IBLRIG_VERSION_TAG': '5.4.1',
    'SUBJECT_WEIGHT': 22,
}

MOCK_SESSION_DICT = {
    'subject': SUBJECT,
    'start_time': '2018-04-01T12:48:26.795526',
    'number': 2,
    'users': [USER]
}


class TestRegistrationEndpoint(unittest.TestCase):

    def test_task_names_extractors(self):
        """
        This is to test against regressions
        """
        task_out = [
            ('_iblrig_tasks_biasedChoiceWorld3.7.0', 'Behavior training/tasks'),
            ('_iblrig_tasks_biasedScanningChoiceWorld5.2.3', 'Behavior training/tasks'),
            ('_iblrig_tasks_trainingChoiceWorld3.6.0', 'Behavior training/tasks'),
            ('_iblrig_tasks_ephysChoiceWorld5.1.3', 'Ephys recording with acute probe(s)'),
            ('_iblrig_calibration_frame2TTL4.1.3', []),
            ('_iblrig_tasks_habituationChoiceWorld3.6.0', 'Behavior training/tasks'),
            ('_iblrig_tasks_scanningOptoChoiceWorld5.0.0', []),
            ('_iblrig_tasks_RewardChoiceWorld4.1.3', []),
            ('_iblrig_calibration_screen4.1.3', []),
            ('_iblrig_tasks_ephys_certification4.1.3', 'Ephys recording with acute probe(s)'),
        ]
        for to in task_out:
            out = registration.IBLRegistrationClient._alyx_procedure_from_task(to[0])
            self.assertEqual(out, to[1])
        # also makes sure that all task types have a defined procedure
        task_types = ibllib.io.extractors.base._get_task_types_json_config()
        for task_type in set([task_types[tt] for tt in task_types]):
            assert registration._alyx_procedure_from_task_type(task_type) is not None, task_type + ' has no associate procedure'


class TestRegistration(unittest.TestCase):

    def setUp(self) -> None:
        self.one = ONE(**TEST_DB, cache_rest=None)
        # makes sure tests start without session created
        eid = self.one.search(subject=SUBJECT, date_range='2018-04-01', query_type='remote')
        for ei in eid:
            self.one.alyx.rest('sessions', 'delete', id=ei)
        self.td = tempfile.TemporaryDirectory()
        self.session_path = Path(self.td.name).joinpath(SUBJECT, '2018-04-01', '002')
        self.alf_path = self.session_path.joinpath('alf')
        self.alf_path.mkdir(parents=True)
        np.save(self.alf_path.joinpath('spikes.times.npy'), np.random.random(500))
        np.save(self.alf_path.joinpath('spikes.amps.npy'), np.random.random(500))
        self.revision = ''.join(random.choices(string.ascii_letters, k=3))
        self.rev_path = self.alf_path.joinpath(f'#{self.revision}#')
        self.rev_path.mkdir(parents=True)
        np.save(self.rev_path.joinpath('spikes.times.npy'), np.random.random(300))
        np.save(self.rev_path.joinpath('spikes.amps.npy'), np.random.random(300))

        self.today_revision = datetime.datetime.today().strftime('%Y-%m-%d')

        # Create a revision if doesn't already exist
        try:
            self.rev = self.one.alyx.rest('revisions', 'read', id=self.revision)
        except HTTPError:
            self.rev = self.one.alyx.rest('revisions', 'create', data={'name': self.revision})
        # Create a tag if doesn't already exist
        try:
            self.tag = next(x for x in self.one.alyx.rest('tags', 'list')
                            if x['name'] == 'test_tag')
        except StopIteration:
            self.tag = self.one.alyx.rest('tags', 'create',
                                          data={'name': 'test_tag', 'protected': True})

    def test_registration_datasets(self):
        # registers a single file
        ses = self.one.alyx.rest('sessions', 'create', data=MOCK_SESSION_DICT)
        st_file = self.alf_path.joinpath('spikes.times.npy')
        registration.register_dataset(file_list=st_file, one=self.one)
        dsets = self.one.alyx.rest('datasets', 'list', session=ses['url'][-36:])
        self.assertTrue(len(dsets) == 1)
        # registers a list of files
        flist = list(self.alf_path.glob('*.npy'))
        r = registration.register_dataset(file_list=flist, one=self.one)
        dsets = self.one.alyx.rest('datasets', 'list', session=ses['url'][-36:])
        self.assertTrue(len(dsets) == 2)
        self.assertTrue(all(not d['revision'] for d in r))
        self.assertTrue(all(d['default'] for d in r))
        self.assertTrue(all(d['collection'] == 'alf' for d in r))

        # simulate all the datasets exists, re-register and asserts that exists is set to True
        # as the files haven't changed
        frs = self.one.alyx.rest('files', 'list', django=f"dataset__session,{ses['url'][-36:]}")
        for fr in frs:
            self.one.alyx.rest('files', 'partial_update',
                               id=fr['url'][-36:], data={'exists': True})
        r = registration.register_dataset(file_list=flist, one=self.one)
        self.assertTrue(all(all(fr['exists'] for fr in rr['file_records']) for rr in r))
        # now that files have changed, makes sure the exists flags are set to False
        np.save(self.alf_path.joinpath('spikes.times.npy'), np.random.random(500))
        np.save(self.alf_path.joinpath('spikes.amps.npy'), np.random.random(500))
        r = registration.register_dataset(file_list=flist, one=self.one)
        self.assertTrue(all(all(not fr['exists'] for fr in rr['file_records']) for rr in r))

        # Add a protected tag to all the datasets
        dsets = self.one.alyx.rest('datasets', 'list', session=ses['url'][-36:])
        for d in dsets:
            self.one.alyx.rest('datasets', 'partial_update',
                               id=d['url'][-36:], data={'tags': ['test_tag']})

        # Test registering with a revision already in the file path, should use this rather than create one with today's date
        flist = list(self.rev_path.glob('*.npy'))
        r = registration.register_dataset(file_list=flist, one=self.one)
        self.assertTrue(all(d['revision'] == self.revision for d in r))
        self.assertTrue(all(d['default'] for d in r))
        self.assertTrue(all(d['collection'] == 'alf' for d in r))

        # Add a protected tag to all the datasets
        dsets = self.one.alyx.rest('datasets', 'list', session=ses['url'][-36:])
        for d in dsets:
            self.one.alyx.rest('datasets', 'partial_update',
                               id=d['url'][-36:], data={'tags': ['test_tag']})

        # Register again with revision in file path, it should register to self.revision + a
        flist = list(self.rev_path.glob('*.npy'))

        r = registration.register_dataset(file_list=flist, one=self.one)
        self.assertTrue(all(d['revision'] == f'{self.revision}a' for d in r))
        self.assertTrue(self.alf_path.joinpath(f'#{self.revision}a#', 'spikes.times.npy').exists())
        self.assertTrue(self.alf_path.joinpath(f'#{self.revision}a#', 'spikes.amps.npy').exists())
        self.assertFalse(self.alf_path.joinpath(f'#{self.revision}#', 'spikes.times.npy').exists())
        self.assertFalse(self.alf_path.joinpath(f'#{self.revision}#', 'spikes.amps.npy').exists())

        # When we re-register the original it should move them into revision with today's date
        flist = list(self.alf_path.glob('*.npy'))
        r = registration.register_dataset(file_list=flist, one=self.one)
        self.assertTrue(all(d['revision'] == self.today_revision for d in r))
        self.assertTrue(self.alf_path.joinpath(f'#{self.today_revision}#', 'spikes.times.npy').exists())
        self.assertTrue(self.alf_path.joinpath(f'#{self.today_revision}#', 'spikes.amps.npy').exists())
        self.assertFalse(self.alf_path.joinpath('spikes.times.npy').exists())
        self.assertFalse(self.alf_path.joinpath('spikes.amps.npy').exists())

        # Protect the latest datasets
        dsets = self.one.alyx.rest('datasets', 'list', session=ses['url'][-36:], no_cache=True)
        for d in dsets:
            self.one.alyx.rest('datasets', 'partial_update',
                               id=d['url'][-36:], data={'tags': ['test_tag']})

        # Same day revision
        # Need to remake the original files
        np.save(self.alf_path.joinpath('spikes.times.npy'), np.random.random(500))
        np.save(self.alf_path.joinpath('spikes.amps.npy'), np.random.random(500))
        flist = list(self.alf_path.glob('*.npy'))
        r = registration.register_dataset(file_list=flist, one=self.one)
        self.assertTrue(all(d['revision'] == self.today_revision + 'a' for d in r))

    def _write_settings_file(self):
        behavior_path = self.session_path.joinpath('raw_behavior_data')
        behavior_path.mkdir()
        settings_file = behavior_path.joinpath('_iblrig_taskSettings.raw.json')
        with open(settings_file, 'w') as fid:
            json.dump(MOCK_SESSION_SETTINGS, fid)
        return settings_file

    def test_create_sessions(self):
        flag_file = self.session_path.joinpath('create_me.flag')
        flag_file.touch()
        rc = registration.IBLRegistrationClient(one=self.one)
        # Should raise an error if settings file does not exist
        self.assertRaises(ValueError, rc.create_sessions, self.session_path)
        self._write_settings_file()
        # Test dry
        sessions, records = rc.create_sessions(self.session_path, dry=True)
        self.assertEqual(1, len(sessions))
        self.assertEqual(sessions[0], self.session_path)
        self.assertIsNone(records[0])
        self.assertTrue(flag_file.exists())
        # Test not dry
        sessions, records = rc.create_sessions(self.session_path, dry=False)
        self.assertEqual(1, len(sessions))
        self.assertEqual(sessions[0], self.session_path)
        self.assertFalse(flag_file.exists())

    def test_registration_session(self):
        settings_file = self._write_settings_file()
        rc = registration.IBLRegistrationClient(one=self.one)
        rc.register_session(str(self.session_path))
        eid = self.one.search(subject=SUBJECT, date_range=['2018-04-01', '2018-04-01'],
                              query_type='remote')[0]
        datasets = self.one.alyx.rest('datasets', 'list', session=eid)
        for ds in datasets:
            self.assertTrue(ds['hash'] is not None)
            self.assertTrue(ds['file_size'] is not None)
            self.assertTrue(ds['version'] == ibllib.__version__)
        # checks the procedure of the session
        ses_info = self.one.alyx.rest('sessions', 'read', id=eid)
        self.assertTrue(ses_info['procedures'] == ['Ephys recording with acute probe(s)'])
        self.one.alyx.rest('sessions', 'delete', id=eid)
        # re-register the session as behaviour this time
        MOCK_SESSION_SETTINGS['PYBPOD_PROTOCOL'] = '_iblrig_tasks_trainingChoiceWorld6.3.1'
        with open(settings_file, 'w') as fid:
            json.dump(MOCK_SESSION_SETTINGS, fid)
        rc.register_session(self.session_path)
        eid = self.one.search(subject=SUBJECT, date_range=['2018-04-01', '2018-04-01'],
                              query_type='remote')[0]
        ses_info = self.one.alyx.rest('sessions', 'read', id=eid)
        self.assertTrue(ses_info['procedures'] == ['Behavior training/tasks'])
        self.one.alyx.rest('sessions', 'delete', id=eid)
        # re-register the session as unknown protocol this time
        MOCK_SESSION_SETTINGS['PYBPOD_PROTOCOL'] = 'gnagnagna'
        with open(settings_file, 'w') as fid:
            json.dump(MOCK_SESSION_SETTINGS, fid)
        rc.register_session(self.session_path)
        eid = self.one.search(subject=SUBJECT, date_range=['2018-04-01', '2018-04-01'],
                              query_type='remote')[0]
        ses_info = self.one.alyx.rest('sessions', 'read', id=eid)
        self.assertTrue(ses_info['procedures'] == [])
        self.one.alyx.rest('sessions', 'delete', id=eid)

    def test_register_chained_session(self):
        """Tests for registering a session with chained (multiple) protocols"""
        behaviour_paths = [self.session_path.joinpath(f'raw_task_data_{i:02}') for i in range(2)]
        for p in behaviour_paths:
            p.mkdir()

        # Set the collections
        params_path = Path(__file__).parent.joinpath('fixtures', 'io', '_ibl_experiment.description.yaml')
        experiment_description = session_params.read_params(params_path)
        assert experiment_description
        collections = map(lambda x: next(iter(x.values())), experiment_description['tasks'])
        for collection, d in zip(map(lambda x: x.parts[-1], behaviour_paths), collections):
            d['collection'] = collection

        # Save experiment description
        session_params.write_params(self.session_path, experiment_description)

        with open(behaviour_paths[1].joinpath('_iblrig_taskSettings.raw.json'), 'w') as fid:
            json.dump(MOCK_SESSION_SETTINGS, fid)

        settings = MOCK_SESSION_SETTINGS.copy()
        settings['PYBPOD_PROTOCOL'] = '_iblrig_tasks_passiveChoiceWorld'
        start_time = (datetime.datetime.fromisoformat(settings['SESSION_DATETIME']) -
                      datetime.timedelta(hours=1, minutes=2, seconds=12))
        settings['SESSION_DATETIME'] = start_time.isoformat()
        with open(behaviour_paths[0].joinpath('_iblrig_taskSettings.raw.json'), 'w') as fid:
            json.dump(settings, fid)

        rc = registration.IBLRegistrationClient(one=self.one)
        session, recs = rc.register_session(self.session_path)

        ses_info = self.one.alyx.rest('sessions', 'read', id=session['id'])
        self.assertCountEqual(experiment_description['procedures'], ses_info['procedures'])
        self.assertCountEqual(experiment_description['projects'], ses_info['projects'])
        self.assertCountEqual({'IS_MOCK': False, 'IBLRIG_VERSION': None}, ses_info['json'])
        self.assertEqual('2018-04-01T11:46:14.795526', ses_info['start_time'])
        # Test task protocol
        expected = '_iblrig_tasks_passiveChoiceWorld5.4.1/_iblrig_tasks_ephysChoiceWorld5.4.1'
        self.assertEqual(expected, ses_info['task_protocol'])
        # Test weightings created on Alyx
        w = self.one.alyx.rest('subjects', 'read', id=SUBJECT)['weighings']
        self.assertEqual(2, len(w))
        self.assertCountEqual({22.}, {x['weight'] for x in w})
        weight_dates = {x['date_time'] for x in w}
        self.assertEqual(2, len(weight_dates))
        self.assertIn(ses_info['start_time'], weight_dates)

    def tearDown(self) -> None:
        self.td.cleanup()
        self.one.alyx.rest('revisions', 'delete', id=self.rev['name'])
        self.one.alyx.rest('tags', 'delete', id=self.tag['id'])
        today_revision = self.one.alyx.rest('revisions', 'list', id=self.today_revision)
        today_rev = [rev for rev in today_revision if self.today_revision in rev['name']]
        for rev in today_rev:
            self.one.alyx.rest('revisions', 'delete', id=rev['name'])

        v1_rev = [rev for rev in today_revision if self.revision in rev['name']]
        for rev in v1_rev:
            self.one.alyx.rest('revisions', 'delete', id=rev['name'])
        # Delete weighings
        for w in self.one.alyx.rest('subjects', 'read', id=SUBJECT)['weighings']:
            self.one.alyx.rest('weighings', 'delete', id=w['url'].split('/')[-1])


if __name__ == '__main__':
    unittest.main()

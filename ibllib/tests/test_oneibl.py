import unittest
import tempfile
from unittest import mock
from pathlib import PurePosixPath, Path
import json

from requests import HTTPError
import numpy as np
from one.api import ONE
import iblutil.io.params as iopar

from ibllib.oneibl import patcher, registration
import ibllib.io.extractors.base
from ibllib.misc import version
from ibllib.tests import TEST_DB


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
     'versions': [version.ibllib()]}

MOCK_SESSION_SETTINGS = {
    'SESSION_DATE': '2018-04-01',
    'SESSION_DATETIME': '2018-04-01T12:48:26.795526',
    'PYBPOD_CREATOR': [USER,
                       'f092c2d5-c98a-45a1-be7c-df05f129a93c',
                       'local'],
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
            ('_iblrig_calibration_frame2TTL4.1.3', None),
            ('_iblrig_tasks_habituationChoiceWorld3.6.0', 'Behavior training/tasks'),
            ('_iblrig_tasks_scanningOptoChoiceWorld5.0.0', None),
            ('_iblrig_tasks_RewardChoiceWorld4.1.3', None),
            ('_iblrig_calibration_screen4.1.3', None),
            ('_iblrig_tasks_ephys_certification4.1.3', 'Ephys recording with acute probe(s)'),
        ]
        for to in task_out:
            out = registration._alyx_procedure_from_task(to[0])
            self.assertEqual(out, to[1])
        # also makes sure that all task types have a defined procedure
        task_types = ibllib.io.extractors.base._get_task_types_json_config()
        for task_type in list(set([task_types[tt] for tt in task_types])):
            assert registration._alyx_procedure_from_task_type(task_type) is not None


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
        self.rev_path = self.alf_path.joinpath('#v1#')
        self.rev_path.mkdir(parents=True)
        np.save(self.rev_path.joinpath('spikes.times.npy'), np.random.random(300))
        np.save(self.rev_path.joinpath('spikes.amps.npy'), np.random.random(300))

        # Create a revision if doesn't already exist
        try:
            self.rev = self.one.alyx.rest('revisions', 'read', id='v1')
        except HTTPError:
            self.rev = self.one.alyx.rest('revisions', 'create', data={'name': 'v1'})
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
        self.assertTrue(all([all([fr['exists'] for fr in rr['file_records']]) for rr in r]))
        # now that files have changed, makes sure the exists flags are set to False
        np.save(self.alf_path.joinpath('spikes.times.npy'), np.random.random(500))
        np.save(self.alf_path.joinpath('spikes.amps.npy'), np.random.random(500))
        r = registration.register_dataset(file_list=flist, one=self.one)
        self.assertTrue(all(all(not(fr['exists']) for fr in rr['file_records']) for rr in r))

        # Test registering with a revision
        flist = list(self.rev_path.glob('*.npy'))
        r = registration.register_dataset(file_list=flist, one=self.one)
        self.assertTrue(all(d['revision'] == 'v1' for d in r))
        self.assertTrue(all(d['default'] for d in r))
        self.assertTrue(all(d['collection'] == 'alf' for d in r))
        dsets = self.one.alyx.rest('datasets', 'list', session=ses['url'][-36:], revision='v1')

        # Add a protected tag to a dataset
        for d in dsets:
            self.one.alyx.rest('datasets', 'partial_update',
                               id=d['url'][-36:], data={'tags': ['test_tag']})
        with self.assertRaises(HTTPError):
            registration.register_dataset(file_list=flist, one=self.one)

    def test_create_sessionS(self):
        flag_file = self.session_path.joinpath('create_me.flag')
        flag_file.touch()
        rc = registration.RegistrationClient(one=self.one)
        rc.create_sessions(self.session_path, dry=True)
        rc.create_sessions(self.session_path)

    def test_registration_session(self):
        behavior_path = self.session_path.joinpath('raw_behavior_data')
        behavior_path.mkdir()
        settings_file = behavior_path.joinpath('_iblrig_taskSettings.raw.json')
        with open(settings_file, 'w') as fid:
            json.dump(MOCK_SESSION_SETTINGS, fid)
        rc = registration.RegistrationClient(one=self.one)
        rc.register_session(str(self.session_path))
        eid = self.one.search(subject=SUBJECT, date_range=['2018-04-01', '2018-04-01'],
                              query_type='remote')[0]
        datasets = self.one.alyx.rest('datasets', 'list', session=eid)
        for ds in datasets:
            self.assertTrue(ds['hash'] is not None)
            self.assertTrue(ds['file_size'] is not None)
            self.assertTrue(ds['version'] == version.ibllib())
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

    def tearDown(self) -> None:
        self.td.cleanup()
        self.one.alyx.rest('revisions', 'delete', id=self.rev['name'])
        self.one.alyx.rest('tags', 'delete', id=self.tag['id'])


if __name__ == '__main__':
    unittest.main()

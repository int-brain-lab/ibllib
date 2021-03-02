import unittest
import tempfile
from pathlib import Path
import json

import numpy as np

import ibllib.io.extractors.base
from ibllib.misc import version
from oneibl import one, registration

one = one.ONE(base_url='https://test.alyx.internationalbrainlab.org', username='test_user',
              password='TapetesBloc18')
SUBJECT = 'clns0730'
USER = 'test_user'

# one = one.ONE(base_url='http://localhost:8000')
# SUBJECT = 'CSP013'
# USER = 'olivier'

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
        # makes sure tests start without session created
        eid = one.search(subjects=SUBJECT, date_range='2018-04-01')
        for ei in eid:
            one.alyx.rest('sessions', 'delete', id=ei)
        self.td = tempfile.TemporaryDirectory()
        self.session_path = Path(self.td.name).joinpath(SUBJECT, '2018-04-01', '002')
        self.alf_path = self.session_path.joinpath('alf')
        self.alf_path.mkdir(parents=True)
        np.save(self.alf_path.joinpath('spikes.times.npy'), np.random.random(500))
        np.save(self.alf_path.joinpath('spikes.amps.npy'), np.random.random(500))

    def test_registration_datasets(self):
        # registers a single file
        ses = one.alyx.rest('sessions', 'create', data=MOCK_SESSION_DICT)
        st_file = self.alf_path.joinpath('spikes.times.npy')
        registration.register_dataset(file_list=st_file, one=one)
        dsets = one.alyx.rest('datasets', 'list', session=ses['url'][-36:])
        self.assertTrue(len(dsets) == 1)
        # registers a list of files
        flist = list(self.alf_path.glob('*.npy'))
        r = registration.register_dataset(file_list=flist, one=one)
        dsets = one.alyx.rest('datasets', 'list', session=ses['url'][-36:])
        self.assertTrue(len(dsets) == 2)
        # simulate all the datasets exists, re-register and asserts that exists is set to True
        # as the files haven't changed
        frs = one.alyx.rest('files', 'list', django=f"dataset__session,{ses['url'][-36:]}")
        for fr in frs:
            one.alyx.rest('files', 'partial_update', id=fr['url'][-36:], data={'exists': True})
        r = registration.register_dataset(file_list=flist, one=one)
        self.assertTrue(all([all([fr['exists'] for fr in rr['file_records']]) for rr in r]))
        # now that files have changed, makes sure the exists flags are set to False
        np.save(self.alf_path.joinpath('spikes.times.npy'), np.random.random(500))
        np.save(self.alf_path.joinpath('spikes.amps.npy'), np.random.random(500))
        r = registration.register_dataset(file_list=flist, one=one)
        self.assertTrue(all([all([not(fr['exists']) for fr in rr['file_records']]) for rr in r]))

    def test_registration_session(self):
        behavior_path = self.session_path.joinpath('raw_behavior_data')
        behavior_path.mkdir()
        settings_file = behavior_path.joinpath('_iblrig_taskSettings.raw.json')
        with open(settings_file, 'w') as fid:
            json.dump(MOCK_SESSION_SETTINGS, fid)
        rc = registration.RegistrationClient(one=one)
        rc.register_session(self.session_path)
        eid = one.search(subjects=SUBJECT, date_range=['2018-04-01', '2018-04-01'])[0]
        datasets = one.alyx.rest('datasets', 'list', session=eid)
        for ds in datasets:
            self.assertTrue(ds['hash'] is not None)
            self.assertTrue(ds['file_size'] is not None)
            self.assertTrue(ds['version'] == version.ibllib())
        # checks the procedure of the session
        ses_info = one.alyx.rest('sessions', 'read', id=eid)
        self.assertTrue(ses_info['procedures'] == ['Ephys recording with acute probe(s)'])
        one.alyx.rest('sessions', 'delete', id=eid)
        # re-register the session as behaviour this time
        MOCK_SESSION_SETTINGS['PYBPOD_PROTOCOL'] = '_iblrig_tasks_trainingChoiceWorld6.3.1'
        with open(settings_file, 'w') as fid:
            json.dump(MOCK_SESSION_SETTINGS, fid)
        rc.register_session(self.session_path)
        eid = one.search(subjects=SUBJECT, date_range=['2018-04-01', '2018-04-01'])[0]
        ses_info = one.alyx.rest('sessions', 'read', id=eid)
        self.assertTrue(ses_info['procedures'] == ['Behavior training/tasks'])
        one.alyx.rest('sessions', 'delete', id=eid)
        # re-register the session as unknown protocol this time
        MOCK_SESSION_SETTINGS['PYBPOD_PROTOCOL'] = 'gnagnagna'
        with open(settings_file, 'w') as fid:
            json.dump(MOCK_SESSION_SETTINGS, fid)
        rc.register_session(self.session_path)
        eid = one.search(subjects=SUBJECT, date_range=['2018-04-01', '2018-04-01'])[0]
        ses_info = one.alyx.rest('sessions', 'read', id=eid)
        self.assertTrue(ses_info['procedures'] == [])
        one.alyx.rest('sessions', 'delete', id=eid)

    def tearDown(self) -> None:
        self.td.cleanup()


if __name__ == '__main__':
    unittest.main(exit=False)

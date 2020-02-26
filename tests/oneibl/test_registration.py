import unittest
import tempfile
from pathlib import Path
import json

from ibllib.misc import version
from oneibl import one, registration

one = one.ONE(base_url='https://test.alyx.internationalbrainlab.org', username='test_user',
              password='TapetesBloc18')

md5_0 = 'add2ab27dbf8428f8140-0870d5080c7f'
r = {'created_by': 'olivier',
     'path': 'clns0730/2018-08-24/002',
     'filenames': ["raw_behavior_data/_iblrig_encoderTrialInfo.raw.ssv"],
     'hashes': [md5_0],
     'filesizes': [1234],
     'versions': [version.ibllib()]}


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

    def test_single_registration(self):
        dataset = one.alyx.rest('register-file', 'create', data=r)
        ds = one.alyx.rest('datasets', 'read', id=dataset[0]['id'])
        self.assertEqual(ds['hash'], md5_0)
        self.assertEqual(ds['file_size'], 1234)
        self.assertEqual(ds['version'], version.ibllib())
        self.assertEqual(len(dataset[0]['file_records']), 2)
        one.alyx.rest('datasets', 'delete', id=dataset[0]['id'])
        # self.assertEqual(ds['collection'], 'raw_behavior_data')

    def test_registration_server_only(self):
        # in this case the repository has only one file record
        r_ = r.copy()
        r_['server_only'] = True
        # r_['name'] = 'ibl_patcher'
        dataset = one.alyx.rest('register-file', 'create', data=r_)[0]
        self.assertEqual(len(dataset['file_records']), 1)
        self.assertEqual(dataset['file_records'][0]['data_repository'], 'flatiron_mainenlab')
        one.alyx.rest('datasets', 'delete', id=dataset['id'])


class TestRegistrationSession(unittest.TestCase):

    def test_registration_session(self):
        settings = {
            'SESSION_DATE': '2018-04-01',
            'SESSION_DATETIME': '2018-04-01T12:48:26.795526',
            'PYBPOD_CREATOR': ['test_user',
                               'f092c2d5-c98a-45a1-be7c-df05f129a93c',
                               'local'],
            'SESSION_NUMBER': '002',
            'SUBJECT_NAME': 'clns0730',
            'PYBPOD_BOARD': '_iblrig_mainenlab_behavior_1',
            'PYBPOD_PROTOCOL': '_iblrig_tasks_ephysChoiceWorld',
            'IBLRIG_VERSION_TAG': '5.4.1',
            'SUBJECT_WEIGHT': 22,
        }
        with tempfile.TemporaryDirectory() as td:
            # creates the local session
            session_path = Path(td).joinpath('clns0730', '2018-04-01', '002')
            alf_path = session_path.joinpath('alf')
            alf_path.mkdir(parents=True)
            alf_path.joinpath('spikes.times.npy').touch()
            alf_path.joinpath('spikes.amps.npy').touch()
            behavior_path = session_path.joinpath('raw_behavior_data')
            behavior_path.mkdir()
            settings_file = behavior_path.joinpath('_iblrig_taskSettings.raw.json')
            eid = one.search(subjects='clns0730', date_range=['2018-04-01', '2018-04-01'])
            if len(eid):
                one.alyx.rest('sessions', 'delete', id=eid[0])
            with open(settings_file, 'w') as fid:
                json.dump(settings, fid)
            rc = registration.RegistrationClient(one=one)
            rc.register_session(session_path)
            eid = one.search(subjects='clns0730', date_range=['2018-04-01', '2018-04-01'])[0]
            datasets = one.alyx.get('/datasets?subject=clns0730&date=2018-04-01')
            for ds in datasets:
                self.assertTrue(ds['hash'] is not None)
                self.assertTrue(ds['file_size'] is not None)
                self.assertTrue(ds['version'] == version.ibllib())
            # checks the procedure of the session
            ses_info = one.alyx.rest('sessions', 'read', id=eid)
            self.assertTrue(ses_info['procedures'] == ['Ephys recording with acute probe(s)'])
            one.alyx.rest('sessions', 'delete', id=eid)
            # re-register the session as behaviour this time
            settings['PYBPOD_PROTOCOL'] = '_iblrig_tasks_trainingChoiceWorld6.3.1'
            with open(settings_file, 'w') as fid:
                json.dump(settings, fid)
            rc.register_session(session_path)
            eid = one.search(subjects='clns0730', date_range=['2018-04-01', '2018-04-01'])[0]
            ses_info = one.alyx.rest('sessions', 'read', id=eid)
            self.assertTrue(ses_info['procedures'] == ['Behavior training/tasks'])

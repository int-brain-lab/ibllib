import unittest

from ibllib.misc import version
from oneibl.one import ONE

one = ONE(base_url='https://test.alyx.internationalbrainlab.org', username='test_user',
          password='TapetesBloc18')

md5_0 = 'add2ab27dbf8428f8140-0870d5080c7f'
r = {'created_by': 'olivier',
     'path': 'clns0730/2018-08-24/002',
     'filenames': ["raw_behavior_data/_iblrig_encoderTrialInfo.raw.ssv"],
     'hashes': [md5_0],
     'filesizes': [1234],
     'versions': [version.ibllib()]}


class TestRegistration(unittest.TestCase):

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

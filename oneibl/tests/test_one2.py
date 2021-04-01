# flake8: noqa
from pathlib import Path, PurePosixPath, PureWindowsPath
import unittest

from oneibl import webclient as wc

dsets = [
    {'url': 'https://alyx.internationalbrainlab.org/datasets/00059298-1b33-429c-a802-fa51bb662d72',
  'name': 'channels.localCoordinates.npy',
  'created_by': 'nate',
  'created_datetime': '2020-02-07T22:08:08.053982',
  'dataset_type': 'channels.localCoordinates',
  'data_format': 'npy',
  'collection': 'alf/probe00',
  'session': 'https://alyx.internationalbrainlab.org/sessions/7cffad38-0f22-4546-92b5-fd6d2e8b2be9',
  'file_size': 6064,
  'hash': 'bc74f49f33ec0f7545ebc03f0490bdf6',
  'version': '1.5.36',
  'experiment_number': 1,
  'file_records': [{'id': 'c9ae1b6e-03a6-41c9-9e1b-4a7f9b5cfdbf',
    'data_repository': 'ibl_floferlab_SR',
    'data_repository_path': '/mnt/s0/Data/Subjects/',
    'relative_path': 'SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy',
    'data_url': None,
    'exists': True},
   {'id': 'f434a638-bc61-4695-884e-70fd1e521d60',
    'data_repository': 'flatiron_hoferlab',
    'data_repository_path': '/hoferlab/Subjects/',
    'relative_path': 'SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy',
    'data_url': 'https://ibl.flatironinstitute.org/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy',
    'exists': True}],
  'auto_datetime': '2021-02-10T20:24:31.484939'},
 {'url': 'https://alyx.internationalbrainlab.org/datasets/00e6dce3-0bb7-44d7-84b5-f41b2c4cf565',
  'name': 'channels.brainLocationIds_ccf_2017.npy',
  'created_by': 'mayo',
  'created_datetime': '2020-10-22T17:10:02.951475',
  'dataset_type': 'channels.brainLocationIds_ccf_2017',
  'data_format': 'npy',
  'collection': 'alf/probe00',
  'session': 'https://alyx.internationalbrainlab.org/sessions/dd4da095-4a99-4bf3-9727-f735077dba66',
  'file_size': 3120,
  'hash': 'c5779e6d02ae6d1d6772df40a1a94243',
  'version': 'unversioned',
  'experiment_number': 1,
  'file_records': [{'id': 'f6965181-ce90-4259-8167-2278af73a786',
    'data_repository': 'flatiron_mainenlab',
    'data_repository_path': '/mainenlab/Subjects/',
    'relative_path': 'ZM_1897/2019-12-02/001/alf/probe00/channels.brainLocationIds_ccf_2017.npy',
    'data_url': 'https://ibl.flatironinstitute.org/mainenlab/Subjects/ZM_1897/2019-12-02/001/alf/probe00/channels.brainLocationIds_ccf_2017.00e6dce3-0bb7-44d7-84b5-f41b2c4cf565.npy',
    'exists': True}],
  'auto_datetime': '2021-02-10T20:24:31.484939'},
 {'url': 'https://alyx.internationalbrainlab.org/datasets/017c6a14-0270-4740-baaa-c4133f331f4f',
  'name': 'channels.localCoordinates.npy',
  'created_by': 'feihu',
  'created_datetime': '2020-07-21T15:55:22.693734',
  'dataset_type': 'channels.localCoordinates',
  'data_format': 'npy',
  'collection': 'alf/probe00',
  'session': 'https://alyx.internationalbrainlab.org/sessions/7622da34-51b6-4661-98ae-a57d40806008',
  'file_size': 6064,
  'hash': 'bc74f49f33ec0f7545ebc03f0490bdf6',
  'version': '1.5.36',
  'experiment_number': 1,
  'file_records': [{'id': '224f8060-bf5c-46f6-8e63-0528fc364f63',
    'data_repository': 'dan_lab_SR',
    'data_repository_path': '/mnt/s0/Data/Subjects/',
    'relative_path': 'DY_014/2020-07-15/001/alf/probe00/channels.localCoordinates.npy',
    'data_url': None,
    'exists': True},
   {'id': '9d53161d-6b46-4a0a-871e-7ddae9626844',
    'data_repository': 'flatiron_danlab',
    'data_repository_path': '/danlab/Subjects/',
    'relative_path': 'DY_014/2020-07-15/001/alf/probe00/channels.localCoordinates.npy',
    'data_url': 'https://ibl.flatironinstitute.org/danlab/Subjects/DY_014/2020-07-15/001/alf/probe00/channels.localCoordinates.017c6a14-0270-4740-baaa-c4133f331f4f.npy',
    'exists': True}],
  'auto_datetime': '2021-02-10T20:24:31.484939'}]


class TestAlyx2Path(unittest.TestCase):

    def test_dsets_2_path(self):
        assert len(wc.globus_path_from_dataset(dsets)) == 3
        sdsc_path = '/mnt/ibl/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy'
        one_path = '/one_root/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy'
        globus_path_sdsc = '/hoferlab/Subjects/SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.00059298-1b33-429c-a802-fa51bb662d72.npy'
        globus_path_sr = '/mnt/s0/Data/Subjects/SWC_014/2019-12-11/001/alf/probe00/channels.localCoordinates.npy'

        # Test sdsc_path_from_dataset
        testable = wc.sdsc_path_from_dataset(dsets[0])
        self.assertEqual(str(testable), sdsc_path)
        self.assertIsInstance(testable, PurePosixPath)

        # Test one_path_from_dataset
        testable = wc.one_path_from_dataset(dsets[0], one_cache=PurePosixPath('/one_root'))
        self.assertEqual(str(testable), one_path)
        # Check handles string inputs
        testable = wc.one_path_from_dataset(dsets[0], one_cache='/one_root')
        self.assertTrue(hasattr(testable, 'is_absolute'), 'Failed to return Path object')
        self.assertEqual(str(testable).replace('\\', '/'), one_path)

        # Test one_path_from_dataset using Windows path
        one_path = PureWindowsPath(r'C:/Users/User/')
        testable = wc.one_path_from_dataset(dsets[0], one_cache=one_path)
        self.assertIsInstance(testable, PureWindowsPath)
        self.assertTrue(str(testable).startswith(str(one_path)))

        # Test sdsc_globus_path_from_dataset
        testable = wc.sdsc_globus_path_from_dataset(dsets[0])
        self.assertEqual(str(testable), globus_path_sdsc)
        self.assertIsInstance(testable, PurePosixPath)

        # Test globus_path_from_dataset
        testable = wc.globus_path_from_dataset(dsets[0], repository='ibl_floferlab_SR')
        self.assertEqual(str(testable), globus_path_sr)
        self.assertIsInstance(testable, PurePosixPath)

        # Tests _path_from_filerecord: when given a string, a system path object should be returned
        fr = dsets[0]['file_records'][0]
        testable = wc._path_from_filerecord(fr, root_path='C:\\')
        self.assertIsInstance(testable, Path)

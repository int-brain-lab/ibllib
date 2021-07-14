import unittest
from unittest import mock
from pathlib import PurePosixPath

from one.api import ONE
import iblutil.io.params as iopar

from ibllib.oneibl import patcher
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


if __name__ == '__main__':
    unittest.main()

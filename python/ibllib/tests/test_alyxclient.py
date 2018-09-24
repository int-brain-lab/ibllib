import unittest
import numpy as np
import os
import ibllib.webclient as wc
import oneibl.params
import tempfile
import shutil

par = oneibl.params.get()


class TestDownloadHTTP(unittest.TestCase):

    def setUp(self):
        # Init connection to the database
        self.ac = wc.AlyxClient(username='test_user', password='TapetesBloc18',
                                base_url='https://test.alyx.internationalbrainlab.org')

    def test_rest_endpoint(self):
        # tests that non-existing endpoints /actions are caught properly
        with self.assertRaises(ValueError):
            res = self.ac.rest(endpoint='turlu', action='create')
        with self.assertRaises(ValueError):
            res = self.ac.rest(endpoint='sessions', action='turlu')
        # test with labs : get
        a = self.ac.rest('labs', 'list')
        self.assertTrue(len(a) == 3)
        b = self.ac.rest('/labs', 'list')
        self.assertTrue(a == b)
        # test with labs: read
        c = self.ac.rest('labs', 'read', 'mainenlab')
        self.assertTrue([lab for lab in a if lab['name'] == 'mainenlab'][0] == c)
        d = self.ac.rest('labs', 'read',
                         'https://test.alyx.internationalbrainlab.org/labs/mainenlab')
        self.assertEqual(c, d)
        # test object creation and deletion with weighings
        wa = {'subject': 'flowers',
              'date_time': '2018-06-30T12:34:57',
              'weight': 22.2,
              'user': 'olivier'
              }
        a = self.ac.rest('weighings', 'create', wa)
        b = self.ac.rest('weighings', 'read', a['url'])
        self.assertEqual(a, b)
        self.ac.rest('weighings', 'delete', a['url'])

    def test_download_datasets_with_api(self):
        ac = self.ac  # easier to debug in console
        cache_dir = tempfile.mkdtemp()

        # Test 1: empty dir, dict mode
        dset = ac.get('/datasets/baa06831-a2a7-4d44-8bce-63a94ad24625')
        url = wc.dataset_record_to_url(dset)
        file_name = wc.http_download_file_list(url, username=par.HTTP_DATA_SERVER_LOGIN,
                                               password=par.HTTP_DATA_SERVER_PWD,
                                               verbose=True, cache_dir=cache_dir)
        # Test 2: empty dir, list mode
        dset = ac.get('/datasets?id=baa06831-a2a7-4d44-8bce-63a94ad24625')
        url = wc.dataset_record_to_url(dset)
        file_name = wc.http_download_file_list(url, username=par.HTTP_DATA_SERVER_LOGIN,
                                               password=par.HTTP_DATA_SERVER_PWD,
                                               verbose=True, cache_dir=cache_dir)
        self.assertTrue(os.path.isfile(file_name[0]))
        shutil.rmtree(cache_dir)

    def test_download_datasets(self):
        # test downloading a single file
        full_link_to_file = r'http://ibl.flatironinstitute.org/cortexlab/Subjects/MW49/2018' \
                            r'-05-11/1/cwResponse.choice.8dfae09d-15a4-489b-a440-18517bc6b67a.npy'
        file_name = wc.http_download_file(full_link_to_file, verbose=True,
                                          username=par.HTTP_DATA_SERVER_LOGIN,
                                          password=par.HTTP_DATA_SERVER_PWD)
        a = np.load(file_name)
        self.assertTrue(len(a) > 0)

        # test downloading a list of files
        links = [r'http://ibl.flatironinstitute.org/cortexlab/Subjects/MW49/2018'
                 r'-05-11/1/cwResponse.choice.8dfae09d-15a4-489b-a440-18517bc6b67a.npy',
                 r'http://ibl.flatironinstitute.org/cortexlab/Subjects/MW49/2018'
                 r'-05-11/1/cwResponse.times.63c1ad01-1ae2-47c3-b51b-50488403c24f.npy']
        file_list = wc.http_download_file_list(links, verbose=True,
                                               username=par.HTTP_DATA_SERVER_LOGIN,
                                               password=par.HTTP_DATA_SERVER_PWD)
        a = np.load(file_list[0])
        b = np.load(file_list[1])
        self.assertTrue(len(a) > 0)
        self.assertTrue(len(b) > 0)


if __name__ == '__main__':
    unittest.main()

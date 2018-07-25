import unittest
import numpy as np
import os
import ibllib.webclient as wc
import oneibl.params as par


class TestDownloadHTTP(unittest.TestCase):

    def setUp(self):
        # Init connection to the database
        self.ac = wc.AlyxClient(username=par.ALYX_LOGIN, password=par.ALYX_PWD,
                                base_url=par.BASE_URL)

    def test_download_datasets_with_api(self):
        # Test 1: empty dir, dict mode
        dset = self.ac.get('/datasets/6e1d0a00-d4c8-4de2-b483-53e0751a6933')
        url = wc.dataset_record_to_url(dset)
        file_name = wc.http_download_file_list(url, username=par.HTTP_DATA_SERVER_LOGIN,
                                               password=par.HTTP_DATA_SERVER_PWD,
                                               verbose=True, cache_dir=par.CACHE_DIR)
        self.assertTrue(file_name == [])

        # Test 2: empty dir, list mode
        dset = self.ac.get('/datasets?id=6e1d0a00-d4c8-4de2-b483-53e0751a6933')
        url = wc.dataset_record_to_url(dset)
        file_name = wc.http_download_file_list(url, username=par.HTTP_DATA_SERVER_LOGIN,
                                               password=par.HTTP_DATA_SERVER_PWD,
                                               verbose=True, cache_dir=par.CACHE_DIR)
        self.assertTrue(file_name == [])

        # Test 3: 1 file, 1 empty, dict mode
        dset = self.ac.get('/datasets/b916b777-2630-46fd-a545-09e18befde2e')  # returns a dict
        url = wc.dataset_record_to_url(dset)
        file_name = wc.http_download_file_list(url, username=par.HTTP_DATA_SERVER_LOGIN,
                                               password=par.HTTP_DATA_SERVER_PWD,
                                               verbose=True, cache_dir=par.CACHE_DIR)
        for fn in file_name:
            self.assertTrue(os.path.isfile(fn))
            os.remove(fn)

        # Test 4: 1 file, 1 empty, list mode
        dset = self.ac.get('/datasets?id=b916b777-2630-46fd-a545-09e18befde2e')  # returns a list
        url = wc.dataset_record_to_url(dset)
        file_name = wc.http_download_file_list(url, username=par.HTTP_DATA_SERVER_LOGIN,
                                               password=par.HTTP_DATA_SERVER_PWD,
                                               verbose=True, cache_dir=par.CACHE_DIR)
        for fn in file_name:
            self.assertTrue(os.path.isfile(fn))
            os.remove(fn)

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

import unittest
from one_ibl.utils import AlyxClient
import one_ibl.utils
import one_ibl.params as par
import os


class TestDownloadHTTP(unittest.TestCase):

    def setUp(self):
        # Init connection to the database
        self.ac = AlyxClient()
        self.ac.authenticate(username=par.ALYX_LOGIN, password=par.ALYX_PWD)

    def test_download_datasets(self):
        # Test 1: empty dir, dict mode
        dset = self.ac.get('/datasets/6e1d0a00-d4c8-4de2-b483-53e0751a6933')
        url = one_ibl.utils.dataset_record_to_url(dset)
        file_name = one_ibl.utils.http_download_file_list(url, username=par.HTTP_DATA_SERVER_LOGIN,
                                                          password=par.HTTP_DATA_SERVER_PWD,
                                                          verbose=True, cache_dir=par.CACHE_DIR)
        self.assertTrue(file_name == [])

        # Test 2: empty dir, list mode
        dset = self.ac.get('/datasets?id=6e1d0a00-d4c8-4de2-b483-53e0751a6933')
        url = one_ibl.utils.dataset_record_to_url(dset)
        file_name = one_ibl.utils.http_download_file_list(url, username=par.HTTP_DATA_SERVER_LOGIN,
                                                          password=par.HTTP_DATA_SERVER_PWD,
                                                          verbose=True, cache_dir=par.CACHE_DIR)
        self.assertTrue(file_name == [])

        # Test 3: 1 file, 1 empty, dict mode
        dset = self.ac.get('/datasets/b916b777-2630-46fd-a545-09e18befde2e')  # returns a dict
        url = one_ibl.utils.dataset_record_to_url(dset)
        file_name = one_ibl.utils.http_download_file_list(url, username=par.HTTP_DATA_SERVER_LOGIN,
                                                          password=par.HTTP_DATA_SERVER_PWD,
                                                          verbose=True, cache_dir=par.CACHE_DIR)
        for fn in file_name:
            self.assertTrue(os.path.isfile(fn))
            os.remove(fn)

        # Test 4: 1 file, 1 empty, list mode
        dset = self.ac.get('/datasets?id=b916b777-2630-46fd-a545-09e18befde2e')  # returns a list
        url = one_ibl.utils.dataset_record_to_url(dset)
        file_name = one_ibl.utils.http_download_file_list(url, username=par.HTTP_DATA_SERVER_LOGIN,
                                                          password=par.HTTP_DATA_SERVER_PWD,
                                                          verbose=True, cache_dir=par.CACHE_DIR)
        for fn in file_name:
            self.assertTrue(os.path.isfile(fn))
            os.remove(fn)


if __name__ == '__main__':
    unittest.main()

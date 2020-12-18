import unittest
from alf import folders
from pathlib import Path


class TestFolders(unittest.TestCase):

    def setUp(self):
        self.no_session = 'nothing/to/report/here'
        self.std_session = '/path/iblrig_data/Subjects/sub_name/1977-08-01/001'
        self.nonstd_session = 'path/to/iblrig_data/sub_name/1977-08-01/1'
        self.nonstd_file = 'path/to/iblrig_data/sub_name/1977-08-01/1/bla.bla'
        self.std_file = self.std_session + '/raw_behavior_data/_iblrig_taskData.raw.jsonable'

    def test__isdatetime(self):
        self.assertTrue(folders._isdatetime('1999-01-01'))
        self.assertFalse(folders._isdatetime('blablabla'))

    def test_session_path(self):
        self.assertIsNone(folders.session_path(self.no_session))
        self.assertEqual(folders.session_path(self.std_session),
                         str(Path(self.std_session)))
        self.assertEqual(folders.session_path(self.nonstd_session),
                         str(Path(self.nonstd_session)))
        self.assertEqual(folders.session_path(self.nonstd_file),
                         str(Path('path/to/iblrig_data/sub_name/1977-08-01/1')))
        self.assertEqual(folders.session_path(self.std_file),
                         str(Path(self.std_session)))

    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main(exit=False)

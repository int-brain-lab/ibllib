import unittest
import tempfile
from pathlib import Path
import shutil

import alf.folders


class TestsAlfPartsFilters(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.gettempdir()) / 'folderstest'
        self.tmpdir.mkdir(exist_ok=True)

    def test_remove_empty_folders(self):
        self.tmpdir.joinpath('empty0').mkdir(exist_ok=True)
        self.tmpdir.joinpath('full0').mkdir(exist_ok=True)
        self.tmpdir.joinpath('full0', 'file.txt').touch()
        self.assertTrue(len(list(self.tmpdir.glob('*'))) == 2)
        alf.folders.remove_empty_folders(self.tmpdir)
        self.assertTrue(len(list(self.tmpdir.glob('*'))) == 1)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

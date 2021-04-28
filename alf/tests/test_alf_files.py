import unittest
import tempfile
from pathlib import Path
import shutil

import alf.files


class TestsAlfPartsFilters(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.gettempdir()) / 'iotest'
        self.tmpdir.mkdir(exist_ok=True)

    def test_filter_by(self):
        files = [
            'noalf.file',
            '_ibl_trials.intervals.npy',
            '_ibl_trials.intervals_bpod.csv',
            'wheel.position.npy',
            'wheel.timestamps.npy',
            'wheelMoves.intervals.npy',
            '_namespace_obj.attr_timescale.raw.v12.ext']

        for f in files:
            (self.tmpdir / f).touch()

        # Test filter with None; should return files with no non-standard timescale
        alf_files, _ = alf.files.filter_by(self.tmpdir, timescale=None)
        expected = [
            'wheel.position.npy',
            'wheel.timestamps.npy',
            'wheelMoves.intervals.npy',
            '_ibl_trials.intervals.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter with None attribute')

        # Test filtering by object; should return only 'wheel' ALF objects
        alf_files, parts = alf.files.filter_by(self.tmpdir, object='wheel')
        expected = ['wheel.position.npy', 'wheel.timestamps.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter by object')
        self.assertEqual(len(alf_files), len(parts))

        # Test wildcards; should return 'wheel' and 'wheelMoves' ALF objects
        alf_files, _ = alf.files.filter_by(self.tmpdir, object='wh*')
        expected = ['wheel.position.npy', 'wheel.timestamps.npy', 'wheelMoves.intervals.npy']
        self.assertCountEqual(alf_files, expected, 'failed to filter with wildcard')

        # Test filtering by specific timescale; test parts returned
        alf_files, parts = alf.files.filter_by(self.tmpdir, timescale='bpod')
        expected = ['_ibl_trials.intervals_bpod.csv']
        self.assertEqual(alf_files, expected, 'failed to filter by timescale')
        expected = ('ibl', 'trials', 'intervals', 'bpod', None, 'csv')
        self.assertTupleEqual(parts[0], expected)
        self.assertEqual(len(parts[0]), len(alf.files.ALF_EXP.groupindex))
        self.assertEqual(parts[0][alf.files.ALF_EXP.groupindex['timescale'] - 1], 'bpod')

        # Test filtering multiple attributes; should return only trials intervals
        alf_files, _ = alf.files.filter_by(self.tmpdir, attribute='intervals', object='trials')
        expected = ['_ibl_trials.intervals.npy', '_ibl_trials.intervals_bpod.csv']
        self.assertCountEqual(alf_files, expected, 'failed to filter by multiple attribute')

        # Test returning only ALF files
        alf_files, _ = alf.files.filter_by(self.tmpdir)
        self.assertCountEqual(alf_files, files[1:], 'failed to return ALF files')

        # Test return empty
        out = alf.files.filter_by(self.tmpdir, object=None)
        self.assertEqual(out, ([], []))

        # Test extras
        alf_files, _ = alf.files.filter_by(self.tmpdir, extra='v12')
        expected = ['_namespace_obj.attr_timescale.raw.v12.ext']
        self.assertEqual(alf_files, expected, 'failed to filter extra attributes')

        alf_files, _ = alf.files.filter_by(self.tmpdir, extra=['v12', 'raw'])
        expected = ['_namespace_obj.attr_timescale.raw.v12.ext']
        self.assertEqual(alf_files, expected, 'failed to filter extra attributes as list')

        alf_files, _ = alf.files.filter_by(self.tmpdir, extra=['foo', 'v12'])
        self.assertEqual(alf_files, [], 'failed to filter extra attributes')

        # Assert kwarg validation; should raise TypeError
        with self.assertRaises(TypeError):
            alf.files.filter_by(self.tmpdir, unknown=None)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

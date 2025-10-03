import unittest
import tempfile

import numpy as np

from brainbox.io import one as bbone


class TestIO_ONE(unittest.TestCase):
    """Tests for brainbox.io.one functions that don't require fixtures on disk."""
    def test_load_iti(self):
        """Test for brainbox.io.one.load_iti function."""
        trials = bbone.alfio.AlfBunch({})
        trials.intervals = np.array([
            [114.52487625, 117.88103707],
            [118.5169474, 122.89742147],
            [123.49302927, 126.12216664],
            [126.68107337, 129.53872083],
            [130.11952807, 133.90539162]
        ])
        trials.stimOff_times = [117.38098379, 122.39736201, 125.62210278, 129.03865947, 133.4053633]
        expected = np.array([1.13596361, 1.09566726, 1.05897059, 1.0808686, np.nan])
        np.testing.assert_array_almost_equal(bbone.load_iti(trials), expected)
        _ = trials.pop('stimOff_times')
        self.assertRaises(ValueError, bbone.load_iti, trials)


from one.api import ONE
from uuid import UUID
from brainbox.io.one import FOVLoader
from one.alf.path import ALFPath

class TestFOVLoader(unittest.TestCase):
    def setUp(self):
        self.one = ONE()
        self.eid = '787b98a3-3176-42e0-8792-ee8c66cf45e1'  # SP072/2025-08-20/001
        self.fov_name = 'FOV_00'

    def test_init_with_name(self):
        loader = FOVLoader(one=self.one, eid=self.eid, name=self.fov_name)
        self.assertEqual(loader.name, self.fov_name)
        self.assertEqual(loader.number, 0)
        self.assertEqual(loader.eid, UUID(self.eid))
        self.assertIsInstance(loader.session_path, ALFPath)

    @unittest.skip('Requires a valid FOV ID from Alyx.')
    def test_init_with_id(self):
        loader = FOVLoader(one=self.one, id=self.fov_id)
        self.assertEqual(loader.id, self.fov_id)
        self.assertEqual(loader.name, self.fov_name)
        self.assertEqual(loader.number, 0)
        self.assertIsInstance(loader.session_path, ALFPath)

    def test_load_roi_times(self):
        loader = FOVLoader(one=self.one, eid=self.eid, name=self.fov_name)
        roi_times = loader.load_roi_times()
        self.assertIsInstance(roi_times, np.ndarray)
        self.assertGreater(roi_times.shape[0], 0)  # Should have some ROI times

    def test_load_roi_mlapdv(self):
        loader = FOVLoader(one=self.one, eid=self.eid, name=self.fov_name)
        roi_mlapdv = loader.load_roi_mlapdv()
        self.assertIsInstance(roi_mlapdv, np.ndarray)
        self.assertGreater(roi_mlapdv.shape[0], 0)  # Should have some ROI MLAPDV values

        # Test offline
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        session_path = ALFPath(tmp.name).joinpath(loader.session_path.session_path_short())
        fov_dir = session_path.joinpath('alf', self.fov_name)
        fov_dir.mkdir(parents=True, exist_ok=True)
        np.save(fov_dir.joinpath('mpciROIs.mlapdv.npy'), np.arange(9).reshape(-1, 3))
        np.save(fov_dir.joinpath('mpciROIs.mlapdv_estimate.npy'), np.arange(9, 18).reshape(-1, 3))
        one = ONE(mode='local', cache_dir=tmp.name)
        loader_offline = FOVLoader(one=one, name=self.fov_name, session_path=session_path)
        roi_mlapdv_offline = loader_offline.load_roi_mlapdv()
        np.testing.assert_array_equal(roi_mlapdv_offline, np.arange(9).reshape(-1, 3))
        roi_mlapdv_estimate_offline = loader_offline.load_roi_mlapdv(provenance=bbone.Provenance.ESTIMATE)
        np.testing.assert_array_equal(roi_mlapdv_estimate_offline, np.arange(9, 18).reshape(-1, 3))


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestFOVLoader('test_load_roi_mlapdv'))
    runner = unittest.TextTestRunner()
    runner.run(suite)


# if __name__ == '__main__':
    # unittest.main(exit=False, verbosity=2)

import unittest
import tempfile
from pathlib import Path

import numpy as np
import numpy.random as nr

# from ibllib.ephys.ephysalf import rename_to_alf


class TestsEphys(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        p = Path(self.tmp_dir.name)
        self.ns = 100
        self.nc = 10
        self.nt = 5
        np.save(p / 'spike_times.npy', np.cumsum(nr.exponential(size=self.ns)))
        np.save(p / 'spike_clusters.npy', nr.randint(low=0, high=10, size=self.ns))
        np.save(p / 'amplitudes.npy', nr.uniform(low=0.5, high=1.5, size=self.ns))
        np.save(p / 'channel_positions.npy', np.c_[np.arange(self.nc), np.zeros(self.nc)])
        np.save(p / 'templates.npy', np.random.normal(size=(self.nt, 50, self.nc)))
        np.save(p / 'channel_map.npy', np.c_[np.arange(self.nc)])

    def _load(self, fn):
        return np.load(Path(self.tmp_dir.name) / fn)

    def test_ephys_1(self):
        self.assertTrue(self._load('spike_times.npy').shape == (self.ns,))

    def tearDown(self):
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main(exit=False)

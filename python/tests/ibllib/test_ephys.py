import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import numpy.random as nr

from ibllib.ephys.ephysalf import rename_to_alf, _FILE_RENAMES


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
        np.save(p / 'rawdata.npy', np.random.normal(size=(1000, self.nc)))
        self.files = os.listdir(self.tmp_dir.name)

    def _load(self, fn):
        return np.load(Path(self.tmp_dir.name) / fn)

    def test_ephys_1(self):
        self.assertTrue(self._load('spike_times.npy').shape == (self.ns,))
        self.assertTrue(self._load('spike_clusters.npy').shape == (self.ns,))
        self.assertTrue(self._load('amplitudes.npy').shape == (self.ns,))
        self.assertTrue(self._load('channel_positions.npy').shape == (self.nc, 2))
        self.assertTrue(self._load('templates.npy').shape == (self.nt, 50, self.nc))
        self.assertTrue(self._load('channel_map.npy').shape == (self.nc, 1))
        self.assertTrue(self._load('rawdata.npy').shape == (1000, self.nc))

    def test_ephys_rename(self):
        tn = self.tmp_dir.name
        rename_to_alf(tn, rawfile='rawdata.npy')
        for old, new in _FILE_RENAMES:
            assert not (Path(tn) / old).exists()
            if old in self.files:
                assert (Path(tn) / new).exists()

    def tearDown(self):
        self.tmp_dir.cleanup()


if __name__ == "__main__":
    unittest.main(exit=False)

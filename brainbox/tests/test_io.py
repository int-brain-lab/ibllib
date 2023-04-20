import json
from pathlib import Path
import unittest
import tempfile
import shutil

import numpy as np
import numpy.testing

from brainbox.io import one as bbone
from one.api import ONE
from one.alf.cache import make_parquet_db


class TestIO_ALF(unittest.TestCase):

    def setUp(self) -> None:
        """
        Creates a mock ephys alf data folder with minimal spikes info for loading
        :return:
        """
        self.tmpdir = Path(tempfile.gettempdir()) / 'test_bbio'
        self.tmpdir.mkdir(exist_ok=True)
        self.one = ONE(mode='local', cache_dir=self.tmpdir)
        self.alf_path = self.tmpdir.joinpath(
            'lab', 'Subjects', 'subject', '2019-08-12', '001', 'alf'
        )
        self.session_path = self.alf_path.parent
        self.probes = ['probe00', 'probe01']
        nspi = [10000, 10001]
        nclusters = [420, 421]
        nch = [64, 62]
        probe_description = []
        for i, prb in enumerate(self.probes):
            prb_path = self.alf_path.joinpath(prb)
            prb_path.mkdir(exist_ok=True, parents=True)
            np.save(prb_path.joinpath('clusters.channels.npy'),
                    np.random.randint(0, nch[i], nclusters[i]))
            np.save(prb_path.joinpath('spikes.depths.npy'),
                    np.random.rand(nspi[i]) * 3200)
            np.save(prb_path.joinpath('spikes.clusters.npy'),
                    np.random.randint(0, nclusters[i], nspi[i]))
            np.save(prb_path.joinpath('spikes.times.npy'),
                    np.sort(np.random.random(nspi[i]) * 1200))

            probe_description.append({'label': prb,
                                      'model': '3B1',
                                      'serial': int(456248),
                                      'raw_file_name': 'gnagnagnaga',
                                      })
        # create session level
        with open(self.alf_path.joinpath('probes.description.json'), 'w+') as fid:
            fid.write(json.dumps(probe_description))
        np.save(self.alf_path.joinpath('trials.gnagnag.npy'), np.random.rand(50))
        # Add some fake records to the cache
        if not self.one.search(subject='subject', date='2019-08-12', number='001'):
            make_parquet_db(self.tmpdir)
            self.one.load_cache(cache_dir=self.tmpdir)

    def test_load_ephys(self):
        # straight test
        spikes, clusters, trials = bbone.load_ephys_session(self.session_path, one=self.one)
        self.assertTrue(set(spikes.keys()) == set(self.probes))
        self.assertTrue(set(clusters.keys()) == set(self.probes))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


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


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

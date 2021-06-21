import json
from pathlib import Path
import unittest
import tempfile
import shutil

import numpy as np

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
            self.one._load_cache(cache_dir=self.tmpdir)

    def test_load_ephys(self):
        # straight test
        spikes, clusters, trials = bbone.load_ephys_session(self.session_path, one=self.one)
        self.assertTrue(list(spikes.keys()) == self.probes)
        self.assertTrue(list(clusters.keys()) == self.probes)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

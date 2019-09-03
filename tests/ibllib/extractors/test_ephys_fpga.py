import unittest
import tempfile
from pathlib import Path

import numpy as np

import ibllib.io.spikeglx as spikeglx
from ibllib.io.extractors import ephys_fpga


class TestsFolderStructure(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        pl = Path(self.dir.name) / 'raw_ephys_data' / 'probe_left'
        pr = Path(self.dir.name) / 'raw_ephys_data' / 'probe_right'
        pl.mkdir(parents=True)
        pr.mkdir(parents=True)
        (pl / 'iblrig_ephysData.raw_g0_t0.imec.lf.bin').touch()
        (pl / 'iblrig_ephysData.raw_g0_t0.imec.ap.bin').touch()
        (pr / 'iblrig_ephysData.raw_g0_t0.imec.lf.bin').touch()
        (pr / 'iblrig_ephysData.raw_g0_t0.imec.ap.bin').touch()

    def test_get_ephys_files(self):
        # first test at the root directory level, with a string input
        ephys_files = spikeglx.glob_ephys_files(self.dir.name)
        for ef in ephys_files:
            self.assertTrue(ef.label in ['probe_right', 'probe_left'])
            self.assertTrue(ef.ap.exists() and ef.lf.exists())
        # second test at the ephys directory level, with a pathlib.Path input
        ephys_files = spikeglx.glob_ephys_files(Path(self.dir.name) / 'raw_ephys_data')
        for ef in ephys_files:
            self.assertTrue(ef.label in ['probe_right', 'probe_left'])
            self.assertTrue(ef.ap.exists() and ef.lf.exists())

    def tearDown(self):
        self.dir.cleanup()


class TestSyncExtraction(unittest.TestCase):

    def setUp(self):
        self.workdir = Path(__file__).parents[1] / 'fixtures' / 'io' / 'spikeglx'
        self.meta_files = list(Path.glob(self.workdir, '*.meta'))

    def test_sync_nidq(self):
        self.sync_gen(fn='sample3B_g0_t0.nidq.meta', ns=32, nc=2, sync_depth=8)

    def test_sync_3B(self):
        self.sync_gen(fn='sample3B_g0_t0.imec1.ap.meta', ns=32, nc=385, sync_depth=16)

    def test_sync_3A(self):
        self.sync_gen(fn='sample3A_g0_t0.imec.ap.meta', ns=32, nc=385, sync_depth=16)

    def sync_gen(self, fn, ns, nc, sync_depth):
        # nidq has 1 analog and 1 digital sync channels
        with tempfile.TemporaryDirectory() as tdir:
            ses_path = Path(tdir).joinpath('raw_ephys_data')
            ses_path.mkdir(parents=True, exist_ok=True)
            nidq = spikeglx._mock_spikeglx_file(ses_path, self.workdir / fn,
                                                ns=ns, nc=nc, sync_depth=sync_depth)
            syncs = ephys_fpga.extract_sync(tdir, save=True)
            self.assertTrue(np.all(syncs[0].channels[slice(0, None, 2)] ==
                                   np.arange(0, nidq['sync_depth'])))
            with self.assertLogs(level='INFO') as log:
                syncs = ephys_fpga.extract_sync(tdir)
                self.assertEqual(len(log.output), 1)
                self.assertIn('SGLX sync found', log.output[0])

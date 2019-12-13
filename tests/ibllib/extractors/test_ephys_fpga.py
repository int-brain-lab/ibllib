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
            bin_file = ses_path.joinpath(fn).with_suffix('.bin')
            nidq = spikeglx._mock_spikeglx_file(bin_file, self.workdir / fn,
                                                ns=ns, nc=nc, sync_depth=sync_depth)
            syncs = ephys_fpga.extract_sync(tdir, save=True)
            self.assertTrue(np.all(syncs[0].channels[slice(0, None, 2)] ==
                                   np.arange(0, nidq['sync_depth'])))
            with self.assertLogs(level='INFO') as log:
                syncs = ephys_fpga.extract_sync(tdir)
                self.assertEqual(len(log.output), 1)
                self.assertIn('SGLX sync found', log.output[0])


class TestIblChannelMaps(unittest.TestCase):

    def setUp(self):
        self.workdir = Path(__file__).parents[1] / 'fixtures'

    def test_ibl_sync_maps(self):
        s = ephys_fpga.get_ibl_sync_map({'ap': 'toto', 'path': self.workdir}, '3A')
        self.assertEqual(s, ephys_fpga.CHMAPS['3A']['ap'])
        s = ephys_fpga.get_ibl_sync_map({'nidq': 'toto', 'path': self.workdir}, '3B')
        self.assertEqual(s, ephys_fpga.CHMAPS['3B']['nidq'])
        s = ephys_fpga.get_ibl_sync_map({'ap': 'toto', 'path': self.workdir}, '3B')
        self.assertEqual(s, ephys_fpga.CHMAPS['3B']['ap'])


class TestWheelExtraction(unittest.TestCase):

    def setUp(self) -> None:
        self.ta = np.array([2, 4, 6, 8, 12, 14, 16, 18])
        self.pa = np.array([1, -1, 1, -1, 1, -1, 1, -1])
        self.tb = np.array([3, 5, 7, 9, 11, 13, 15, 17])
        self.pb = np.array([1, -1, 1, -1, 1, -1, 1, -1])

    def test_x1_decoding(self):
        p_ = np.array([1, 2, 1, 0])
        t_ = np.array([2, 6, 11, 15])
        t, p = ephys_fpga._rotary_encoder_positions_from_fronts(
            self.ta, self.pa, self.tb, self.pb, ticks=np.pi * 2, coding='x1')
        self.assertTrue(np.all(t == t_))
        self.assertTrue(np.all(p == p_))

    def test_x4_decoding(self):
        p_ = np.array([1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1, 0]) / 4
        t_ = np.array([2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18])
        t, p = ephys_fpga._rotary_encoder_positions_from_fronts(
            self.ta, self.pa, self.tb, self.pb, ticks=np.pi * 2, coding='x4')
        self.assertTrue(np.all(t == t_))
        self.assertTrue(np.all(np.isclose(p, p_)))

    def test_x2_decoding(self):
        p_ = np.array([1, 2, 3, 4, 3, 2, 1, 0]) / 2
        t_ = np.array([2, 4, 6, 8, 12, 14, 16, 18])
        t, p = ephys_fpga._rotary_encoder_positions_from_fronts(
            self.ta, self.pa, self.tb, self.pb, ticks=np.pi * 2, coding='x2')
        self.assertTrue(np.all(t == t_))
        self.assertTrue(np.all(p == p_))

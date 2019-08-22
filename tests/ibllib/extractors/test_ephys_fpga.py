import unittest
import tempfile
import json
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
                self.assertIn('spike GLX sync found', log.output[0])


class TestsHardwareParameters(unittest.TestCase):

    def setUp(self):
        self.tdir = tempfile.TemporaryDirectory()
        self.dir = Path(self.tdir.name)
        self.par = {'SYSTEM': "3A",
                    'SYNC_WIRING':
                        {
                            "pin03": "left_camera",
                            "pin04": "right_camera",
                            "pin05": "Camera level shifter board ground",
                            "pin06": "body_camera",
                            "pin09": "bpod",
                            "pin10": "bpod ground",
                            "pin15": "frame2ttl",
                            "pin16": "frame2ttl ground",
                            "pin17": "rotary_encoder_0",
                            "pin18": "rotary_encoder_1",
                            "pin19": "audio",
                            "pin20": "audio",
                            "pin23": "rotary_encoder ground",
                        }
                    }
        self.map = {'left_camera': 2,
                    'right_camera': 3,
                    'body_camera': 4,
                    'bpod': 7,
                    'frame2ttl ground': 12,
                    'rotary_encoder_0': 13,
                    'rotary_encoder_1': 14,
                    'audio': 15}
        self.file_json = Path(self.dir) / 'neuropixel_wirings.json'
        with open(self.file_json, 'w+') as fid:
            fid.write(json.dumps(self.par, indent=1))

    def test_get_parameters(self):
        # get params providing full file path
        par = ephys_fpga.get_hardware_config(self.file_json)
        self.assertEqual(par, self.par)
        # get params providing directory path
        par = ephys_fpga.get_hardware_config(self.file_json.parent)
        self.assertEqual(par, self.par)

    def test_get_channel_map(self):
        map = ephys_fpga.get_sync_map(self.file_json)
        self.assertEqual(map, self.map)
        map = ephys_fpga._sync_map_from_hardware_config(self.par)
        self.assertEqual(map, self.map)
        map = ephys_fpga.get_sync_map(self.dir / 'idontexist.json')
        self.assertEqual(map, ephys_fpga.SYNC_CHANNEL_MAP)

    def tearDown(self):
        self.tdir.cleanup()

import unittest
import tempfile
import shutil
import json
from pathlib import Path

import ibllib.io.spikeglx
from ibllib.io.extractors import ephys_fpga


class TestsFolderStructure(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory().name
        pl = Path(self.dir) / 'raw_ephys_data' / 'probe_left'
        pr = Path(self.dir) / 'raw_ephys_data' / 'probe_right'
        pl.mkdir(parents=True)
        pr.mkdir(parents=True)
        (pl / 'iblrig_ephysData.raw_g0_t0.imec.lf.bin').touch()
        (pl / 'iblrig_ephysData.raw_g0_t0.imec.ap.bin').touch()
        (pr / 'iblrig_ephysData.raw_g0_t0.imec.lf.bin').touch()
        (pr / 'iblrig_ephysData.raw_g0_t0.imec.ap.bin').touch()

    def test_get_ephys_files(self):
        # first test at the root directory level, with a string input
        ephys_files = ibllib.io.spikeglx.glob_ephys_files(self.dir)
        for ef in ephys_files:
            self.assertTrue(ef.label in ['probe_right', 'probe_left'])
            self.assertTrue(ef.ap.exists() and ef.lf.exists())
        # second test at the ephys directory level, with a pathlib.Path input
        ephys_files = ibllib.io.spikeglx.glob_ephys_files(Path(self.dir) / 'raw_ephys_data')
        for ef in ephys_files:
            self.assertTrue(ef.label in ['probe_right', 'probe_left'])
            self.assertTrue(ef.ap.exists() and ef.lf.exists())

    def tearDown(self):
        shutil.rmtree(self.dir)


class TestsHardwareParameters(unittest.TestCase):

    def setUp(self):
        self.dir = Path(tempfile.TemporaryDirectory().name)
        self.dir.mkdir()
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
        shutil.rmtree(self.dir)

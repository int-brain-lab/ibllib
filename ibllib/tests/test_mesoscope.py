"""Tests for ibllib.pipes.mesoscope_tasks"""
import sys
import unittest
from unittest.mock import MagicMock
import tempfile
import json
from pathlib import Path
import numpy as np

from ibllib.pipes.mesoscope_tasks import MesoscopePreprocess, MesoscopeFOV

# Mock suit2p which is imported in MesoscopePreprocess
attrs = {'default_ops.return_value': {}}
sys.modules['suite2p'] = MagicMock(**attrs)


class TestMesoscopePreprocess(unittest.TestCase):
    """Test for MesoscopePreprocess task."""

    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.session_path = Path(self.td.name).joinpath('subject', '2020-01-01', '001')
        self.img_path = self.session_path.joinpath('raw_imaging_data_00')
        self.img_path.mkdir(parents=True)
        self.task = MesoscopePreprocess(self.session_path)

    def test_meta(self):
        """
        Test arguments that are overwritten by meta file and set in task.kwargs,
        and that explicitly passed kwargs overwrite default and meta args
        """
        expected = {
            'data_path': [str(self.img_path)],
            'fast_disk': '',
            'num_workers': 4,
            'save_path0': str(self.session_path.joinpath('alf')),
            'move_bin': True,
            'keep_movie_raw': False,
            'delete_bin': False,
            'batch_size': 500,
            'combined': False,
            'look_one_level_down': False,
            'num_workers_roi': -1,
            'nimg_init': 400,
            'nonrigid': True,
            'maxregshift': 0.05,
            'denoise': 1,
            'block_size': [128, 128],
            'save_mat': True,
            'scalefactor': 1,
            'mesoscan': True,
            'nplanes': 1,
            'tau': 1.5,
            'functional_chan': 1,
            'align_by_chan': 1,
            'nrois': 1,
            'nchannels': 1,
            'fs': 6.8,
            'lines': [[3, 4, 5]],
            'dx': np.array([0], dtype=int),
            'dy': np.array([0], dtype=int),
        }

        meta = {
            'scanImageParams': {'hStackManager': {'zs': 320},
                                'hRoiManager': {'scanVolumeRate': 6.8}},
            'FOV': [{'topLeftDeg': [-1, 1.3], 'topRightDeg': [3, 1.3], 'bottomLeftDeg': [-1, 5.2],
                     'nXnYnZ': [512, 512, 1], 'channelIdx': 2, 'lineIdx': [4, 5, 6]}]
        }
        with open(self.img_path.joinpath('_ibl_rawImagingData.meta.json'), 'w') as f:
            json.dump(meta, f)
        self.img_path.joinpath('test.tif').touch()
        _ = self.task.run(run_suite2p=False, rename_files=False)
        self.assertEqual(self.task.status, 0)
        self.assertDictEqual(self.task.kwargs, {**expected})
        # Now overwrite a specific option with task.run kwarg
        _ = self.task.run(run_suite2p=False, rename_files=False, nchannels=2, delete_bin=True)
        self.assertEqual(self.task.status, 0)
        self.assertEqual(self.task.kwargs['nchannels'], 2)
        self.assertEqual(self.task.kwargs['delete_bin'], True)
        with open(self.img_path.joinpath('_ibl_rawImagingData.meta.json'), 'w') as f:
            json.dump({}, f)

    def tearDown(self) -> None:
        self.td.cleanup()


class TestMesoscopeFOV(unittest.TestCase):
    """Test for MesoscopeFOV task."""

    def test_get_provenance(self):
        filename = 'mpciMeanImage.mlapdv_estimate.npy'
        provenance = MesoscopeFOV.get_provenance(filename)
        self.assertEqual('ESTIMATE', provenance.name)
        filename = 'mpciROIs.brainLocation_ccf_2017.npy'
        provenance = MesoscopeFOV.get_provenance(filename)
        self.assertEqual('HISTOLOGY', provenance.name)

import sys
import unittest
from unittest.mock import MagicMock
import tempfile
import json
from pathlib import Path
import numpy as np

from ibllib.pipes.mesoscope_tasks import MesoscopePreprocess

# Mock suit2p which is imported in MesoscopePreprocess
attrs = {'default_ops.return_value': {}}
sys.modules['suite2p'] = MagicMock(**attrs)

class TestMesoscopePreprocess(unittest.TestCase):

    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.session_path = Path(self.td.name).joinpath('subject', 'date', 'number')
        self.img_path = self.session_path.joinpath('raw_imaging_data')
        self.img_path.mkdir(parents=True)
        with open(self.img_path.joinpath('rawImagingData.meta.json'), 'w') as f:
            json.dump({}, f)
        self.defaults = {
            'data_path': [str(self.img_path)],
            'save_path0': str(self.session_path.joinpath('alf')),
            'move_bin': True,
            'keep_movie_raw': False,
            'delete_bin': False,
            'batch_size': 1000,
            'combined': True
        }
        self.task = MesoscopePreprocess(self.session_path)

    def test_rename_files(self):
        """Test that the files and subdirectories are renamed correctly"""
        alf_dir = self.session_path.joinpath('alf')
        suite2p_dir = alf_dir.joinpath('suite2p')
        expected_files = ['stat.npy', 'ops.npy', 'data.bin', 'mpci.ROIActivityF.npy', 'mpci.ROIActivityFneu.npy',
                          'mpci.ROIActivityDeconvolved.npy','mpciROIs.included.npy', 'mpci.validFrames.npy']
        expected_outputs = [alf_dir.joinpath(s, e) for e in expected_files for s in ['fov00', 'fov01', 'fov_combined']
                            if not (s == 'fov_combined' and e == 'data.bin')]
        for subdir in ['plane00', 'plane01', 'combined']:
            suite2p_dir.joinpath(subdir).mkdir(parents=True)
            # create a bunch of empty files to rename
            for file in ['stat.npy', 'F.npy', 'Fneu.npy', 'spks.npy', 'iscell.npy']:
                suite2p_dir.joinpath(subdir, file).touch()
                if subdir != 'combined':
                    suite2p_dir.joinpath(subdir, 'data.bin').touch()
            # Create ops with data to make valid frames file
            np.save(suite2p_dir.joinpath(subdir, 'ops.npy'), {'badframes': np.array([True, False, False])}, allow_pickle=True)
        with self.assertLogs('ibllib.pipes.mesoscope_tasks', level='WARNING'):
            _ = self.task.run(run_suite2p=False, rename_files=True)
        self.assertCountEqual(self.task.outputs, expected_outputs)
        self.assertEqual(self.task.status, 0)
        self.assertFalse(suite2p_dir.exists())

    def test_defaults(self):
        """Test that the defaults are set correctly and reflected in task.kwargs"""
        missing_keys = ['nrois', 'mesoscan', 'nplanes', 'nchannels', 'tau', 'fs', 'dx', 'dy', 'lines']
        with self.assertLogs('ibllib.pipes.mesoscope_tasks', level='WARNING') as capture:
            _ = self.task.run(run_suite2p=False, rename_files=False)
            self.assertEqual(len(capture.records), len(missing_keys))
            for i in range(len(missing_keys)):
                self.assertEqual(capture.records[i].getMessage(),
                                 f"Setting for {missing_keys[i]} not found in metadata file. Keeping default.")
        self.assertEqual(self.task.status, 0)
        self.assertCountEqual(self.task.kwargs, self.defaults)

    def test_meta(self):
        """
        Test arguments that are overwritten by meta file and set in task.kwargs,
        and that explicitly passed kwargs overwrite default and meta args
        """
        meta = {
            'nrois': 6,
            'mesoscan': True,
            'nplanes': 1,
            'nchannels': 1,
            'tau': 1.5,
            'fs': 7.1,
            'dx': [1, 2, 3],
            'dy': [4, 5, 6],
            'lines': [7, 8, 9, 10]
        }
        with open(self.img_path.joinpath('rawImagingData.meta.json'), 'w') as f:
            json.dump(meta, f)
        _ = self.task.run(run_suite2p=False, rename_files=False)
        self.assertEqual(self.task.status, 0)
        self.assertCountEqual(self.task.kwargs, {**self.defaults, **meta})
        # Now overwrite a specific option with task.run kwarg
        _ = self.task.run(run_suite2p=False, rename_files=False, nchannels=2, delete_bin=True)
        self.assertEqual(self.task.status, 0)
        self.assertEqual(self.task.kwargs['nchannels'], 2)
        self.assertEqual(self.task.kwargs['delete_bin'], True)
        with open(self.img_path.joinpath('rawImagingData.meta.json'), 'w') as f:
            json.dump({}, f)

    def tearDown(self) -> None:
        self.td.cleanup()


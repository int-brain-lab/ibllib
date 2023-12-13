"""Tests for ibllib.pipes.mesoscope_tasks"""
import sys
import unittest
from unittest import mock
import tempfile
import json
from itertools import chain
from pathlib import Path

from one.api import ONE
import numpy as np

from ibllib.pipes.mesoscope_tasks import MesoscopePreprocess, MesoscopeFOV, \
    find_triangle, surface_normal, _nearest_neighbour_1d
from ibllib.io.extractors import mesoscope
from ibllib.tests import TEST_DB

# Mock suit2p which is imported in MesoscopePreprocess
attrs = {'default_ops.return_value': {}}
sys.modules['suite2p'] = mock.MagicMock(**attrs)


class TestMesoscopePreprocess(unittest.TestCase):
    """Test for MesoscopePreprocess task."""

    def setUp(self) -> None:
        self.td = tempfile.TemporaryDirectory()
        self.session_path = Path(self.td.name).joinpath('subject', '2020-01-01', '001')
        self.img_path = self.session_path.joinpath('raw_imaging_data_00')
        self.img_path.mkdir(parents=True)
        self.task = MesoscopePreprocess(self.session_path, one=ONE(**TEST_DB))

    def test_meta(self):
        """
        Test arguments that are overwritten by meta file and set in task.kwargs,
        and that explicitly passed kwargs overwrite default and meta args
        """
        expected = {
            'data_path': [str(self.img_path)],
            'save_path0': str(self.session_path.joinpath('alf')),
            'fast_disk': '',
            'look_one_level_down': False,
            'num_workers': -1,
            'num_workers_roi': -1,
            'keep_movie_raw': False,
            'delete_bin': False,
            'batch_size': 500,
            'nimg_init': 400,
            'combined': False,
            'nonrigid': True,
            'maxregshift': 0.05,
            'denoise': 1,
            'block_size': [128, 128],
            'save_mat': True,
            'move_bin': True,
            'mesoscan': True,
            'nplanes': 1,
            'tau': 1.5,
            'functional_chan': 1,
            'align_by_chan': 1,
            'nrois': 1,
            'nchannels': 1,
            'fs': 6.8,
            'lines': [[3, 4, 5]],
            'slices': [0],
            'dx': np.array([0], dtype=int),
            'dy': np.array([0], dtype=int),
        }

        meta = {
            'scanImageParams': {'hStackManager': {'zs': 320},
                                'hRoiManager': {'scanVolumeRate': 6.8}},
            'FOV': [{'topLeftDeg': [-1, 1.3], 'topRightDeg': [3, 1.3], 'bottomLeftDeg': [-1, 5.2],
                     'nXnYnZ': [512, 512, 1], 'channelIdx': 2, 'lineIdx': [4, 5, 6], 'slice_id': 0}]
        }
        with open(self.img_path.joinpath('_ibl_rawImagingData.meta.json'), 'w') as f:
            json.dump(meta, f)
        self.img_path.joinpath('test.tif').touch()
        with mock.patch.object(self.task, 'get_default_tau', return_value=1.5):
            _ = self.task.run(run_suite2p=False, rename_files=False)
        self.assertEqual(self.task.status, 0)
        self.assertDictEqual(self.task.kwargs, expected)
        # {k: v for k, v in self.task.kwargs.items() if expected[k] != v}
        # Now overwrite a specific option with task.run kwarg
        with mock.patch.object(self.task, 'get_default_tau', return_value=1.5):
            _ = self.task.run(run_suite2p=False, rename_files=False, nchannels=2, delete_bin=True)
        self.assertEqual(self.task.status, 0)
        self.assertEqual(self.task.kwargs['nchannels'], 2)
        self.assertEqual(self.task.kwargs['delete_bin'], True)
        with open(self.img_path.joinpath('_ibl_rawImagingData.meta.json'), 'w') as f:
            json.dump({}, f)

    def test_get_default_tau(self):
        """Test for MesoscopePreprocess.get_default_tau method."""
        subject_detail = {'genotype': [{'allele': 'Cdh23', 'zygosity': 1},
                                       {'allele': 'Ai95-G6f', 'zygosity': 1},
                                       {'allele': 'Camk2a-tTa', 'zygosity': 1}]}
        with mock.patch.object(self.task.one.alyx, 'rest', return_value=subject_detail):
            self.assertEqual(self.task.get_default_tau(), .7)
            subject_detail['genotype'].pop(1)
            self.assertEqual(self.task.get_default_tau(), 1.5)  # return the default value

    def tearDown(self) -> None:
        self.td.cleanup()


class TestMesoscopeFOV(unittest.TestCase):
    """Test for MesoscopeFOV task and associated functions."""

    def test_get_provenance(self):
        """Test for MesoscopeFOV.get_provenance method."""
        filename = 'mpciMeanImage.mlapdv_estimate.npy'
        provenance = MesoscopeFOV.get_provenance(filename)
        self.assertEqual('ESTIMATE', provenance.name)
        filename = 'mpciROIs.brainLocation_ccf_2017.npy'
        provenance = MesoscopeFOV.get_provenance(filename)
        self.assertEqual('HISTOLOGY', provenance.name)

    def test_find_triangle(self):
        """Test for find_triangle function."""
        points = np.array([[2.435, -3.37], [2.435, -1.82], [2.635, -2.], [2.535, -1.7]])
        connectivity_list = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]], dtype=np.intp)
        point = np.array([2.6, -1.9])
        self.assertEqual(1, find_triangle(point, points, connectivity_list))
        point = np.array([3., 1.])  # outside of defined vertices
        self.assertEqual(-1, find_triangle(point, points, connectivity_list))

    def test_surface_normal(self):
        """Test for surface_normal function."""
        vertices = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]])
        expected = np.array([0, 0, 1])
        np.testing.assert_almost_equal(surface_normal(vertices), expected)

        # Test against multiple triangles
        vertices = np.r_[vertices[np.newaxis, :, :], [[[0, 0, 0], [0, 2, 0], [2, 0, 0]]]]
        expected = np.array([[0, 0, 1], [0, 0, -1]])
        np.testing.assert_almost_equal(surface_normal(vertices), expected)

        # Some real data
        vertices = np.array([[2.435, -1.82, -0.53], [2.635, -2., -0.58], [2.535, -1.7, -0.58]])
        expected = np.array([0.33424239, 0.11141413, 0.93587869])
        np.testing.assert_almost_equal(surface_normal(vertices), expected)

        # Test input validation
        self.assertRaises(ValueError, surface_normal, np.array([[1, 2, 3, 4]]))

    def test_nearest_neighbour_1d(self):
        """Test for _nearest_neighbour_1d function."""
        x = np.array([2., 1., 4., 5., 3.])
        x_new = np.array([-3, 0, 1.2, 3, 3, 2.5, 4.7, 6])
        val, ind = _nearest_neighbour_1d(x, x_new)
        np.testing.assert_array_equal(val, [1., 1., 1., 3., 3., 2., 5., 5.])
        np.testing.assert_array_equal(ind, [1, 1, 1, 4, 4, 0, 3, 3])

    def test_update_surgery_json(self):
        """Test for MesoscopeFOV.update_surgery_json method.

        Here we mock the Alyx object and simply check the method's calls.
        """
        one = ONE(**TEST_DB)
        task = MesoscopeFOV('/foo/bar/subject/2020-01-01/001', one=one)
        record = {'json': {'craniotomy_00': {'center': [1., -3.]}, 'craniotomy_01': {'center': [2.7, -1.3]}}}
        normal_vector = np.array([0.5, 1., 0.])
        meta = {'centerMM': {'ML': 2.7, 'AP': -1.30000000001}}
        with mock.patch.object(one.alyx, 'rest', return_value=[record, {}]), \
                mock.patch.object(one.alyx, 'json_field_update') as mock_rest:
            task.update_surgery_json(meta, normal_vector)
            expected = {'craniotomy_01': {'center': [2.7, -1.3],
                                          'surface_normal_unit_vector': (0.5, 1., 0.)}}
            mock_rest.assert_called_once_with('subjects', 'subject', data=expected)

        # Check errors and warnings
        # No matching craniotomy center
        with self.assertLogs('ibllib.pipes.mesoscope_tasks', 'ERROR'), \
                mock.patch.object(one.alyx, 'rest', return_value=[record, {}]):
            task.update_surgery_json({'centerMM': {'ML': 0., 'AP': 0.}}, normal_vector)
        # No matching surgery records
        with self.assertLogs('ibllib.pipes.mesoscope_tasks', 'ERROR'), \
                mock.patch.object(one.alyx, 'rest', return_value=[]):
            task.update_surgery_json(meta, normal_vector)
        # ONE offline
        one.mode = 'local'
        try:
            with self.assertLogs('ibllib.pipes.mesoscope_tasks', 'WARNING'):
                task.update_surgery_json(meta, normal_vector)
        finally:
            # ONE function is cached so we must reset the mode for other tests
            one.mode = 'auto'


class TestRegisterFOV(unittest.TestCase):
    """Test for MesoscopeFOV.register_fov method."""

    def setUp(self) -> None:
        self.one = ONE(**TEST_DB)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        self.session_path = Path(tmpdir.name, 'subject', '2020-01-01', '001')
        self.session_path.joinpath('alf', 'FOV_00').mkdir(parents=True)
        filename = self.session_path.joinpath('alf', 'FOV_00', 'mpciMeanImage.brainLocationIds_ccf_2017_estimate.npy')
        np.save(filename, np.array([0, 1, 2, 2, 4, 7], dtype=int))

    def test_register_fov(self):
        """Test MesoscopeFOV.register_fov method.

        Note this doesn't actually hit Alyx.  Also this doesn't test stack creation.
        """
        task = MesoscopeFOV(self.session_path, device_collection='raw_imaging_data', one=self.one)
        mlapdv = {'topLeft': [2317.2, -1599.8, -535.5], 'topRight': [2862.7, -1625.2, -748.7],
                  'bottomLeft': [2317.3, -2181.4, -466.3], 'bottomRight': [2862.7, -2206.9, -679.4],
                  'center': [2596.1, -1900.5, -588.6]}
        meta = {'FOV': [{'MLAPDV': mlapdv, 'nXnYnZ': [512, 512, 1], 'roiUUID': 0}]}
        with unittest.mock.patch.object(task.one.alyx, 'rest') as mock_rest:
            task.register_fov(meta, 'estimate')
        calls = mock_rest.call_args_list
        self.assertEqual(3, len(calls))

        args, kwargs = calls[1]
        self.assertEqual(('fields-of-view', 'create'), args)
        expected = {'data': {'session': None, 'imaging_type': 'mesoscope', 'name': 'FOV_00', 'stack': None}}
        self.assertEqual(expected, kwargs)

        args, kwargs = calls[2]
        self.assertEqual(('fov-location', 'create'), args)
        expected = ['field_of_view', 'default_provenance', 'coordinate_system', 'n_xyz', 'provenance', 'x', 'y', 'z',
                    'brain_region']
        self.assertCountEqual(expected, kwargs.get('data', {}).keys())
        self.assertEqual(5, len(kwargs['data']['brain_region']))
        self.assertEqual([512, 512, 1], kwargs['data']['n_xyz'])
        self.assertIs(kwargs['data']['field_of_view'], mock_rest().get('id'))
        self.assertEqual('E', kwargs['data']['provenance'])
        self.assertEqual([2317.2, 2862.7, 2317.3, 2862.7], kwargs['data']['x'])

        # Check dry mode with suffix input = None
        for file in self.session_path.joinpath('alf', 'FOV_00').glob('mpciMeanImage.*'):
            file.replace(file.with_name(file.name.replace('_estimate', '')))
        task.one.mode = 'local'
        with unittest.mock.patch.object(task.one.alyx, 'rest') as mock_rest:
            out = task.register_fov(meta, None)
            mock_rest.assert_not_called()
        self.assertEqual(1, len(out))
        self.assertEqual('FOV_00', out[0].get('name'))
        locations = out[0]['location']
        self.assertEqual(1, len(locations))
        self.assertEqual('L', locations[0].get('provenance', 'L'))

    def tearDown(self) -> None:
        """
        The ONE function is cached and therefore the One object persists beyond this test.
        Here we return the mode back to the default after testing behaviour in offline mode.
        """
        self.one.mode = 'auto'


class TestImagingMeta(unittest.TestCase):
    """Test raw imaging metadata versioning."""
    def test_patch_imaging_meta(self):
        """Test for ibllib.io.extractors.mesoscope.patch_imaging_meta function."""
        meta = {'version': '0.1.0', 'FOV': [{'roiUuid': None}, {'roiUUID': None}]}
        new_meta = mesoscope.patch_imaging_meta(meta)
        self.assertEqual(set(chain(*map(dict.keys, new_meta['FOV']))), {'roiUUID'})
        meta = {'FOV': [
            dict.fromkeys(['topLeftDeg', 'topRightDeg', 'bottomLeftDeg', 'bottomRightDeg']),
            dict.fromkeys(['topLeftMM', 'topRightMM', 'bottomLeftMM', 'bottomRightMM'])
        ]}
        new_meta = mesoscope.patch_imaging_meta(meta)
        self.assertIn('channelSaved', new_meta)
        self.assertCountEqual(new_meta['FOV'][0], ('Deg', 'MM'))
        expected = ('topLeft', 'topRight', 'bottomLeft', 'bottomRight')
        self.assertCountEqual(new_meta['FOV'][0]['MM'], expected)

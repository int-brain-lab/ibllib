import unittest

import numpy as np
import matplotlib.pyplot as plt

from ibllib.atlas import (BrainCoordinates, cart2sph, sph2cart, Trajectory,
                          Insertion, ALLEN_CCF_LANDMARKS_MLAPDV_UM, AllenAtlas)
from ibllib.atlas.regions import BrainRegions, FranklinPaxinosRegions
from ibllib.atlas.plots import prepare_lr_data, reorder_data
from iblutil.numerical import ismember


def _create_mock_atlas():
    """
    Instantiates a mock atlas.BrainAtlas for testing purposes mimicking Allen Atlas
    using the IBL Bregma and coordinate system
    """
    ba = AllenAtlas(res_um=25, mock=True)
    X, Y = np.meshgrid(ba.bc.xscale, ba.bc.yscale)
    top = X ** 2 + Y ** 2
    ba.top = (top - np.min(top)) / (np.max(top) - np.min(top)) * .001
    return ba


class TestBrainRegions(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.brs = BrainRegions()

    def test_rgba(self):
        assert self.brs.rgba.shape == (self.brs.rgb.shape[0], 4)

    def test_get(self):
        ctx = self.brs.get(688)
        self.assertTrue(len(ctx.acronym) == 1 and ctx.acronym == 'CTX')

    def test_ancestors_descendants(self):
        # here we use the same brain region as in the alyx test
        self.assertTrue(self.brs.descendants(ids=688).id.size == 567)
        self.assertTrue(self.brs.ancestors(ids=688).id.size == 4)
        # the leaves have no descendants but themselves
        leaves = self.brs.leaves()
        d = self.brs.descendants(ids=leaves['id'])
        self.assertTrue(np.all(np.sort(leaves['id']) == np.sort(d['id'])))

    def test_ancestors_descendants_indices(self):
        br = self.brs
        tpath = np.array([997, 8, 567, 688, 695, 315, 453, 12993])
        # /997/8/567/688/695/315/453/12993/
        # check ancestors
        ancs = br.ancestors(12993)
        assert np.all(ancs.id == tpath)
        # check ancestors with indices
        ancs, inds = br.ancestors(12993, return_indices=True)
        assert np.all(ancs.id == tpath)
        # check descendants with indices
        desdc, inds = br.descendants(12993, return_indices=True)
        self.assertEqual(inds, np.where(br.id == 12993))
        # check full subtree
        chemin = br.subtree(453)
        assert np.all(np.sort(chemin.id) == np.unique(np.r_[br.descendants(453).id, br.ancestors(453).id]))

    def test_mappings_lateralized(self):
        # the mapping assigns all non found regions to root (1:997), except for the void (0:0)
        # here we're looking at the retina (1327:304325711), so we expect 1327 at index 1327
        inds = self.brs._mapping_from_regions_list(np.array([304325711]), lateralize=True)
        inds_ = np.zeros_like(self.brs.id) + 1
        inds_[int((inds.size - 1) / 2)] = 1327
        inds_[-1] = 1327 * 2
        inds_[0] = 0
        assert np.all(inds == inds_)

    def test_mappings_not_lateralized(self):
        # if it's not lateralize, retina for both eyes should be in the
        inds = self.brs._mapping_from_regions_list(np.array([304325711]), lateralize=False)
        inds_ = np.zeros_like(self.brs.id) + 1
        inds_[int((inds.size - 1) / 2)] = 1327
        inds_[-1] = 1327
        inds_[0] = 0
        assert np.all(inds == inds_)

    def test_remap(self):
        # Test mapping atlas ids from one map to another
        atlas_id = np.array([463, 685])  # CA3 and PO
        cosmos_id = self.brs.remap(atlas_id, source_map='Allen', target_map='Cosmos')
        expected_cosmos_id = [1089, 549]  # HPF and TH
        assert np.all(cosmos_id == expected_cosmos_id)

    def test_id2id(self):
        # Test remapping of atlas id to atlas id
        atlas_id = np.array([463, 685])
        # Allen mapping, positive ids -> positive ids
        allen_id = self.brs.id2id(atlas_id, mapping='Allen')
        assert np.all(allen_id == atlas_id)
        # Allen mapping, negative ids -> positive ids
        allen_id = self.brs.id2id(-1 * atlas_id, mapping='Allen')
        assert np.all(allen_id == atlas_id)
        # Allen-lr mapping, positive ids -> positive ids
        allen_id = self.brs.id2id(atlas_id, mapping='Allen-lr')
        assert np.all(allen_id == atlas_id)
        # Allen-lr mapping, negative ids -> negative ids
        allen_id = self.brs.id2id(-1 * atlas_id, mapping='Allen-lr')
        assert np.all(allen_id == -1 * atlas_id)

        expected_cosmos_id = np.array([1089, 549])  # HPF and TH
        # Cosmos mapping, negative ids -> positive ids
        cosmos_id = self.brs.id2id(-1 * atlas_id, mapping='Cosmos')
        assert np.all(cosmos_id == expected_cosmos_id)
        # Cosmos-lr mapping, negative ids -> negative ids
        cosmos_id = self.brs.id2id(-1 * atlas_id, mapping='Cosmos-lr')
        assert np.all(cosmos_id == -1 * expected_cosmos_id)

    def test_id2acro(self):
        atlas_id = np.array([463, 685])  # CA3 and VM
        expected_allen_acronym = np.array(['CA3', 'VM'])
        # Allen mapping, positive ids,
        allen_acronym = self.brs.id2acronym(atlas_id, mapping='Allen')
        assert np.all(allen_acronym == expected_allen_acronym)
        # Allen-lr mapping, negative ids
        allen_acronym = self.brs.id2acronym(-1 * atlas_id, mapping='Allen-lr')
        assert np.all(allen_acronym == expected_allen_acronym)

        expected_cosmos_acronym = np.array(['HPF', 'TH'])
        cosmos_acronym = self.brs.id2acronym(atlas_id, mapping='Cosmos')
        assert np.all(cosmos_acronym == expected_cosmos_acronym)

    def test_id2index(self):
        atlas_id = np.array([463, 685])

        # Allen mapping, positive ids -> returns index on both side
        allen_id, index_both = self.brs.id2index(atlas_id, mapping='Allen')
        assert np.all(allen_id == atlas_id)
        for exp, ind in zip(atlas_id, index_both):
            assert np.all(ind == np.where(self.brs.id[self.brs.mappings['Allen']] == exp)[0])

        # Allen mapping, negative ids -> returns index on both side
        allen_id, index_both = self.brs.id2index(-1 * atlas_id, mapping='Allen')
        assert np.all(allen_id == atlas_id)
        for exp, ind in zip(atlas_id, index_both):
            assert np.all(ind == np.where(self.brs.id[self.brs.mappings['Allen']] == exp)[0])

        # Allen-lr mapping, positive ids ->  returns index on right side
        allen_id, index = self.brs.id2index(atlas_id, mapping='Allen-lr')
        assert np.all(allen_id == atlas_id)
        for i, (exp, ind) in enumerate(zip(atlas_id, index)):
            assert np.all(ind == index_both[i][index_both[i] <= self.brs.n_lr])

        # Allen-lr mapping, negative ids ->  returns index on left side
        allen_id, index = self.brs.id2index(-1 * atlas_id, mapping='Allen-lr')
        assert np.all(allen_id == -1 * atlas_id)
        for i, (exp, ind) in enumerate(zip(atlas_id, index)):
            assert np.all(ind == index_both[i][index_both[i] > self.brs.n_lr])

        # Cosmos mapping, positive ids ->  returns index on both sides
        expected_cosmos_id = [1089, 549]  # HPF and TH
        cosmos_id, index_both = self.brs.id2index(atlas_id, mapping='Cosmos')
        assert np.all(cosmos_id == expected_cosmos_id)
        for exp, ind in zip(expected_cosmos_id, index_both):
            assert np.all(ind == np.where(self.brs.id[self.brs.mappings['Cosmos']] == exp)[0])

    def test_acro2acro(self):
        acronym = np.array(['CA3', 'VM'])
        # Allen mapping
        allen_acronym = self.brs.acronym2acronym(acronym, mapping='Allen')
        assert np.all(acronym == allen_acronym)

        expected_cosmos_acronym = np.array(['HPF', 'TH'])
        # Cosmos mapping
        cosmos_acronym = self.brs.acronym2acronym(acronym, mapping='Cosmos-lr')
        assert np.all(cosmos_acronym == expected_cosmos_acronym)

    def test_acro2id(self):
        acronym = np.array(['CA3', 'VM'])
        expected_allen_id = np.array([463, 685])
        # Allen mapping, both hemisphere -> positive ids
        allen_id = self.brs.acronym2id(acronym, mapping='Allen', hemisphere=None)
        assert np.all(allen_id == expected_allen_id)
        # Allen mapping, left hemisphere -> positive ids
        allen_id = self.brs.acronym2id(acronym, mapping='Allen', hemisphere='left')
        assert np.all(allen_id == expected_allen_id)
        # Allen mapping, right hemisphere -> positive ids
        allen_id = self.brs.acronym2id(acronym, mapping='Allen', hemisphere='right')
        assert np.all(allen_id == expected_allen_id)

        # Allen-lr mapping, both hemisphere -> negative and positive ids
        allen_id = self.brs.acronym2id(acronym, mapping='Allen-lr', hemisphere=None)
        assert np.all(allen_id.ravel() == np.c_[-1 * expected_allen_id, expected_allen_id].ravel())
        # Allen-lr mapping, left hemisphere -> negative ids
        allen_id = self.brs.acronym2id(acronym, mapping='Allen-lr', hemisphere='left')
        assert np.all(allen_id == -1 * expected_allen_id)
        # Allen-lr mapping, right hemisphere -> positive ids
        allen_id = self.brs.acronym2id(acronym, mapping='Allen-lr', hemisphere='right')
        assert np.all(allen_id == expected_allen_id)

        expected_cosmos_id = np.array([1089, 549])
        # Cosmos-lr mapping, left hemisphere -> negative ids
        cosmos_id = self.brs.acronym2id(acronym, mapping='Cosmos-lr', hemisphere='left')
        assert np.all(cosmos_id == -1 * expected_cosmos_id)
        # Cosmos mapping, left hemisphere -> positive ids
        cosmos_id = self.brs.acronym2id(acronym, mapping='Cosmos', hemisphere='left')
        assert np.all(cosmos_id == expected_cosmos_id)

    def test_acro2index(self):
        acronym = np.array(['CA3', 'VM'])
        # Expect it to be same regardless of lateralised or non lateralised mapping
        for map, expected_acronym in zip(['Allen', 'Allen-lr', 'Cosmos', 'Cosmos-lr'],
                                         [np.array(['CA3', 'VM']), np.array(['CA3', 'VM']),
                                          np.array(['HPF', 'TH']), np.array(['HPF', 'TH'])]):
            # Mapping, both hemisphere, returns index on both sides
            map_acronym, index_both = self.brs.acronym2index(acronym, mapping=map, hemisphere=None)
            assert np.all(map_acronym == expected_acronym)
            for exp, ind in zip(expected_acronym, index_both):
                assert np.all(ind == np.where(self.brs.acronym[self.brs.mappings[map]] == exp)[0])

            # Mapping, left hemisphere, returns index that are > 1327
            map_acronym, index = self.brs.acronym2index(acronym, mapping=map, hemisphere='left')
            assert np.all(map_acronym == expected_acronym)
            for i, (exp, ind) in enumerate(zip(expected_acronym, index)):
                assert np.all(ind == index_both[i][index_both[i] > self.brs.n_lr])

            # Mapping, right hemisphere, returns index that are < 1327
            map_acronym, index = self.brs.acronym2index(acronym, mapping=map, hemisphere='right')
            assert np.all(map_acronym == expected_acronym)
            for i, (exp, ind) in enumerate(zip(expected_acronym, index)):
                assert np.all(ind == index_both[i][index_both[i] <= self.brs.n_lr])

    def test_index2id(self):
        index = np.array([468, 646, 1973])
        # Allen mapping
        allen_id = np.array([463, 685, 685])
        assert np.all(self.brs.index2id(index, mapping='Allen') == allen_id)
        # Allen-lr mapping
        allen_id = np.array([463, 685, -685])
        assert np.all(self.brs.index2id(index, mapping='Allen-lr') == allen_id)
        # Cosmos-lr mapping
        cosmos_id = np.array([1089, 549, -549])
        assert np.all(self.brs.index2id(index, mapping='Cosmos-lr') == cosmos_id)

    def test_index2acronym(self):
        index = np.array([468, 646, 1973])
        # Allen mapping
        allen_acronym = np.array(['CA3', 'VM', 'VM'])
        assert np.all(self.brs.index2acronym(index, mapping='Allen') == allen_acronym)
        # Allen-lr mapping
        allen_acronym = np.array(['CA3', 'VM', 'VM'])
        assert np.all(self.brs.index2acronym(index, mapping='Allen-lr') == allen_acronym)
        # Cosmos-lr mapping
        cosmos_acronym = np.array(['HPF', 'TH', 'TH'])
        assert np.all(self.brs.index2acronym(index, mapping='Cosmos-lr') == cosmos_acronym)

    def test_prepare_lr_data(self):
        acronyms_lh = np.array(['VPM', 'VPL', 'PO'])
        values_lh = np.array([0, 1, 2])
        acronyms_rh = np.array(['VPL', 'PO', 'CA1'])
        values_rh = np.array([3, 4, 5])
        acronyms, values = prepare_lr_data(acronyms_lh, values_lh, acronyms_rh, values_rh)

        assert np.array_equal(np.unique(np.r_[acronyms_lh, acronyms_rh]), acronyms)
        assert np.array_equal(values[acronyms == 'VPL'][0], np.array([1, 3]))
        np.testing.assert_equal(values[acronyms == 'VPM'][0], np.array([0, np.nan]))
        np.testing.assert_equal(values[acronyms == 'CA1'][0], np.array([np.nan, 5]))

    def test_reorder_data(self):
        acronyms = np.array(['AUDp1', 'AUDpo1', 'AUDv1', 'SSp-m1', 'SSp-n1'])
        values = np.array([0, 1, 2, 3, 4])
        _, idx = ismember(acronyms, self.brs.acronym)
        expected_acronyms = acronyms[np.argsort(self.brs.order[idx])]
        expected_values = values[np.argsort(self.brs.order[idx])]
        values = np.array([0, 1, 2, 3, 4])
        acronnyms_ordered, values_ordered = reorder_data(acronyms, values)
        assert np.array_equal(acronnyms_ordered, expected_acronyms)
        assert np.array_equal(values_ordered, expected_values)

    def test_argument_parser(self):
        acronyms = ['AUDp1', 'AUDpo1', 'AUDv1', 'SSp-m1', 'SSp-n1']
        ids = self.brs.acronym2id(acronyms)
        assert np.all(self.brs.parse_acronyms_argument(acronyms) == ids)
        assert np.all(self.brs.parse_acronyms_argument(np.array(acronyms)) == ids)
        assert np.all(self.brs.parse_acronyms_argument(ids) == ids)
        assert np.all(self.brs.parse_acronyms_argument(list(ids)) == ids)
        # makes sure it handles exception
        with self.assertRaises(AssertionError):
            self.brs.parse_acronyms_argument(acronyms + ['toto'])
        assert np.all(self.brs.parse_acronyms_argument(acronyms + ['toto'], mode='clip') == ids)

    def test_compute_hierarchy(self):
        self.brs.compute_hierarchy()
        np.testing.assert_equal(
            self.brs.hierarchy[:, 349], np.array([2, 3, 4, 5, 6, 340, 347, 349, 349, 349]))

    def test_propagate_down(self):
        acronyms = ['CB', 'ANcr2']
        ind0 = ismember(self.brs.acronym2id(acronyms), self.brs.id)[1]
        ind1 = [7, 45, 66]
        for ind in [ind0, ind1]:
            with self.subTest(indices=ind):
                acronyms = self.brs.acronym[ind]
                values = np.arange(len(acronyms))
                ibr, ivals = self.brs.propagate_down(acronyms, values)
                idesc = []
                for c, i in enumerate(ind):
                    ii = ismember(self.brs.descendants(self.brs.id[i])['id'], self.brs.id)[1]
                    idesc.append(ii)
                np.testing.assert_equal(ibr, np.unique(np.concatenate(idesc)))


class TestAtlas(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ba = _create_mock_atlas()

    def test_compute_regions_volume(self):
        self.ba.compute_regions_volume()
        self.assertTrue(self.ba.regions.volume.shape == self.ba.regions.acronym.shape)


class TestAtlasPlots(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ba = _create_mock_atlas()

    def test_coronal_slice(self):
        ax = self.ba.plot_cslice(ap_coordinate=0)
        im = ax.get_images()[0]
        assert im.get_array().shape == (self.ba.bc.nz, self.ba.bc.nx)
        ax.clear()
        ax = self.ba.plot_cslice(ap_coordinate=0, volume='annotation')
        im = ax.get_images()[0]
        assert im.get_array().shape == (self.ba.bc.nz, self.ba.bc.nx, 3)
        ax.clear()

    def test_sagittal_slice(self):
        ax = self.ba.plot_sslice(ml_coordinate=0)
        im = ax.get_images()[0]
        assert im.get_array().shape == (self.ba.bc.nz, self.ba.bc.ny)
        ax.clear()
        ax = self.ba.plot_sslice(ml_coordinate=0, volume='annotation')
        im = ax.get_images()[0]
        assert im.get_array().shape == (self.ba.bc.nz, self.ba.bc.ny, 3)
        ax.clear()

    def test_horizontal_slice(self):
        ax = self.ba.plot_hslice(dv_coordinate=-0.002)
        im = ax.get_images()[0]
        assert im.get_array().shape == (self.ba.bc.ny, self.ba.bc.nx)
        ax.clear()
        ax = self.ba.plot_hslice(dv_coordinate=-0.002, volume='annotation')
        im = ax.get_images()[0]
        assert im.get_array().shape == (self.ba.bc.ny, self.ba.bc.nx, 3)
        ax.clear()

    def tearDown(self) -> None:
        plt.close('all')


class TestAtlasSlicesConversion(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.ba = _create_mock_atlas()

    def test_allen_ba(self):
        self.assertTrue(np.allclose(self.ba.bc.xyz2i(np.array([0, 0, 0]), round=False),
                                    ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / 25))

    def test_lookups(self):
        # the get_labels lookup returns the regions ids (not the indices !!)
        assert self.ba.get_labels([0, 0, self.ba.bc.i2z(103)]) == 304325711
        # the beryl mapping doesn't include the retina so it returns root
        assert self.ba.get_labels([0, 0, self.ba.bc.i2z(103)], mapping='Beryl') == 997  # root
        # unlike the retina, root stays root whatever the mapping
        assert self.ba.get_labels([0, 0, 0]) == 0  # void !
        assert self.ba.get_labels([0, 0, 0], mapping='Beryl') == 0  # root
        # Check the cosmos mapping too
        assert self.ba.get_labels([0, 0, self.ba.bc.i2z(103)], mapping='Cosmos') == 997
        assert self.ba.get_labels([0, 0, 0], mapping='Cosmos') == 0

    def test_lookups_probabilistic(self):
        xyz = np.array([0, -.0058, -.0038])
        aid = self.ba.get_labels(xyz, mapping='Beryl')
        aids, proportions = self.ba.get_labels(xyz, mapping='Beryl', radius_um=250)
        self.assertEqual(aid, 0)
        assert (np.all(aids == np.array([0])))
        assert (np.isclose(proportions, np.array([1.])))

    def test_plot_slices(self):
        axs = self.ba.plot_slices(np.array([0, -.0024, -.0038]))
        assert axs.shape == (2, 2)

    def test_slice(self):
        ba = self.ba
        nx, ny, nz = ba.bc.nxyz
        # tests output shapes
        self.assertTrue(ba.slice(axis=0, coordinate=0).shape == (ny, nz))  # sagittal
        self.assertTrue(ba.slice(axis=1, coordinate=0).shape == (nx, nz))  # coronal
        self.assertTrue(ba.slice(axis=2, coordinate=-.002).shape == (ny, nx))  # horizontal
        # tests out of bound
        with self.assertRaises(ValueError):
            ba.slice(axis=1, coordinate=123)
        self.assertTrue(ba.slice(axis=1, coordinate=21, mode='clip').shape == (nx, nz))
        """
        here we test the different volumes and mappings
        """
        # the label volume contains the region index (not id!),
        iregions = ba.slice(axis=0, coordinate=0, volume=ba.label)
        assert np.all(np.unique(iregions) == np.array([0, 1327]))
        rgb_slice = ba.slice(axis=0, coordinate=0, volume='annotation')
        assert rgb_slice.shape == (ny, nz, 3)
        # check that without remapping (default full remap) the retina gets returned
        assert np.all(np.unique(rgb_slice) == np.unique(np.r_[ba.regions.rgb[1327], 0]))
        # now with a remap the retina should not be there anymore and there should be only white
        rgb_slice = ba.slice(axis=0, coordinate=0, volume='annotation', mapping='Beryl')
        assert np.all(np.unique(rgb_slice) == np.array([0, 255]))
        assert ba.slice(axis=0, coordinate=0, volume='surface').shape == (ny, nz)

    def test_ccf_xyz(self):
        # test with bregma first
        assert np.all(np.abs(ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] -
                             self.ba.xyz2ccf(np.array([[0, 0, 0]]))) < 12)
        # check it works with a single coordinate
        coords = (np.array([0, 0, 0]),  # tests a single coordinate
                  np.array([[0, 0, 0], [-2000., 500, 200]]) / 1.e6)
        for xyz in coords:
            assert np.all(np.isclose(self.ba.ccf2xyz(self.ba.xyz2ccf(xyz)), xyz))
        # test with the vertices from a mesh -
        vertices = np.array([[7896.46, 3385.61, 514.179],  # this is apdvml
                             [7777.87, 3416.28, 512.176],
                             [7902.99, 3481.55, 503.324]])
        vxyz = self.ba.ccf2xyz(vertices, ccf_order='mlapdv')
        self.assertTrue(np.all(np.isclose(self.ba.xyz2ccf(vxyz, ccf_order='mlapdv'), vertices)))
        vxyz = self.ba.ccf2xyz(vertices, ccf_order='apdvml')
        self.assertTrue(np.all(np.isclose(self.ba.xyz2ccf(vxyz, ccf_order='apdvml'), vertices)))

        # check if we have the ccf origin we get extremes of bregma atlas
        ccf_apdvml = np.array([0, 0, 0])
        xyz_mlapdv = self.ba.ccf2xyz(ccf_apdvml, ccf_order='apdvml')
        assert np.all(np.isclose(xyz_mlapdv, np.array([self.ba.bc.xlim[0], self.ba.bc.ylim[0],
                                                       self.ba.bc.zlim[0]])))
        # check that if we move in one direction in ccf the correct dimension in xyz change
        # Move in ML
        ccf_apdvml = np.array([0, 0, 1000])
        xyz_mlapdv = self.ba.ccf2xyz(ccf_apdvml, ccf_order='apdvml')
        assert np.all(np.isclose(xyz_mlapdv[1:], np.array([self.ba.bc.ylim[0],
                                                           self.ba.bc.zlim[0]])))
        self.assertFalse(xyz_mlapdv[0] == self.ba.bc.xlim[0])
        self.assertTrue(self.ba.bc.xlim[0] < xyz_mlapdv[0] < self.ba.bc.xlim[1])
        assert np.all(np.isclose(self.ba.xyz2ccf(xyz_mlapdv, 'apdvml'), ccf_apdvml))

        # Move in DV
        ccf_apdvml = np.array([0, 1000, 0])
        xyz_mlapdv = self.ba.ccf2xyz(ccf_apdvml, ccf_order='apdvml')
        assert np.all(np.isclose(xyz_mlapdv[0:2], np.array([self.ba.bc.xlim[0],
                                                            self.ba.bc.ylim[0]])))
        self.assertFalse(xyz_mlapdv[2] == self.ba.bc.zlim[0])
        self.assertTrue(self.ba.bc.zlim[0] > xyz_mlapdv[2] > self.ba.bc.zlim[1])
        assert np.all(np.isclose(self.ba.xyz2ccf(xyz_mlapdv, 'apdvml'), ccf_apdvml))

        # Move in AP
        ccf_apdvml = np.array([1000, 0, 0])
        xyz_mlapdv = self.ba.ccf2xyz(ccf_apdvml, ccf_order='apdvml')
        assert np.all(np.isclose(xyz_mlapdv[[0, 2]], np.array([self.ba.bc.xlim[0],
                                                               self.ba.bc.zlim[0]])))
        self.assertFalse(xyz_mlapdv[1] == self.ba.bc.ylim[0])
        self.assertTrue(self.ba.bc.ylim[0] > xyz_mlapdv[1] > self.ba.bc.ylim[1])
        assert np.all(np.isclose(self.ba.xyz2ccf(xyz_mlapdv, 'apdvml'), ccf_apdvml))


class TestInsertion(unittest.TestCase):

    def test_init_from_dict(self):
        d = {
            'label': 'probe00',
            'x': 544.0,
            'y': 1285.0,
            'z': 0.0,
            'phi': 0.0,
            'theta': 5.0,
            'depth': 4501.0,
            'beta': 0.0}
        ins = Insertion.from_dict(d)
        # eval the entry point, should be super close
        dxyz = ins.trajectory.eval_x(d['x'] / 1e6) - np.array((d['x'], d['y'], d['z'])) / 1e6
        self.assertTrue(np.all(np.isclose(dxyz, 0)))
        # test methods tip/entry/xyz
        dd = np.sum(np.sqrt(np.diff(ins.xyz, axis=0) ** 2)) - d['depth'] / 1e6
        self.assertLess(abs(dd), 0.01)

    def test_init_from_track(self):
        brain_atlas = _create_mock_atlas()
        xyz_track = np.array([[0.003139, -0.00405, -0.000793],
                              [0.003089, -0.00405, -0.001043],
                              [0.003014, -0.004025, -0.001268],
                              [0.003014, -0.00405, -0.001393],
                              [0.002939, -0.00405, -0.001643],
                              [0.002914, -0.004025, -0.001918],
                              [0.002989, -0.0041, -0.002168],
                              [0.002914, -0.004075, -0.002318],
                              [0.002939, -0.0041, -0.002368],
                              [0.002914, -0.0041, -0.002443],
                              [0.002839, -0.0041, -0.002743],
                              [0.002764, -0.0041, -0.003068],
                              [0.002589, -0.004125, -0.003768],
                              [0.002489, -0.004275, -0.004893],
                              [0.002439, -0.004375, -0.005093],
                              [0.002364, -0.0044, -0.005418]])
        insertion = Insertion.from_track(xyz_track, brain_atlas)
        self.assertTrue(abs(insertion.theta - 10.58704241) < 1e6)
        # Test that the entry and exit intersection are computed properly
        brain_entry = insertion.get_brain_entry(insertion.trajectory, brain_atlas)
        self.assertTrue(brain_entry[2] == brain_atlas.bc.i2z(100))
        brain_exit = insertion.get_brain_exit(insertion.trajectory, brain_atlas)
        self.assertTrue(brain_exit[2] == brain_atlas.bc.i2z(104))


class TestTrajectory(unittest.TestCase):

    def test_project_mindist(self):
        # test min dist
        traj = Trajectory.fit(np.array([[0.3, 0.3, 0.4], [0, 0, 1]]))
        min_dist = np.sqrt(np.sum(traj.project(np.array([0, 0, 0])) ** 2))
        assert np.isclose(min_dist, traj.mindist(np.array([0, 0, 0])))

        # test projection, single point and vectorized
        point = np.array([0.06656238, 0.47127062, 0.17440139])
        expected = [0.36483837, 0.36483837, 0.27032326]
        assert np.all(np.isclose(traj.project(point), expected))
        assert np.all(np.isclose(traj.project(np.tile(point, (2, 1))), np.tile(expected, (2, 1))))

    def test_eval_trajectory(self):
        line = Trajectory.fit(np.array([[0.3, 0.3, 0.4], [0, 0, 1]]))
        # test integer
        self.assertTrue(np.all(np.isclose(line.eval_y(0), np.array([0, 0, 1]))))
        # test float
        self.assertTrue(np.all(np.isclose(line.eval_y(0.0), np.array([0, 0, 1]))))
        # test list
        self.assertTrue(np.all(np.isclose(line.eval_y([0.0, 0.0]), np.array([0, 0, 1]))))
        # test array
        arr = np.array([0.0, 0.0])[..., np.newaxis]
        self.assertTrue(np.all(np.isclose(line.eval_y(arr), np.array([0, 0, 1]))))
        # test void direction
        vertical = Trajectory.fit(np.array([[0, 0, 0], [0, 0, 1]]))
        self.assertTrue(np.all(np.isnan(vertical.eval_x(5))))

    def test_trajectory(self):
        np.random.seed(42)
        xyz = np.zeros([120, 3])
        xyz[:, 0] = np.linspace(1, 9, 120)
        xyz[:, 1] = np.linspace(2, 4, 120)
        xyz[:, 2] = np.linspace(-2, 3, 120)
        xyz += np.random.normal(size=xyz.shape) * 0.4
        traj = Trajectory.fit(xyz)
        # import matplotlib.pyplot as plt
        # import mpl_toolkits.mplot3d as m3d
        # ax = m3d.Axes3D(plt.figure())
        # ax.scatter3D(*xyz.T)
        # ax.plot3D(*insertion.eval_x(np.array([0, 10])).T)
        # ax.plot3D(*insertion.eval_y(xyz[:, 1]).T, 'r')
        d = xyz[:, 0] - traj.eval_y(xyz[:, 1])[:, 0]
        self.assertTrue(np.abs(np.mean(d)) < 0.001)
        d = xyz[:, 0] - traj.eval_y(xyz[:, 1])[:, 0]
        self.assertTrue(np.abs(np.mean(d)) < 0.001)
        d = xyz[:, 1] - traj.eval_z(xyz[:, 2])[:, 1]
        self.assertTrue(np.abs(np.mean(d)) < 0.001)

    def test_exit_volume(self):
        bc = BrainCoordinates((11, 13, 15), xyz0=(-5, -6, -7))
        # test arbitrary line
        line = Trajectory.fit(np.array([[0.1, 0.1, 0], [0, 0, 1]]))
        epoints = Trajectory.exit_points(line, bc)
        self.assertTrue(np.all(np.isclose(epoints, np.array([[0.8, 0.8, -7.], [-0.6, -0.6, 7.]]))))
        # test apline
        hline = Trajectory.fit(np.array([[0, 0, 0], [0, 1, 0]]))
        epoints = Trajectory.exit_points(hline, bc)
        self.assertTrue(np.all(np.isclose(epoints, np.array([[0, -6, 0], [0, 6, 0]]))))
        # test mlline
        hline = Trajectory.fit(np.array([[0, 0, 0], [1, 0, 0]]))
        epoints = Trajectory.exit_points(hline, bc)
        self.assertTrue(np.all(np.isclose(epoints, np.array([[-5, 0, 0], [5, 0, 0]]))))
        # test vertical line
        vline = Trajectory.fit(np.array([[0, 0, 0], [0, 0, 1]]))
        epoints = Trajectory.exit_points(vline, bc)
        self.assertTrue(np.all(np.isclose(epoints, np.array([[0, 0, -7.], [0, 0, 7.]]))))


class TestsCoordinatesSimples(unittest.TestCase):

    def test_brain_coordinates(self):
        vshape = (6, 7, 8)
        bc = BrainCoordinates(vshape)
        self.assertTrue(bc.i2x(0) == 0)
        self.assertTrue(bc.i2x(6) == 6)
        self.assertTrue(bc.nx == 6)
        self.assertTrue(bc.ny == 7)
        self.assertTrue(bc.nz == 8)
        # test array functions
        in_out = [([3, 4, 5], np.array([3, 4, 5])),
                  (np.array([3, 4, 5]), np.array([3, 4, 5])),
                  (np.array([[3, 4, 5], [3, 4, 5]]), np.array([[3, 4, 5], [3, 4, 5]])),
                  ]
        for io in in_out:
            self.assertTrue(np.all(bc.xyz2i(io[0]) == io[1]))
            self.assertTrue(np.all(bc.i2xyz(io[1]) == io[0]))

    def test_reverse_directions(self):
        bc = BrainCoordinates(nxyz=(6, 7, 8), xyz0=[50, 60, 70], dxyz=[-10, -10, -10])
        self.assertTrue(bc.i2x(0) == 50 and bc.i2x(bc.nx - 1) == 0)
        self.assertTrue(bc.i2y(0) == 60 and bc.i2y(bc.ny - 1) == 0)
        self.assertTrue(np.all(bc.i2z(np.array([0, 1])) == np.array([70, 60])))
        bc = BrainCoordinates(nxyz=(6, 7, 8), xyz0=[50, 60, 70], dxyz=-10)
        self.assertTrue(bc.dx == bc.dy == bc.dz == -10)

    def test_sph2cart_and_back(self):
        dv = np.array([0, -1, 1, 0, 0, 0, 0, 0, 0])  # z
        ml = np.array([0, 0, 0, 0, -1, 1, 0, 0, 0])  # x
        ap = np.array([0, 0, 0, 0, 0, 0, 0, -1, 1])  # y

        phi = np.array([0., 0., 0., 0., 180., 0., 0., -90., 90.])
        theta = np.array([0., 180., 0., 0., 90., 90., 0., 90., 90.])
        r = np.array([0., 1, 1, 0., 1, 1, 0., 1, 1])

        r_, t_, p_ = cart2sph(ml, ap, dv)
        assert np.all(np.isclose(r, r_))
        assert np.all(np.isclose(phi, p_))
        assert np.all(np.isclose(theta, t_))

        x_, y_, z_ = sph2cart(r, theta, phi)
        assert np.all(np.isclose(ml, x_))
        assert np.all(np.isclose(ap, y_))
        assert np.all(np.isclose(dv, z_))


class TestFranklinPaxinos(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.frs = FranklinPaxinosRegions()
        self.brs = BrainRegions()

    def test_level(self):

        # This is a region that is only in FranklinPaxinos Atlas
        info_fr5 = self.frs.get(2325)
        # This is a region that is in both Allen and Franklin that should share parent with above region
        info_fr23 = self.frs.get(667)
        np.testing.assert_equal(info_fr5['level'], info_fr23['level'])

        # These are the levels in the franklin paxinos
        info_V1M6b = self.frs.get(2433)  # only in franklin
        info_V1M = self.frs.get(2098)  # only in franklin
        info_V1 = self.frs.get(385)  # in frankin and in Allen

        # equivalent of V1 in Allen
        info_VISp = self.brs.get(self.brs.acronym2id('VISp'))

        np.testing.assert_equal(info_V1['level'], info_VISp['level'])
        np.testing.assert_equal(info_V1M['level'], info_V1['level'] + 1)
        np.testing.assert_equal(info_V1M6b['level'], info_V1['level'] + 2)

    def test_conversions(self):

        # These are regions just in FP atlas
        np.testing.assert_equal(self.frs.acronym2id('FrA-5'), 2325)
        np.testing.assert_equal(self.frs.acronym2index('FrA-5')[1][0][0], 10)
        np.testing.assert_equal(self.frs.acronym2index('FrA-5')[1][0][1], 1440)
        np.testing.assert_equal(self.frs.acronym2id('V1M-6b'), 2433)
        np.testing.assert_equal(self.frs.acronym2index('V1M-6b')[1][0][0], 209)
        np.testing.assert_equal(self.frs.acronym2index('V1M-6b')[1][0][1], 1639)

        # These are regions in both and so have taken on Allen acronyms
        np.testing.assert_equal(self.frs.acronym2id('ORBl2/3'), 412)
        np.testing.assert_equal(self.brs.acronym2id('ORBl2/3'), 412)
        np.testing.assert_equal(self.frs.acronym2id('LO-2/3'), np.array([]))


if __name__ == "__main__":
    unittest.main(exit=False, verbosity=2)

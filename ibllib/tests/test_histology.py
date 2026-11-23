from pathlib import Path
import tempfile
import unittest

import numpy as np

from ibllib.pipes import histology
from ibllib.pipes.ephys_alignment import EphysAlignment, TIP_SIZE_UM, _cumulative_distance
import iblatlas.atlas as atlas

# TODO Place this in setUpModule()
brain_atlas = atlas.AllenAtlas(res_um=25)


class TestHistology(unittest.TestCase):
    def setUp(self) -> None:
        self.path_tracks = Path(__file__).parent.joinpath('fixtures', 'histology', 'tracks')

    def test_histology_get_brain_regions(self):
        # first part of the test is to check on an actual track file
        for file_track in self.path_tracks.rglob('*_pts.csv'):
            xyz = histology.load_track_csv(file_track, brain_atlas=brain_atlas)
            channels, ins = histology.get_brain_regions(xyz=xyz, brain_atlas=brain_atlas)
        # also check that it works from an insertion
        channels, ins2 = histology.get_brain_regions(xyz=ins.xyz, brain_atlas=brain_atlas)
        self.assertTrue(channels.acronym[-1] == 'VISpm1')
        self.assertTrue(channels.acronym[0] == 'APN')
        a = np.array([ins.x, ins.y, ins.z, ins.phi, ins.theta, ins.depth])
        b = np.array([ins2.x, ins2.y, ins2.z, ins2.phi, ins2.theta, ins2.depth])
        self.assertTrue(np.all(np.isclose(a, b)))

    def test_histology_insertion_from_track(self):

        for file_track in self.path_tracks.rglob('*_pts.csv'):
            xyz = histology.load_track_csv(file_track, brain_atlas=brain_atlas)
            insertion = atlas.Insertion.from_track(xyz, brain_atlas=brain_atlas)
            # checks that the tip coordinate is not the deepest point but its projection
            self.assertFalse(np.all(np.isclose(insertion.tip, xyz[-1])))

    def test_get_brain_exit_entry(self):
        traj = atlas.Trajectory.fit(np.array([[0, 0, 0], [0, 0, 0.005]]))
        top = atlas.Insertion.get_brain_entry(traj, brain_atlas=brain_atlas)
        bottom = atlas.Insertion.get_brain_exit(traj, brain_atlas=brain_atlas)
        ix, iy = (brain_atlas.bc.x2i(0), brain_atlas.bc.y2i(0))
        self.assertTrue(np.isclose(brain_atlas.top[iy, ix], top[2]))
        self.assertTrue(np.isclose(brain_atlas.bottom[iy, ix], bottom[2]))

    def test_filename_parser(self):
        tdata = [
            {
                'input': Path('/gna/electrode_tracks_SWC_014/2019-12-12_SWC_014_001_probe01_fit.csv'),
                'output': {'date': '2019-12-12', 'experiment_number': 1, 'name': 'probe01', 'subject': 'SWC_014'},
            },
            {
                'input': Path('/gna/datadisk/Data/Histology/tracks/ZM_2407/2019-11-06_ZM_2407_001_probe_00_pts.csv'),
                'output': {'date': '2019-11-06', 'experiment_number': 1, 'name': 'probe_00', 'subject': 'ZM_2407'},
            },
            {
                'input': Path('/gna/2019-12-06_KS023_001_probe01_pts.csv'),
                'output': {'date': '2019-12-06', 'experiment_number': 1, 'name': 'probe01', 'subject': 'KS023'},
            },
        ]
        for t in tdata:
            track_file = t['input']
            assert t['output'] == histology._parse_filename(track_file)


class TestLoadTrackCsvTransformDir(unittest.TestCase):
    """load_track_csv with a brainregister transform folder; the mapping itself is tested in ibl-brain-transforms."""

    # sample2ARA fixture points computed with transformix, (ML, DV, AP) voxels
    ARA_MLDVAP = np.array([
        [327.55, 243.48, 467.88],
        [355.52, 253.58, 188.88],
        [190.58, 104.34, 308.58],
        [378.58, 122.71, 288.37],
        [153.7, 80.75, 140.03],
        [70.28, 267.83, 225.18],
        [341.94, 203.52, 138.69],
        [111.41, 234.88, 168.53],
    ])
    SAMPLE_MLDVAP = np.array([
        [315.766123, 233.730672, 496.998882],
        [338.145779, 232.031787, 212.184286],
        [173.618186, 109.080041, 343.207508],
        [355.353786, 115.098122, 317.873022],
        [133.466978, 84.022924, 172.263323],
        [63.289704, 264.40041, 252.061039],
        [322.253399, 186.614419, 163.29388],
        [102.149702, 229.307379, 196.526479],
    ])

    def setUp(self) -> None:
        self.transform_dir = Path(__file__).parent.joinpath('fixtures', 'histology', 'sample2ARA')

    def test_load_track_csv_transform_dir(self):
        with tempfile.TemporaryDirectory() as td:
            file_ds, file_ara = Path(td, 'ds_pts.csv'), Path(td, 'ara_pts.csv')
            # lasagna files are Z,X,Y = (AP, ML, DV)
            np.savetxt(file_ds, self.SAMPLE_MLDVAP[:, [2, 0, 1]], delimiter=',', fmt='%.6f')
            np.savetxt(file_ara, self.ARA_MLDVAP[:, [2, 0, 1]], delimiter=',', fmt='%.6f')
            # a track traced in sample space loads to the same IBL coordinates as the same track in ARA space
            xyz_ds = histology.load_track_csv(file_ds, brain_atlas=brain_atlas, transform_dir=self.transform_dir)
            xyz_ara = histology.load_track_csv(file_ara, brain_atlas=brain_atlas)
            np.testing.assert_allclose(xyz_ds, xyz_ara, rtol=0, atol=1e-9)
            self.assertTrue(np.all(np.diff(xyz_ds[:, 2]) <= 0))  # sorted along DV, deepest last
            # points outside the ARA volume are kept, with a warning
            np.savetxt(file_ara, [[-5.0, 100.0, 100.0], [200.0, 100.0, 100.0]], delimiter=',')
            with self.assertLogs('ibllib.pipes.histology', level='WARNING'):
                xyz = histology.load_track_csv(file_ara, brain_atlas=brain_atlas)
            self.assertEqual(xyz.shape, (2, 3))
            # only the 25 um atlas is accepted
            with self.assertRaises(ValueError):
                histology.load_track_csv(file_ds, brain_atlas=atlas.AllenAtlas(res_um=50), transform_dir=self.transform_dir)
            # empty file
            Path(td, 'empty_pts.csv').touch()
            xyz = histology.load_track_csv(Path(td, 'empty_pts.csv'), brain_atlas=brain_atlas, transform_dir=self.transform_dir)
            self.assertEqual(xyz.size, 0)


# Load in data for ephys alignment testing
data = np.load(Path(Path(__file__).parent.joinpath('fixtures', 'ephysalignment'), 'alignment_data.npz'), allow_pickle=True)
xyz_picks = data['xyz_picks']
feature_prev = data['feature_prev']
track_prev = data['track_prev']
xyz_channels_ref = data['xyz_channels_ref']
brain_regions_ref = data['brain_regions_ref']
depths = np.arange(20, 3860, 20) / 1e6


class TestsEphysAlignment(unittest.TestCase):
    def setUp(self):
        self.ephysalign = EphysAlignment(xyz_picks, brain_atlas=brain_atlas)
        self.feature = self.ephysalign.feature_init
        self.track = self.ephysalign.track_init

    def test_no_scaling(self):
        xyz_channels = self.ephysalign.get_channel_locations(self.feature, self.track, depths=depths)
        coords = np.r_[[xyz_picks[-1, :]], [xyz_channels[0, :]]]
        dist_to_fist_electrode = np.around(_cumulative_distance(coords)[-1], 5)
        assert np.isclose(dist_to_fist_electrode, (TIP_SIZE_UM + 20) / 1e6)

    def test_offset(self):
        feature_val = 500 / 1e6
        track_val = 1000 / 1e6

        tracks = np.sort(np.r_[self.track[[0, -1]], track_val])
        track_new = self.ephysalign.feature2track(tracks, self.feature, self.track)
        feature_new = np.sort(np.r_[self.feature[[0, -1]], feature_val])
        track_new = self.ephysalign.adjust_extremes_uniform(feature_new, track_new)

        xyz_channels = self.ephysalign.get_channel_locations(feature_new, track_new, depths=depths)
        coords = np.r_[[xyz_picks[-1, :]], [xyz_channels[0, :]]]
        dist_to_fist_electrode = np.around(_cumulative_distance(coords)[-1], 5)
        assert np.isclose(dist_to_fist_electrode, ((TIP_SIZE_UM + 20) / 1e6 + feature_val))
        track_val = self.ephysalign.track2feature(track_val, feature_new, track_new)
        self.assertTrue(np.all(np.isclose(track_val, feature_val)))

        region_new, _ = self.ephysalign.scale_histology_regions(feature_new, track_new)
        _, scale_factor = self.ephysalign.get_scale_factor(region_new)
        self.assertTrue(np.all(np.isclose(scale_factor, 1)))

    def test_uniform_scaling(self):
        feature_val = np.array([500, 700, 2000]) / 1e6
        track_val = np.array([1000, 1300, 2700]) / 1e6

        tracks = np.sort(np.r_[self.track[[0, -1]], track_val])
        track_new = self.ephysalign.feature2track(tracks, self.feature, self.track)
        feature_new = np.sort(np.r_[self.feature[[0, -1]], feature_val])
        track_new = self.ephysalign.adjust_extremes_uniform(feature_new, track_new)

        region_new, _ = self.ephysalign.scale_histology_regions(feature_new, track_new)
        _, scale_factor = self.ephysalign.get_scale_factor(region_new)
        self.assertTrue(np.isclose(scale_factor[0], 1))
        self.assertTrue(np.isclose(scale_factor[-1], 1))

    def test_linear_scaling(self):
        feature_val = np.array([500, 700, 2000]) / 1e6
        track_val = np.array([1000, 1300, 2700]) / 1e6

        tracks = np.sort(np.r_[self.track[[0, -1]], track_val])
        track_new = self.ephysalign.feature2track(tracks, self.feature, self.track)
        feature_new = np.sort(np.r_[self.feature[[0, -1]], feature_val])

        fit = np.polyfit(feature_new[1:-1], track_new[1:-1], 1)
        linear_fit = np.around(1 / fit[0], 3)

        feature_new, track_new = self.ephysalign.adjust_extremes_linear(feature_new, track_new, extend_feature=1)

        region_new, _ = self.ephysalign.scale_histology_regions(feature_new, track_new)
        _, scale_factor = self.ephysalign.get_scale_factor(region_new)

        self.assertTrue(np.isclose(np.around(scale_factor[0], 3), linear_fit))
        self.assertTrue(np.isclose(np.around(scale_factor[-1], 3), linear_fit))


class TestsEphysReconstruction(unittest.TestCase):
    def setUp(self):
        self.ephysalign = EphysAlignment(xyz_picks, track_prev=track_prev, feature_prev=feature_prev, brain_atlas=brain_atlas)
        self.feature = self.ephysalign.feature_init
        self.track = self.ephysalign.track_init

    def test_channel_locations(self):
        xyz_channels = self.ephysalign.get_channel_locations(self.feature, self.track, depths=depths)
        self.assertTrue(np.all(np.isclose(xyz_channels[0, :], xyz_channels_ref[0])))
        self.assertTrue(np.all(np.isclose(xyz_channels[-1, :], xyz_channels_ref[-1])))
        brain_regions = self.ephysalign.get_brain_locations(xyz_channels)
        self.assertTrue(np.all(np.equal(np.unique(brain_regions.acronym), brain_regions_ref)))


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

from pathlib import Path
import shutil
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


class TestSample2AraTransform(unittest.TestCase):
    """
    Synthetic brainregister sample2ARA parameter files (affine + bspline, fixed = 25 um ARA grid).
    Expected values computed with transformix (SimpleITK-elastix), including points at / beyond the
    edges of the bspline grid where the displacement is zero.
    """

    ARA_MLDVAP = np.array([
        [327.55, 243.48, 467.88],
        [355.52, 253.58, 188.88],
        [190.58, 104.34, 308.58],
        [378.58, 122.71, 288.37],
        [153.7, 80.75, 140.03],
        [70.28, 267.83, 225.18],
        [341.94, 203.52, 138.69],
        [111.41, 234.88, 168.53],
        [0.0, 0.0, 0.0],
        [455.0, 319.0, 527.0],
        [-5.5, 160.0, 264.0],
        [228.0, 330.25, 10.0],
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
        [-22.17, 16.5365, 35.666],
        [444.63, 295.8635, 552.434],
        [-14.745, 170.6505, 297.332],
        [217.49005, 304.116452, 27.711646],
    ])

    def setUp(self) -> None:
        self.transform_dir = Path(__file__).parent.joinpath('fixtures', 'histology', 'sample2ARA')
        self.ara_apmldv = self.ARA_MLDVAP[:, [2, 0, 1]]
        self.sample_apmldv = self.SAMPLE_MLDVAP[:, [2, 0, 1]]

    def test_ara_to_sample(self):
        with self.assertLogs('ibllib.pipes.histology', level='WARNING'):  # some points are outside the ARA volume
            apmldv = histology.lasagna_ara_to_sample(self.ara_apmldv, self.transform_dir)
        np.testing.assert_allclose(apmldv, self.sample_apmldv, atol=1e-5)

    def test_sample_to_ara(self):
        apmldv = histology.lasagna_sample_to_ara(self.sample_apmldv[:10], self.transform_dir)
        np.testing.assert_allclose(apmldv, self.ara_apmldv[:10], atol=1e-5)

    def test_ibl_round_trip(self):
        xyz = histology.lasagna_to_ibl(self.ara_apmldv[:8], brain_atlas=brain_atlas)
        np.testing.assert_allclose(histology.ibl_to_lasagna(xyz, brain_atlas=brain_atlas), self.ara_apmldv[:8], atol=1e-9)
        mldvap = histology.ibl_ara_to_sample(xyz, self.transform_dir, brain_atlas=brain_atlas)
        np.testing.assert_allclose(mldvap, self.SAMPLE_MLDVAP[:8], atol=1e-5)
        xyz_back = histology.ibl_sample_to_ara(mldvap, self.transform_dir, brain_atlas=brain_atlas)
        np.testing.assert_allclose(xyz_back, xyz, atol=1e-10)

    def test_transform_lasagna_csv(self):
        with tempfile.TemporaryDirectory() as td:
            file_in, file_out = Path(td, 'ds_pts.csv'), Path(td, 'ara_pts.csv')
            np.savetxt(file_in, self.sample_apmldv[:8], delimiter=',')
            apmldv = histology.transform_lasagna_csv(file_in, file_out, self.transform_dir)
            np.testing.assert_allclose(apmldv, self.ara_apmldv[:8], atol=1e-5)
            np.testing.assert_allclose(np.loadtxt(file_out, delimiter=','), self.ara_apmldv[:8], atol=1e-3)
            # the output file is read as ARA points by load_track_csv, consistent with lasagna_to_ibl
            xyz = histology.load_track_csv(file_out, brain_atlas=brain_atlas)
            xyz_ref = histology.lasagna_to_ibl(np.loadtxt(file_out, delimiter=','), brain_atlas=brain_atlas)
            np.testing.assert_allclose(np.sort(xyz, axis=0), np.sort(xyz_ref, axis=0))
            with self.assertRaises(ValueError):
                histology.transform_lasagna_csv(file_in, file_out, self.transform_dir, direction='foo')

    def test_load_track_csv_transform_dir(self):
        with tempfile.TemporaryDirectory() as td:
            file_ds, file_ara = Path(td, 'ds_pts.csv'), Path(td, 'ara_pts.csv')
            histology.write_lasagna_csv(file_ds, self.sample_apmldv[:8])
            histology.write_lasagna_csv(file_ara, self.ara_apmldv[:8])
            # a track traced in sample space loads to the same IBL coordinates as the same track in ARA space
            xyz_ds = histology.load_track_csv(file_ds, brain_atlas=brain_atlas, transform_dir=self.transform_dir)
            xyz_ara = histology.load_track_csv(file_ara, brain_atlas=brain_atlas)
            # both files are rounded to 1e-3 voxel: allow 2e-3 voxel (5e-8 m) differences
            np.testing.assert_allclose(xyz_ds, xyz_ara, rtol=0, atol=5e-8)
            self.assertTrue(np.all(np.diff(xyz_ds[:, 2]) <= 0))  # sorted along DV, deepest last
            with self.assertRaises(ValueError):
                histology.load_track_csv(file_ds, brain_atlas=atlas.AllenAtlas(res_um=50), transform_dir=self.transform_dir)
            # empty file
            Path(td, 'empty_pts.csv').touch()
            xyz = histology.load_track_csv(Path(td, 'empty_pts.csv'), brain_atlas=brain_atlas, transform_dir=self.transform_dir)
            self.assertEqual(xyz.size, 0)

    def test_read_write_lasagna_csv(self):
        with tempfile.TemporaryDirectory() as td:
            file = Path(td, 'pts.csv')
            histology.write_lasagna_csv(file, self.ara_apmldv)
            np.testing.assert_allclose(histology.read_lasagna_csv(file), self.ara_apmldv, atol=5e-4)
            # single point
            histology.write_lasagna_csv(file, self.ara_apmldv[0])
            self.assertEqual(histology.read_lasagna_csv(file).shape, (1, 3))
            # empty file
            file.write_text('')
            self.assertEqual(histology.read_lasagna_csv(file).shape, (0, 3))
            # wrong number of columns
            file.write_text('1,2\n3,4\n')
            with self.assertRaises(ValueError):
                histology.read_lasagna_csv(file)
            with self.assertRaises(ValueError):
                histology.write_lasagna_csv(file, np.zeros((4, 2)))

    def test_invalid_transform_dir(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                histology.lasagna_ara_to_sample(self.ara_apmldv, Path(td, 'missing'))
            # only the affine file
            shutil.copy(self.transform_dir / 'brainregister-affine_to-target.txt', td)
            with self.assertRaises(ValueError):
                histology.lasagna_ara_to_sample(self.ara_apmldv, td)
            # ARA2sample like files: the fixed image is the sample, not the ARA
            shutil.copy(self.transform_dir / 'brainregister-bspline_to-target.txt', td)
            for f in Path(td).glob('*.txt'):
                f.write_text(f.read_text().replace('(Size 456 320 528)', '(Size 437 313 494)'))
            with self.assertRaises(ValueError):
                histology.lasagna_ara_to_sample(self.ara_apmldv, td)

    def test_transform_file_names(self):
        expected = histology.lasagna_ara_to_sample(self.ara_apmldv, self.transform_dir)
        with tempfile.TemporaryDirectory() as td:
            # older brainregister outputs: TransformParameters.0.txt (affine), TransformParameters.1.txt (bspline)
            shutil.copy(self.transform_dir / 'brainregister-affine_to-target.txt', Path(td, 'TransformParameters.0.txt'))
            shutil.copy(self.transform_dir / 'brainregister-bspline_to-target.txt', Path(td, 'TransformParameters.1.txt'))
            # other .txt files, even elastix parameter files, are ignored
            shutil.copy(self.transform_dir / 'brainregister-affine_to-target.txt', Path(td, 'notes.txt'))
            np.testing.assert_array_equal(histology.lasagna_ara_to_sample(self.ara_apmldv, td), expected)
            # current and older names mixed: two affine files
            shutil.copy(self.transform_dir / 'brainregister-affine_to-target.txt', td)
            with self.assertRaises(ValueError):
                histology.lasagna_ara_to_sample(self.ara_apmldv, td)

    def test_atlas_resolution(self):
        with self.assertRaises(ValueError):
            histology.lasagna_to_ibl(self.ara_apmldv, brain_atlas=atlas.AllenAtlas(res_um=50))


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

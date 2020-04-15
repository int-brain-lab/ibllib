from pathlib import Path
import unittest

import numpy as np

from ibllib.pipes import histology
import ibllib.atlas as atlas


class TestHistology(unittest.TestCase):

    def setUp(self) -> None:
        self.brain_atlas = atlas.AllenAtlas(res_um=25)
        self.path_tracks = Path(__file__).parent.joinpath('fixtures', 'histology', 'tracks')

    def test_histology_get_brain_regions(self):
        # first part of the test is to check on an actual track file
        for file_track in self.path_tracks.rglob("*_pts.csv"):
            xyz = histology.load_track_csv(file_track)
            channels, ins = histology.get_brain_regions(xyz=xyz, brain_atlas=self.brain_atlas)
        # also check that it works from an insertion
        channels, ins2 = histology.get_brain_regions(xyz=ins.xyz, brain_atlas=self.brain_atlas)
        self.assertTrue(channels.acronym[-1] == 'VISpm1')
        self.assertTrue(channels.acronym[0] == 'MRN')
        a = np.array([ins.x, ins.y, ins.z, ins.phi, ins.theta, ins.depth])
        b = np.array([ins2.x, ins2.y, ins2.z, ins2.phi, ins2.theta, ins2.depth])
        self.assertTrue(np.all(np.isclose(a, b)))

    def test_histology_insertion_from_track(self):

        for file_track in self.path_tracks.rglob("*_pts.csv"):
            xyz = histology.load_track_csv(file_track)
            insertion = atlas.Insertion.from_track(xyz, brain_atlas=self.brain_atlas)
            # checks that the tip coordinate is not the deepest point but its projection
            self.assertFalse(np.all(np.isclose(insertion.tip, xyz[-1])))

    def test_get_brain_exit_entry(self):
        traj = atlas.Trajectory.fit(np.array([[0, 0, 0], [0, 0, 0.005]]))
        top = atlas.Insertion.get_brain_entry(traj, brain_atlas=self.brain_atlas)
        bottom = atlas.Insertion.get_brain_exit(traj, brain_atlas=self.brain_atlas)
        self.brain_atlas.bc.nx
        ix, iy = (self.brain_atlas.bc.x2i(0), self.brain_atlas.bc.y2i(0))
        self.assertTrue(np.isclose(self.brain_atlas.top[iy, ix], top[2]))
        self.assertTrue(np.isclose(self.brain_atlas.bottom[iy, ix], bottom[2]))

    def test_filename_parser(self):
        tdata = [
            {'input': Path("/gna/electrode_tracks_SWC_014/2019-12-12_SWC_014_001_probe01_fit.csv"),
             'output': {'date': '2019-12-12', 'experiment_number': 1, 'name': 'probe01',
                        'subject': 'SWC_014'}},
            {'input': Path("/gna/datadisk/Data/Histology/"
                           "tracks/ZM_2407/2019-11-06_ZM_2407_001_probe_00_pts.csv"),
             'output': {'date': '2019-11-06', 'experiment_number': 1, 'name': 'probe_00',
                        'subject': 'ZM_2407'}},
            {'input': Path("/gna/2019-12-06_KS023_001_probe01_pts.csv"),
             'output': {'date': '2019-12-06', 'experiment_number': 1, 'name': 'probe01',
                        'subject': 'KS023'}},
        ]
        for t in tdata:
            track_file = t['input']
            assert t['output'] == histology._parse_filename(track_file)

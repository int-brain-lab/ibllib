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

        for file_track in self.path_tracks.rglob("*_pts.csv"):
            xyz = histology.load_track_csv(file_track)
            # TODO: test 2 values (insertion)
            channels, ins = histology.get_brain_regions(xyz=xyz, brain_atlas=self.brain_atlas)
            # the algorithms gets the best fit, which won't include the last coordinate

    def test_histology_insertion_from_track(self):

        for file_track in self.path_tracks.rglob("*_pts.csv"):
            xyz = histology.load_track_csv(file_track)
            insertion = atlas.Insertion.from_track(xyz, brain_atlas=self.brain_atlas)
            # self.assertFalse(np.all(np.isclose(ins.tip, xyz[-1])))

    def test_get_brain_exit_entry(self):
        traj = atlas.Trajectory.fit(np.array([[0, 0, 0], [0, 0, 0.005]]))
        top = atlas.Insertion.get_brain_entry(traj, brain_atlas=self.brain_atlas)
        bottom = atlas.Insertion.get_brain_exit(traj, brain_atlas=self.brain_atlas)
        self.brain_atlas.bc.nx
        ix, iy = (self.brain_atlas.bc.x2i(0), self.brain_atlas.bc.y2i(0))
        self.assertTrue(np.isclose(self.brain_atlas.top[iy, ix], top[2]))
        self.assertTrue(np.isclose(self.brain_atlas.bottom[iy, ix], bottom[2]))

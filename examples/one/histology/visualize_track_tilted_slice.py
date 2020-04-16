"""
For a given track file (pts), plot the tilted slice.
"""
# Author: Olivier Winter
from ibllib.pipes import histology
import ibllib.atlas as atlas

# === Parameters section (edit) ===

track_file = "/Users/gaelle/Downloads/electrodetracks_lic3/2019-08-27_lic3_002_probe00_pts.csv"

# === Code (do not edit) ===
ba = atlas.AllenAtlas(res_um=25)
xyz_picks = histology.load_track_csv(track_file)
bl, ins = histology.get_brain_regions(xyz_picks)

sax = ba.plot_tilted_slice(ins.xyz, axis=0)
cax = ba.plot_tilted_slice(ins.xyz, axis=1)
cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6)
cax.plot(bl.xyz[:, 0] * 1e6, bl.xyz[:, 2] * 1e6, '*')

import numpy as np

from ibllib.pipes import histology
import ibllib.atlas as atlas

# Parameters section
FULL_BLOWN_GUI = True  # set to False for simple matplotlib view
track_file = "/datadisk/Data/Histology/_track/2019-12-10_KS023_001_probe01_pts.csv"

# Code
ba = atlas.AllenAtlas(res_um=25)
xyz_picks = histology.load_track_csv(track_file)
bl, ins = histology.get_brain_regions(xyz_picks)

if FULL_BLOWN_GUI:
    from iblapps.histology import atlas_mpl
    mw, cax = atlas_mpl.viewatlas(ba, ap_um=np.mean(ins.xyz[:, 1]) * 1e6)
else:
    cax = ba.plot_cslice(ap_coordinate=np.mean(ins.xyz[:, 1]))

cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6)
cax.plot(bl.xyz[:, 0] * 1e6, bl.xyz[:, 2] * 1e6, '*')
# cax.plot(ba.bc.xscale * 1e6, ba.top[ba.bc.y2i(np.mean(ins.xyz[:, 1])), :] * 1e6)

if FULL_BLOWN_GUI:
    mw.mpl_widget.draw()

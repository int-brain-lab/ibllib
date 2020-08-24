import matplotlib.pyplot as plt

import ibllib.atlas as atlas
from ibllib.pipes.histology import coverage

from oneibl.one import ONE
one = ONE()

ba = atlas.AllenAtlas()
trajs = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track')

full_coverage = coverage(trajs)

ins = atlas.Insertion.from_dict(trajs[60], brain_atlas=ba)

fig, axs = plt.subplots(2, 2)
ax = ba.plot_hslice(-4000 * 1e-6, volume=full_coverage, ax=axs[0, 0])
ax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 1] * 1e6)

ax = ba.plot_tilted_slice(xyz=ins.xyz, volume=full_coverage, axis=1, ax=axs[0, 1])
ax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6)

ax = ba.plot_tilted_slice(xyz=ins.xyz, volume=full_coverage, axis=0, ax=axs[1, 1])
ax.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6)

# ba.plot_sslice(ml_coordinate=np.mean(ins.xyz[:, 0]), volume=full_coverage)
# ba.plot_cslice(ap_coordinate=np.mean(ins.xyz[:, 1]), volume=full_coverage)

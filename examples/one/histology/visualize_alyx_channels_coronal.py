import numpy as np

import ibllib.atlas as atlas
from oneibl.one import ONE
import brainbox.io.one as bbone

# parameters section
eid = 'aad23144-0e52-4eac-80c5-c4ee2decb198'
probe_label = 'probe01'
FULL_BLOWN_GUI = True  # set to False for simple matplotlib view

# code
ba = atlas.AllenAtlas(25)
one = ONE(base_url="https://alyx.internationalbrainlab.org")
one.path_from_eid(eid)
ses = one.alyx.rest('sessions', 'read', id=eid)
traj = one.alyx.rest('trajectories', 'list', session=eid,
                     provenance='Histology track', probe=probe_label)[0]
channels = bbone.load_channel_locations(eid=eid, one=one, probe=probe_label)

ins = atlas.Insertion.from_dict(traj)

if FULL_BLOWN_GUI:
    from iblapps.histology import atlas_mpl
    mw, cax = atlas_mpl.viewatlas(ba, ap_um=np.mean(ins.xyz[:, 1]) * 1e6)
else:
    cax = ba.plot_cslice(ap_coordinate=np.mean(ins.xyz[:, 1]), volume='annotation')

cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6)
cax.plot(channels.probe01.x * 1e6, channels.probe01.z * 1e6, 'k*')

if FULL_BLOWN_GUI:
    mw.mpl_widget.draw()

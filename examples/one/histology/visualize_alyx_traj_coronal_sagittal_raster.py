'''
For a given eID, plot the probe(s) track(s) and the corresponding raster(s).
'''
# Author: Gaelle Chapuis

import numpy as np
import matplotlib.pyplot as plt
import ibllib.atlas as atlas
from oneibl.one import ONE
import brainbox.io.one as bbone

# === Parameters section (edit) ===
ba = atlas.AllenAtlas(25)
one = ONE(base_url="https://alyx.internationalbrainlab.org")

eid = 'aad23144-0e52-4eac-80c5-c4ee2decb198'

prob_des = one.load(eid, dataset_types=['probes.description'])
n_probe = len(prob_des[0])

for i_probe in range(0, n_probe):
    # Get single probe trajectory
    probe_label = prob_des[0][i_probe].get('label')

    traj = one.alyx.rest('trajectories', 'list', session=eid,
                         provenance='Histology track', probe=probe_label)[0]
    channels = bbone.load_channel_locations(eid=eid, one=one, probe=probe_label)

    ins = atlas.Insertion.from_dict(traj)

    # Initialise fig subplots
    fig, axs = plt.subplots(1, 4)

    # Coronal view
    cax = ba.plot_cslice(ap_coordinate=np.mean(ins.xyz[:, 1]), volume='annotation', ax=axs[0])
    cax.plot(ins.xyz[:, 0] * 1e6, ins.xyz[:, 2] * 1e6)
    cax.plot(channels[probe_label].x * 1e6, channels[probe_label].z * 1e6, 'k*')

    # Sagittal view
    sax = ba.plot_sslice(ap_coordinate=np.mean(ins.xyz[:, 0]), volume='annotation', ax=axs[1])
    sax.plot(ins.xyz[:, 1] * 1e6, ins.xyz[:, 2] * 1e6)
    sax.plot(channels[probe_label].y * 1e6, channels[probe_label].z * 1e6, 'k*')

    # Tilted slice


    # Raster plot

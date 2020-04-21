'''
For a given eID, plot the PLANNED (blue) and HISTOLOGY (red)
 probe(s) track(s) in 3D template brain

'''
# Author: Gaelle Chapuis

import ibllib.atlas as atlas
from oneibl.one import ONE
from mayavi import mlab
from atlaselectrophysiology import rendering

# === Parameters section (edit) ===

one = ONE(base_url="https://alyx.internationalbrainlab.org")

eid = 'aad23144-0e52-4eac-80c5-c4ee2decb198'

# == CODE SECTION (DO NOT EDIT) ==


def _plot3d_traj(traj, color, label, fig_handle,
                 ba=atlas.AllenAtlas(25), line_width=3, tube_radius=20):
    ins = atlas.Insertion.from_dict(traj)
    mlapdv = ba.xyz2ccf(ins.xyz)
    # Display the track
    mlab.plot3d(mlapdv[:, 1], mlapdv[:, 2], mlapdv[:, 0],
                line_width=line_width, color=color, tube_radius=tube_radius)

    # Setup the  label at the top of the planned trajectory
    mlab.text3d(mlapdv[0, 1], mlapdv[0, 2], mlapdv[0, 0] - 500, label,
                line_width=4, color=color, figure=fig_handle, scale=500)
    return mlapdv, ins


prob_des = one.load(eid, dataset_types=['probes.description'])
n_probe = len(prob_des[0])

# Plot empty atlas template
fig = rendering.figure()

# Loop over probes
for i_probe in range(0, n_probe):
    # Get single probe trajectory
    probe_label = prob_des[0][i_probe].get('label')

    # Histology (red)
    traj_h = one.alyx.rest('trajectories', 'list', session=eid,
                           provenance='Histology track', probe=probe_label)[0]
    color_h = (1., 0., 0.)

    # Planned (blue)
    traj_p = one.alyx.rest('trajectories', 'list', session=eid,
                           provenance='Planned', probe=probe_label)[0]
    color_p = (0., 0., 1.)

    # Plot traj
    _plot3d_traj(traj_h, color=color_h, label=probe_label, fig_handle=fig)
    _plot3d_traj(traj_p, color=color_p, label=probe_label, fig_handle=fig)

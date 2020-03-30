'''
Get list of subjects associated to the repeated site probe trajectory from ONE.
'''
# Author: Gaelle Chapuis

from oneibl.one import ONE

one = ONE()
# find projects: proj = one.alyx.rest('projects', 'list')
# proj_list = [p['name'] for p in proj]
traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                     x=-2243, y=-2000)  # repeated site coordinate
                     # TODO add filter: project='ibl_neuropixel_brainwide_01'


# Display subjects names
sess = [p['session'] for p in traj]
sub = [p['subject'] for p in sess]

print(f'Subject: {sub}')


'''
Get all probe trajectory for a given session eID.
'''
# Author: Olivier Winter

from oneibl.one import ONE

one = ONE()

eid = 'dda5fc59-f09a-4256-9fb5-66c67667a466'

# Get all trajectories for the session
trajs = one.alyx.rest('trajectories', 'list', session=eid)

# Filter by provenance
trajs = one.alyx.rest('trajectories', 'list', session=eid, provenance='micro-manipulator')

"""
Get single probe label and directory, using the probes description dataset.
"""
# Author: Gaelle Chapuis

from oneibl.one import ONE
one = ONE()

eid = 'da188f2c-553c-4e04-879b-c9ea2d1b9a93'

# --- Get single probe directory filename either by
# 1. getting probe description in alf
# 2. using alyx rest end point

# Option 1.
prob_des = one.load(eid, dataset_types=['probes.description'])
n_probe = len(prob_des[0])
# i_probe can be 0:n_probe-1 ; in this example = 1 (2 probes)
i_probe = 1
label1 = prob_des[0][i_probe].get('label')

# You can then use this label into dict, e.g. channels[label1]

# -- Set single probe directory path
session_path = one.path_from_eid(eid)
probe_dir = session_path.joinpath('alf', label1)

# Option 2. TODO

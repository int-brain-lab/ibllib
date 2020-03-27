'''
Get spikes data for all probes in a single session via ONE and brainbox.
'''
# Author: Gaelle Chapuis

from oneibl.one import ONE
import alf.io as aio
import brainbox as bb
import brainbox.io.one as bbone
# import os

one = ONE()

# --- Example session:
eid = 'da188f2c-553c-4e04-879b-c9ea2d1b9a93'

# --- Download spikes data
# 1. either a specific subset of dataset types via the one command
# 2. either the whole spikes object via the one

# Option 1
dataset_types = ['spikes.times',
                 'spikes.clusters']
one.load(eid, dataset_types=dataset_types)
session_path = one.path_from_eid(eid)
alf_dir = os.path.join(session_path, 'alf')

# Option 2 -- Download and load into memory
spks_b1 = one.load_object(eid, 'spikes')  # TODO DOES NOT WORK for multiple probes -- which probe


# --- Get probe directory either by
# 1. getting probe description in alf
# 2. using alyx rest end point

# Option 1.
prob_des = one.load(eid, dataset_types='probes.description')
n_probe = len(prob_des[0])
# i_probe can be 0:n_probe-1 ; in this example = 1 (2 probes)
i_probe = 1
label1 = prob_des[0][i_probe].get('label')

probe_dir = os.path.join(alf_dir, label1) # session_path.joinpath('alf', 'label1')

# Make bunch per probe using brainbox
spks_b = aio.load_object(probe_dir, 'spikes')
units_b = bb.processing.get_units_bunch(spks_b)

# TODO list of bunch for several probes

dic_spk_bunch, dic_clus = bbone.load_spike_sorting(eid, one=one, dataset_types=dataset_types)

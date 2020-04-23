'''
Get spikes data and associate brain regions for all probes in a single session via
ONE and brainbox.
TODO return dict of bunch via one.load_object
'''
# Author: Gaelle Chapuis

# import alf.io as aio
# import brainbox as bb
from oneibl.one import ONE
import brainbox.io.one as bbone

one = ONE()

# --- Example session:
eid = 'aad23144-0e52-4eac-80c5-c4ee2decb198'  # Example: repeated site

# Test with eid that does not have any probe planned/histology values in Alyx
# eid = 'da188f2c-553c-4e04-879b-c9ea2d1b9a93'


# ----- RECOMMENDED: Option 1 (default) ------
# 1. Get spikes, cluster (with brain regions assigned to clusters) and channels
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
del spikes, clusters, channels  # Delete for the purpose of the example

# ---------------------------------------------
# 2. Get only spikes and clusters (without brain regions assigned to clusters)
#    data separately from channels
#    Use merger function to get channels information into clusters
#    Adding feature x, y from default
spikes, clusters = bbone.load_spike_sorting(eid, one=one)
channels = bbone.load_channel_locations(eid, one=one)
keys = ['x', 'y']
clusters_brain = bbone.merge_clusters_channels(clusters, channels, keys_to_add_extra=keys)
del spikes, clusters, clusters_brain, channels  # Delete for the purpose of the example

# ---------------------------------------------
# 3. I don't want to connect to ONE and I already know my session path
session_path = one.path_from_eid(eid)  # replace by your local path
spikes, clusters = bbone.load_spike_sorting(session_path, one=one)
# TODO offline loading of channel locations ? Probably by caching the queries.

# ---------------- WIP ---------------------

# TODO one.load_object(): return dict of bunch

# --- Download spikes data
# 1. either a specific subset of dataset types via the one command
# 2. either the whole spikes object via the one
'''
# Option 1 -- Download only subset of dataset in spike object
dataset_types = ['spikes.times',
                 'spikes.clusters']
one.load(eid, dataset_types=dataset_types)


# Option 2 -- Download and load into memory the whole spikes object
spks_b1 = one.load_object(eid, 'spikes')
# TODO OUTPUT DOES NOT WORK for multiple probes,  which probe returned unknown
# TODO return dict of bunch


# --- Get single probe directory filename either by
# 1. getting probe description in alf
# 2. using alyx rest end point

# Option 1.
prob_des = one.load(eid, dataset_types=['probes.description'])
n_probe = len(prob_des[0])
# i_probe can be 0:n_probe-1 ; in this example = 1 (2 probes)
i_probe = 1
label1 = prob_des[0][i_probe].get('label')
#channels[label1]

# -- Set single probe directory path
session_path = one.path_from_eid(eid)
probe_dir = session_path.joinpath('alf', label1)

# Make bunch per probe using brainbox
spks_b = aio.load_object(probe_dir, 'spikes')
units_b = bb.processing.get_units_bunch(spks_b)
'''

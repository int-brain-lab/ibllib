"""
Get spikes data and associate brain regions for all probes in a single session via
ONE and brainbox.
TODO return dict of bunch via one.load_object
"""
# Author: Gaelle Chapuis

from one.api import ONE
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
# 3. Can also use a session path
session_path = one.path_from_eid(eid)  # replace by your local path
spikes, clusters = bbone.load_spike_sorting(session_path, one=one)

# --- Download spikes data
# 1. either a specific subset of dataset types via the one command
# 2. either the whole spikes object via the one
"""
# Option 1 -- Download only subset of dataset in spike object for 1 probe
datasets = ['spikes.times.npy',
            'spikes.clusters.npy']
spike_times, clusters = one.load_datasets(eid, datasets, collections='alf/probe00')


# Option 2 -- Download and load into memory the whole spikes object for a given probe
spikes = one.load_object(eid, 'spikes', collection='alf/probe00')

# Make bunch per probe using brainbox
units_b = bb.processing.get_units_bunch(spikes)
"""

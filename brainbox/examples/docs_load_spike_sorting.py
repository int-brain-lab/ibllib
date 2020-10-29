"""
Get spikes, clusters and channels data
========================================
Downloads and loads in spikes, clusters and channels data for a given session. Data is returned

"""

from oneibl.one import ONE
import brainbox.io.one as bbone
one = ONE()

# Find eid of interest
eid = one.search(subject='CSH_ZAD_001', date='2020-01-14')[0]

##################################################################################################
# Example 1
# =========
# Download spikes, clusters and channels data for all available probes for this session.
# The data for each probe is returned as a dict
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
print(spikes.keys())
print(spikes['probe00'].keys())

##################################################################################################
# Example 2
# =========
# Download spikes, clusters and channels data for a single probe
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one, probe='probe00')
print(spikes.keys())

##################################################################################################
# Example 3
# =========
# The default spikes and clusters datasets that are downloaded are '
# ['clusters.channels',
#  'clusters.depths',
#  'clusters.metrics',
#  'spikes.clusters',
#  'spikes.times']
# If we also want to load for example, 'clusters.peakToTrough we can add a dataset_types argument

dtypes_extra = ['clusters.peakToTrough']
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one, probe='probe00',
                                                                   dataset_types=dtypes_extra)
print(clusters['probe00'].keys())

##################################################################################################
# Example 4
# =========
# By default, load_spike_sorting_with_channel will look for available data on your
# local computer and if all the datasets are found it will not check for consistency with files
# stored on flatiron (to speed up loading process). To make sure that data is always synced to
# flatiron we can set a force flag to True
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one, probe='probe00',
                                                                   force=True)


"""
Get spikes, clusters and channels data
========================================
Downloads and loads in spikes, clusters and channels data for a given session. Data is returned

"""
import brainbox.io.one as bbone

from one.api import ONE

one = ONE(base_url='https://openalyx.internationalbrainlab.org', silent=True)

# Find eid of interest
eid = one.search(subject='CSH_ZAD_029', date='2020-09-19')[0]

##################################################################################################
# Example 1:
# Download spikes, clusters and channels data for all available probes for this session.
# The data for each probe is returned as a dict
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one)
print(spikes.keys())
print(spikes['probe01'].keys())

##################################################################################################
# Example 2:
# Download spikes, clusters and channels data for a single probe
spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one, probe='probe01')
print(spikes.keys())

##################################################################################################
# Example 3:
# The default spikes and clusters datasets that are downloaded are '
# ['clusters.channels',
#  'clusters.depths',
#  'clusters.metrics',
#  'spikes.clusters',
#  'spikes.times']
# If we also want to load for example, 'clusters.peakToTrough we can add a dataset_types argument

spikes, clusters, channels = bbone.load_spike_sorting_with_channel(eid, one=one, probe='probe01',
                                                                   dataset_types=['clusters.peakToTrough'])
print(clusters['probe01'].keys())


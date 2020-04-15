import numpy as np
from brainbox.population import decode
from sklearn.utils import shuffle
from oneibl.one import ONE
import brainbox.io.one as bbone

# %% Load in data
one = ONE()
eid = one.search(subject='ZM_2240', date_range=['2020-01-23', '2020-01-23'])
spikes, clusters = bbone.load_spike_sorting(eid[0], one=one)
trials = one.load_object(eid[0], 'trials')

# %% Only use units with KS2 label 'good' from probe00

spikes = spikes['probe00']
clusters = clusters['probe00']

clusters_to_use = clusters.metrics.ks2_label == 'good'
spikes.times = spikes.times[
        np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
spikes.clusters = spikes.clusters[
        np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
cluster_ids = clusters.metrics.cluster_id[clusters_to_use]

# %% Do decoding
print('Decoding whether the stimulus was on the left or the right..')

stim_times = trials.goCue_times
stim_sides = np.isnan(trials.contrastLeft).astype(int)

# Decode left vs right stimulus from a 1 second window after stimulus onset using default settings:
# Naive Bayes classifier with 5-fold cross-validation
decode_result = decode(spikes.times, spikes.clusters, stim_times, stim_sides,
                       pre_time=0, post_time=1)

# Get the accuracy over chance
print('\nNaive Bayes with 5-fold cross-validation')
print('Performance: %.2f%% correct [chance level: %.2f%%]' % (
                                        decode_result['accuracy'] * 100,
                                        ((stim_sides.sum() / stim_sides.shape[0]) * 100)))

# Decode stimulus side using a subset of 50 random neurons drawn 300 times
decode_result = decode(spikes.times, spikes.clusters, stim_times, stim_sides,
                       pre_time=0, post_time=1, n_neurons=50, iterations=300)
print('\nDecoding with 50 randomly drawn units')
print('Performance: %.2f%% correct [chance level: %.2f%%]' % (
                                        (decode_result['accuracy'].mean() * 100),
                                        (stim_sides.sum() / stim_sides.shape[0]) * 100))

import numpy as np
from brainbox.population import decode
import matplotlib.pyplot as plt
import seaborn as sns
from oneibl.one import ONE
import brainbox.io.one as bbone

# %% Load in data
one = ONE()
eid = one.search(subject='ZM_2240', date_range=['2020-01-23', '2020-01-23'])
spikes, clusters = bbone.load_spike_sorting(eid[0], one=one)
trials = one.load_object(eid, 'trials')

# %% Only use units with KS2 label 'good'

clusters_to_use = clusters.metrics.ks2_label == 'good'
spikes.times = spikes.times[
        np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
spikes.clusters = spikes.clusters[
        np.isin(spikes.clusters, clusters.metrics.cluster_id[clusters_to_use])]
cluster_ids = clusters.metrics.cluster_id[clusters_to_use]

# %% Classify left/right block based on population activity before stimulus onset

# Get trial indices
trial_times = trials.goCue_times
probability_left = trials.probabilityLeft[incl_trials]
trial_blocks = (trials.probabilityLeft[incl_trials] > 0.55).astype(int)

# Decode block identity using a Naive Bayes classifier with 500 ms window from -600 to -100 ms
# relative to stimulus onset, cross-validation is done using 5-fold cross-validation
decode_result = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                       pre_time=0.6, post_time=-0.1,
                       classifier='bayes', cross_validation='kfold', num_splits=5)
print('Decoding of block identity: %f F1 score' % np.round(decode_result['f1'], 2))

# Decode block identity using logistic regression with 500 ms window from -600 to -100 ms
# relative to stimulus onset, cross-validation is done using 5-fold cross-validation
decode_result = decode(spikes.times, spikes.clusters, trial_times, trial_blocks,
                       pre_time=0.6, post_time=-0.1,
                       classifier='bayes', cross_validation='kfold', num_splits=5)
print('Decoding of block identity: %f F1 score' % np.round(decode_result['f1'], 2))

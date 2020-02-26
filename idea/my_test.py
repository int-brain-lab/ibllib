import numpy as np
import matplotlib.pyplot as plt
import brainbox as bb
from brainbox.io.one import load_spike_sorting
from oneibl.one import ONE
import numpy as np
import pandas as pd


one = ONE()

eids = []

eids = one.search(subject='KS023', date_range=['2019-12-10', '2019-12-10'], task_protocol='ephysChoiceWorld')
eids = one.search(subject='ibl_witten_14', date_range=['2019-12-11', '2019-12-11'], task_protocol='ephysChoiceWorld')
eids = one.search(subject='CSHL049', date_range=['2020-01-08', '2020-01-08'], task_protocol='ephysChoiceWorld')

print(eids)

i = 0
eid = eids[i]
coords = one.load(eid, dataset_types=['probes.trajectory'])
coords = coords[0][0]


print(eid)
spikes_indiv = one.load_object(eid, 'spikes.times')
clusters_indiv = one.load_object(eid, 'spikes.clusters')

spikes, clusters = load_spike_sorting(eid)


quit()

s = spikes.times
c = clusters.clusters
ff = bb.metrics.firing_rate_fano_factor(s[c == 9], n_bins=5)
cv = bb.metrics.firing_rate_coeff_var(s[c == 9])

quit()

# extraction from folders

path = '../Downloads/FlatIron/churchlandlab/Subjects/CSHL049/2020-01-08/001/alf/'
path = '../Downloads/FlatIron/cortexlab/Subjects/KS023/2019-12-10/001/alf/'

spikes = np.load(path + 'probe01/spikes.times.npy')
clusters = np.load(path + 'probe01/spikes.clusters.npy')

mask = a < 5
neurons_5 = spikes[mask]
clusters_5 = spikes[mask]

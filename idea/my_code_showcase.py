from oneibl.one import ONE
import numpy as np

# change path name
path = '../../Downloads/FlatIron/churchlandlab/Subjects/CSHL049/2020-01-08/001/alf/'
path = '../../Downloads/FlatIron/cortexlab/Subjects/KS023/2019-12-10/001/alf/'
times_0_alf = np.load(path + 'probe00/spikes.times.npy')
times_1_alf = np.load(path + 'probe01/spikes.times.npy')
clusters_0_alf = np.load(path + 'probe00/spikes.clusters.npy')
clusters_1_alf = np.load(path + 'probe01/spikes.clusters.npy')

one = ONE()
eid = one.search(subject='CSHL049', date_range=['2020-01-08', '2020-01-08'], task_protocol='ephysChoiceWorld')
eid = one.search(subject='KS023', date_range=['2019-12-10', '2019-12-10'], task_protocol='ephysChoiceWorld')
times_direct = one.load_object(eid[0], 'spikes.times')
clusters_direct = one.load_object(eid[0], 'spikes.clusters')

print(times_0_alf.shape)
print(times_1_alf.shape)
print(times_direct.times.shape)
print()
print(clusters_0_alf.shape)
print(clusters_1_alf.shape)
print(clusters_direct.clusters.shape)

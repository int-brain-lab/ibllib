from oneibl.one import ONE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brainbox.io.one import load_channel_locations, load_spike_sorting

one = ONE()

subjects = ['KS023', 'CSHL051']
date = [['2019-12-6', '2019-12-6'], ['2020-2-05', '2020-2-05']]
probes = ['probe00', 'probe01']

eids = []

for d, s in zip(date, subjects):
    eids.append(one.search(subject=s, date_range=d, task_protocol='ephysChoiceWorld')[0])
print(eids)


for i, eid in enumerate(eids):
    print(eid)
    name = one.list(eid, 'subjects')
    print(name)

    channels = load_channel_locations(eid, one=one)
    spikes_full, clusters_full = load_spike_sorting(eid, one=one)

    for probe in probes:
        clusters = clusters_full[probe]
        cluster_regions = channels[probe].acronym[clusters.channels]
        df = pd.DataFrame(data={'Region': cluster_regions})
        df.to_csv('Cluster_to_region_{}_{}.csv'.format(subjects[i], probe))

"""
    Analysis of spike data in repeated sites
    Author: Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
import brainbox as bb
from oneibl.one import ONE
from brainbox.io.one import load_spike_sorting
import seaborn as sns


def calc_fr(spikes, clusters):
    temp = []
    for c in range(np.max(clusters)):
        if np.sum(clusters == c) == 0:
            continue
        s = spikes[clusters == c]
        fr = (s[-1] - s[0]) / len(s)
        temp.append(fr)
    return temp


def neuron_metric(metric, spikes, clusters):
    temp = []
    for c in range(np.max(clusters)):
        if np.sum(clusters == c) <= 1:
            continue
        s = spikes[clusters == c]
        m, _, _ = metric(s)
        temp.append(m)
    return temp


def metric_mean_fr(fr):
    return np.mean(fr)


def metric_max_fr(fr):
    return np.max(fr)


def metric_min_fr(fr):
    return np.min(fr)


def metric_mean_ff(spikes, clusters):
    x = neuron_metric(lambda x: bb.metrics.firing_rate_fano_factor(x, n_bins=1), spikes, clusters)
    return np.mean(x)


metric_funcs = [
    (metric_mean_fr, 'mean_firing_rate'),
    (metric_max_fr, 'max_firing_rate'),
    (metric_min_fr, 'min_firing_rate'),
    (metric_mean_ff, 'mean_fano_factor')
]

metric_funcs = [
    (metric_mean_ff, 'mean_fano_factor')
]

eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199']
labs = []
bad_eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '9a629642-3a9c-42ed-b70a-532db0e86199']#'ecb5520d-1358-434c-95ec-93687ecd1396', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']

print("Will try to compare {} data sets".format(len(eids)))

probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]
one = ONE()


assert len(eids) == len(probes)
metrics = {}
for _, metric_name in metric_funcs:
    metrics[metric_name] = []


for i, (eid, probe) in enumerate(zip(eids, probes)):
    print(eid)
    if eid in bad_eids: continue
    print("{} from {}".format(i, len(eids)))
    print(one.list(eid, 'subjects'))
    coords = one.load(eid, dataset_types=['probes.trajectory'])
    for c in coords[0]:
        if c['label'] == probe:
            print("{}, x: {}, y: {}, z: {}".format(c['label'], c['x'], c['y'], c['z']))

    continue
    spikes, _ = load_spike_sorting(eid, one=one)
    spikes = spikes[0]

    if spikes[probe]['times'] is None:
        print('empty times skip')
        continue

    fr = calc_fr(spikes[probe]['times'], spikes[probe]['clusters'])
    labs.append(one.list(eid, 'labs'))


    for j, (metric, metric_name) in enumerate(metric_funcs):

        if str.endswith(metric_name, '_fr'):
            metrics[metric_name].append(metric(fr))
        else:
            metrics[metric_name].append(metric(spikes[probe]['times'], spikes[probe]['clusters']))

def split(d, l):
    dizzle = {}
    for data, label, in zip(d, l):
        if label not in dizzle:
            dizzle[label] = [data]
        else:
            dizzle[label].append(data)
    return list(dizzle.values())


for i, (metric, metric_name) in enumerate(metric_funcs):

    data = metrics[metric_name]
    print(data)
    title = 'RSI_' + metric_name
    list_data = split(data, labs)
    print(list_data)
    plt.hist(list_data, color=sns.color_palette("colorblind", len(list_data)), stacked=True)
    plt.title(title)
    plt.savefig('../../figures/hists/' + title)
    plt.show()

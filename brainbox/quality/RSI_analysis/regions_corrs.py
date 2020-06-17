"""
    Analysis of spike data in repeated sites
    Author: Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
from oneibl.one import ONE
import brainbox as bb
import seaborn as sns
import pickle

regions = ['PO', 'LP']#, 'DG-mo', 'VISa6a', 'VISa5', 'CA1']

one = ONE()
eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199', '03cf52f6-fba6-4743-a42e-dd1ac3072343', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']
probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]

bad_eids = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']
good_eids = ['4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']


folder = "one_data/"

for region in regions:
    activities = []
    names = []

    for i, (eid, probe) in enumerate(zip(eids, probes)):
        if eid not in good_eids:
            print('skipped')
            continue
        if eid == '614e1937-4b24-4ad3-9055-c8253d089919':
            probe = 'probe00'
        print(one.list(eid, 'subjects'))



        # Data Processing from here _______________________________________________________________________________________________________________________________________
        channels = pickle.load(open(folder + eid + probe + "_channels.p", "rb"))
        spikes = pickle.load(open(folder + eid + probe + "_spikes.p", "rb"))
        clusters = pickle.load(open(folder + eid + probe + "_clusters.p", "rb"))
        times_stimon = pickle.load(open(folder + eid + probe + "_times_stimon.p", "rb"))

        cluster_channels = clusters.channels
        cluster_regions = channels.acronym[cluster_channels]

        quality = clusters.metrics.ks2_label == 'good'

        specific_region = cluster_regions == region
        qualified = np.logical_and(quality, specific_region)

        # TODO: delete
        if np.sum(qualified) < 15:
            continue

        activity, binned_spikes = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(cluster_channels))[qualified], times_stimon, bin_size=0.01)
        active = np.sum(activity.means, axis=1) != 0

        activities.append(activity.means[active])
        names.append(one.list(eid, 'subjects'))

    neuron_counts = [len(x) for x in activities]
    total_neurons = np.sum(neuron_counts)
    total_activities = np.zeros((total_neurons, 70))
    so_far = 0

    for a in activities:
        temp = len(a)
        total_activities[so_far: so_far + temp] = a
        so_far += temp


    if region == 'PO':
        plt.figure(figsize=(20, 7))
        ax1 = plt.subplot(121)
        sns.heatmap(np.corrcoef(total_activities), square=True, vmin=-1, vmax=1, ax=ax1, cbar=False)
    else:
        ax2 = plt.subplot(122)
        sns.heatmap(np.corrcoef(total_activities), square=True, vmin=-1, vmax=1, ax=ax2)
        cbar = ax2.collections[0].colorbar
        cbar.ax.tick_params(labelsize=16)

    so_far = 0
    label_points = []
    label_names = []

    for i, c in enumerate(neuron_counts):
        if c > 4:
            label_points.append(so_far + int(c / 2))
            label_names.append(names[i])

        so_far += c
        if region == 'PO':
            ax1.plot([-1, total_neurons + 1], [so_far, so_far], 'k', clip_on=False, linewidth=2)
            ax1.plot([so_far, so_far], [0, total_neurons + 1], 'k', clip_on=False, linewidth=2)
        else:
            ax2.plot([-1, total_neurons + 1], [so_far, so_far], 'k', clip_on=False, linewidth=2)
            ax2.plot([so_far, so_far], [0, total_neurons + 1], 'k', clip_on=False, linewidth=2)

    plt.yticks(label_points, label_names, rotation='horizontal', fontsize=21)
    plt.xticks(np.arange(0, total_neurons+1, int(total_neurons / 12)), np.arange(0, total_neurons+1, int(total_neurons / 12)), fontsize=16)

    plt.title("Correlations over region {}".format(region), fontsize=24)

plt.tight_layout()
plt.savefig('figures/corrs/' + region.replace('/', '_') + "double_corr_plot")
plt.show()
quit()

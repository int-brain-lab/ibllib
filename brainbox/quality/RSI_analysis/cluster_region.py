"""
    Analysis of spike data in repeated sites
    Author: Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
from oneibl.one import ONE
import seaborn as sns
import pandas as pd
import pickle
import brain_region as br


one = ONE()
eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199', '03cf52f6-fba6-4743-a42e-dd1ac3072343', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']
probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]

bad_eids = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']
good_eids = ['4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']
names_n_counts = []
names = []
all_data = []
qualities = []

folder = "one_data/"

def reappearance(names, depths, mouse):
    current = names[0]
    seen = {names[0]}
    counter = 0

    for name in names:
        if name == current:
            counter += 1
            continue
        elif name not in seen:
            counter = 1
            seen.add(current)
            current = name
            continue
        else:
            print("{} reappeared along probe after {} steps".format(br.ac2name[name], counter))

            plt.plot(np.zeros_like(depths[names == current]), depths[names == current], 'r.', label=br.ac2name[current])
            plt.plot(np.ones_like(depths[names == name]), depths[names == name], 'k.', label=br.ac2name[name])
            plt.xlim(left=-0.5, right=8.5)
            plt.legend()
            plt.xticks([])
            plt.ylabel("Probe z")
            title = mouse + '_' + br.ac2name[name]
            plt.title(title)
            plt.tight_layout()
            plt.savefig(title)
            plt.show()

            current = name
            counter = 1


def unique_to_unique(xs, ys):
    indexes = []
    for x in xs:
        indexes.append(np.where(ys == x))
    return indexes


for i, (eid, probe) in enumerate(zip(eids, probes)):
    if eid not in good_eids:
        print('skipped')
        continue
    print(eid)
    print(one.list(eid, 'subjects'))
    print(probe)

    channels = pickle.load(open(folder + eid + probe + "_channels.p", "rb"))
    spikes = pickle.load(open(folder + eid + probe + "_spikes.p", "rb"))
    clusters_full = pickle.load(open(folder + eid + probe + "_clusters.p", "rb"))

    # reappearance(channels[probe].acronym[np.argsort(channels[probe].z)], channels[probe].z[np.argsort(channels[probe].z)], one.list(eid, 'subjects'))

    spikes, clusters = spikes['times'], spikes['clusters']

    cluster_channels = clusters_full.channels
    cluster_regions = channels.acronym[cluster_channels]
    names_n_counts.append(cluster_regions)
    qualities.append(clusters_full.metrics.ks2_label)
    names.append(one.list(eid, 'subjects'))

    # reappearance(cluster_regions[np.argsort(depths)], depths[np.argsort(depths)], one.list(eid, 'subjects'))

    # x = br.name2ac['Dentate gyrus, polymorph layer']
    # y = br.name2ac['Dentate gyrus, molecular layer']
    # z = cluster_regions[np.argsort(depths)]
    #
    # print(depths[z == x])
    # print(depths[z == y])
    # quit()

fs = 15
x = 3
y = 5

for i, (d, q) in enumerate(zip(names_n_counts, qualities)):
    ax = plt.subplot(x, y, i + 1)
    plt.title(names[i], fontsize=fs)
    regions_total, indexes, counts = np.unique(d, return_index=True, return_counts=True)
    print(np.max(counts))
    regions = d[np.sort(indexes)]
    region_order = np.array(unique_to_unique(regions, regions_total)).flatten()
    _, counts_good = np.unique(np.concatenate((d[q == 'good'], regions_total)), return_counts=True)
    _, counts_bad = np.unique(np.concatenate((d[q == 'mua'], regions_total)), return_counts=True)
    ax.barh(np.arange(len(regions_total)), (counts_good - 1)[region_order], label='Good')
    ax.barh(np.arange(len(regions_total)), (counts_bad - 1)[region_order], left=(counts_good - 1)[region_order], label='mua')
    plt.yticks(range(len(regions_total)), list(regions))
    plt.xlim(left=0, right=210)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if i == 0:
        plt.legend(frameon=False, loc='lower right')

    if i / y < x - 1:
        plt.xlabel(None)
    else:
        plt.xlabel('Cluster counts', fontsize=fs)
    if i % y != 0:
        plt.ylabel(None)
    else:
        plt.ylabel('Brain region', fontsize=fs)
plt.show()


# for i, d in enumerate(names_n_counts):
#     df = pd.DataFrame(data=d, columns=["acronym"])
#     df['c'] = df['acronym'].map(colors)
#     ax = plt.subplot(3, 5, i + 1)
#     plt.title(names[i], fontsize=fs)
#     g = sns.countplot(y='acronym', data=df, orient='v')
#     if i < 3:
#         plt.xlabel(None)
#     else:
#         plt.xlabel('Cluster counts', fontsize=fs)
#     if i % 3 != 0:
#         plt.ylabel(None)
#     else:
#         plt.ylabel('Brain region', fontsize=fs)
# plt.show()

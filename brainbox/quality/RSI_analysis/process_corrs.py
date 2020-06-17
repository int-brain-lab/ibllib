import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from oneibl.one import ONE
import pickle

def add1(x):
    for i in range(len(x)-1):
        if x[i] == x[i+1]:
            x[i+1:] += 1
    return x

one = ONE()
eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199', '03cf52f6-fba6-4743-a42e-dd1ac3072343', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']
probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]

bad_eids = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']
good_eids = ['aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', 'ee5c418c-e7fa-431d-8796-b2033e000b75']


for i, (eid, probe) in enumerate(zip(eids, probes)):
    if eid not in good_eids:
        print('skipped')
        continue
    if eid == '614e1937-4b24-4ad3-9055-c8253d089919':
        probe = 'probe00'

    # pickle.dump(temp, open(eid + "_cluster_channels.p", "wb"))
    # pickle.dump(channels[probe].acronym[temp], open(eid + "_cluster_regions.p", "wb"))
    # pickle.dump(a_stim.means[active], open(eid + "activity_means.p", "wb"))
    # pickle.dump(depths, open(eid + "depths.p", "wb"))
    activity_mean_stim = pickle.load(open(eid + "activity_means.p", "rb"))
    depths = pickle.load(open(eid + "depths.p", "rb"))
    cluster_regions = pickle.load(open(eid + "_cluster_regions.p", "rb"))

    depths /= 20
    activity_mean_stim = activity_mean_stim[np.argsort(depths)]
    cluster_regions = cluster_regions[np.argsort(depths)]
    depths = depths[np.argsort(depths)]
    spread_depths = add1(depths).astype(int)
    corrs = np.ones((np.max(spread_depths)+1, activity_mean_stim.shape[0]))

    real_corrs = np.corrcoef(activity_mean_stim)
    corrs[spread_depths] = real_corrs

    full_corrs = np.ones((np.max(spread_depths)+1, np.max(spread_depths)+1))
    full_corrs[:, spread_depths] = corrs
    # sns.heatmap(full_corrs, square=True, vmin=-1, vmax=1)
    f, axarr = plt.subplots(1, 2, sharey=True, figsize=(16, 9), subplot_kw={'ymargin': -0.4})
    axarr[0].imshow(full_corrs, cmap='magma')
    title = 'spread_correlations_{}'.format(one.list(eid, 'subjects'))
    # title = 'spread_correlations_{}'.format(eid[:7])

    uniq_names = np.unique(cluster_regions)
    names_to_nums = dict(zip(uniq_names, range(1, 1+len(uniq_names))))
    colors = np.array([names_to_nums[x] for x in cluster_regions])
    color_container = np.zeros(np.max(spread_depths)+1)
    color_container[spread_depths] = colors

    axarr[1].imshow(np.repeat(color_container[:, np.newaxis], 5, axis=1), cmap='hsv')
    axarr[1].set_xticklabels([])
    axarr[1].set_yticklabels([])
    axarr[1].set_xticks([])
    axarr[1].set_yticks([])

    plt.tight_layout()
    plt.savefig(title + '.png')
    plt.close()

"""
    Analysis of spike data in repeated sites
    Author: Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
import brainbox as bb
import alf.io
from oneibl.one import ONE
from brainbox.io.one import load_spike_sorting
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram


eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199']
labs = []

print("Will try to compare {} data sets".format(len(eids)))

probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]
bad_eids = ['9a629642-3a9c-42ed-b70a-532db0e86199', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', 'ee40aece-cffd-4edb-a4b6-155f158c666a', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9']
one = ONE()


def start_reorder(tree):
    return reorder(tree[-1, 0], 'L', tree) + reorder(tree[-1, 1], 'R', tree)


def stupid_reorder(tree):
    return s_reorder(tree[-1, 0], 'L', tree) + s_reorder(tree[-1, 1], 'R', tree)


def s_reorder(node, side, tree):
    if node <= len(tree):
        return [node]
    A = s_reorder(tree[node - len(tree) - 1, 0], 'L', tree)
    B = s_reorder(tree[node - len(tree) - 1, 1], 'R', tree)
    return A + B


def leaf_depth(node, tree):
    if node <= len(tree):
        return 1
    L = leaf_depth(tree[node - len(tree) - 1, 0], tree)
    R = leaf_depth(tree[node - len(tree) - 1, 1], tree)
    return 1 + min(L, R)


def reorder(node, side, tree):
    if node <= len(tree):
        return [node]
    L = leaf_depth(tree[node - len(tree) - 1, 0], tree)
    R = leaf_depth(tree[node - len(tree) - 1, 1], tree)
    A = reorder(tree[node - len(tree) - 1, 0], 'L', tree)
    B = reorder(tree[node - len(tree) - 1, 1], 'R', tree)
    if (L < R and side == 'L') or (L > R and side == 'R'):
        return A + B
    elif (L < R and side == 'R') or (L > R and side == 'L'):
        return B + A
    else:
        # TODO: Handle this case
        return A + B


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

for i, (eid, probe) in enumerate(zip(eids, probes)):
    if eid in bad_eids: continue
    print(eid)
    spikes, _ = load_spike_sorting(eid, one=one)
    spikes = spikes[0]

    spikes, clusters = spikes[probe]['times'], spikes[probe]['clusters']

    times_stimon = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    contrastRight = one.load(eid, dataset_types=['trials.contrastRight'])[0]
    times_feedback = one.load(eid, dataset_types=['trials.feedback_times'])[0]
    feedback = one.load(eid, dataset_types=['trials.feedbackType'])[0]
    depths = np.array(one.load(eid, dataset_types=['clusters.depths']))
    block = one.load(eid, dataset_types=['trials.probabilityLeft'])[0]


    # times = times_feedback[feedback == 1]
    # times = times_stimon[(feedback == 1) & (np.nan_to_num(contrastRight) > 0)]
    times = times_stimon

    # np.random.shuffle(times)
    # times1, times2 = times[:int(len(times) / 2)], times[int(len(times) / 2):]
    times1, times2 = times[block == 0.2], times[block == 0.8]
    print(np.sum(times[block == 0.2]))
    print(np.sum(times[block == 0.8]))
    if np.sum(times[block == 0.2]) == 0 or np.sum(times[block == 0.8]) == 0:
        continue

    session_path = one.path_from_eid(eid)
    if not session_path:
        print(session_path)
        print("no session path")
        continue

    _ = one.load(eid, dataset_types='clusters.metrics', download_only=True)

    try:
        _ = alf.io.load_object(session_path.joinpath('alf'), 'probes')
    except FileNotFoundError:
        print(session_path.joinpath('alf'))
        print("no probes")
        continue

    probe_path = session_path.joinpath('alf', probe)
    try:
        metrics = alf.io.load_object(probe_path, object='clusters.metrics')
    except FileNotFoundError:
        print(probe_path)
        print("one probe missing")
        continue

    quality = metrics.metrics.ks2_label == 'good'

    for d in depths:
        if d.shape[0] == np.max(np.unique(clusters)) + 1:
            depths = d
            break

    a1, b1 = bb.singlecell.calculate_peths(spikes, clusters, quality.index[quality], times1)
    a2, b2 = bb.singlecell.calculate_peths(spikes, clusters, quality.index[quality], times2)

    depths = depths[quality]
    activity = np.logical_and(np.sum(a1.means, axis=1) != 0, np.sum(a2.means, axis=1) != 0)
    depths = depths[activity]
    a1.means = a1.means[activity]
    a2.means = a2.means[activity]

    a1.means = a1.means[np.argsort(depths)]
    correlations1 = np.corrcoef(a1.means)
    # sns.heatmap(correlations1, square=True)
    # plt.title('depth_sorted half 1')
    # plt.show()

    a2.means = a2.means[np.argsort(depths)]
    correlations2 = np.corrcoef(a2.means)
    # sns.heatmap(correlations2, square=True)
    # plt.title('depth_sorted half 2')
    # plt.show()

    sns.heatmap(np.abs(correlations1 - correlations2), square=True)
    plt.title('correlations difference session halves')
    title = "Correlation differences (block halves), Time stimon, Mouse {}".format(one.list(eid, 'subjects'))
    plt.title(title)
    plt.savefig('../../figures/' + title + '.png')
    plt.close()
    if i == 5:
        quit()

    # pca = PCA(n_components=2)
    # firstpca = pca.fit(a.means).components_
    # first_rot = pca.fit_transform(a.means)[:, 0]
    #
    # correlations = np.corrcoef(a.means[np.argsort(first_rot)])
    # sns.heatmap(correlations, square=True)
    # title = "PC sorted correlations, Time stimon, Mouse {}".format(one.list(eid, 'subjects'))
    # plt.title(title)
    # plt.savefig('../../figures/' + title + '.png')
    # plt.close()
    # quit()

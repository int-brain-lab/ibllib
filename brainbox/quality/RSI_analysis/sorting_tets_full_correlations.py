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
import pickle


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

# TODO: no resorting seems better, maybe actually try opposite of this
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
    # spikes = spikes[0]

    spikes, clusters = spikes[probe]['times'], spikes[probe]['clusters']

    times_stimon = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
    contrastRight = one.load(eid, dataset_types=['trials.contrastRight'])[0]
    times_feedback = one.load(eid, dataset_types=['trials.feedback_times'])[0]
    feedback = one.load(eid, dataset_types=['trials.feedbackType'])[0]
    depths = np.array(one.load(eid, dataset_types=['clusters.depths']))

    times_rew = times_feedback[feedback == 1]
    # times = times_stimon[(feedback == 1) & (np.nan_to_num(contrastRight) > 0)]
    times_stim = times_stimon

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

    a_stim, b_stim_temp = bb.singlecell.calculate_peths(spikes, clusters, quality.index[quality], times_stim)
    a_rew, b_rew_temp = bb.singlecell.calculate_peths(spikes, clusters, quality.index[quality], times_rew)

    b_stim = np.zeros((b_stim_temp.shape[1], b_stim_temp.shape[2] * b_stim_temp.shape[0]))
    b_rew = np.zeros((b_rew_temp.shape[1], b_rew_temp.shape[2] * b_rew_temp.shape[0]))

    for i in range(b_rew_temp.shape[1]):
        b_stim[i] = b_stim_temp[:, i].flatten()
        b_rew[i] = b_rew_temp[:, i].flatten()


    # depths = depths[quality]
    # activity = np.logical_and(np.sum(a_stim.means, axis=1) != 0, np.sum(a_rew.means, axis=1) != 0)
    # depths = depths[activity]
    # mean_stim = a_stim.means[activity]
    # mean_rew = a_rew.means[activity]

    depths = depths[quality]
    activity = np.logical_and(np.sum(b_stim, axis=1) != 0, np.sum(b_rew, axis=1) != 0)
    depths = depths[activity]
    mean_stim = b_stim[activity]
    mean_rew = b_rew[activity]

    # a.means = a.means[np.argsort(depths)]


    #clustering = AgglomerativeClustering(linkage='ward', distance_threshold=0, n_clusters=None)
    clustering = AgglomerativeClustering(linkage='average', affinity='precomputed', distance_threshold=0, n_clusters=None)
    model = clustering.fit(1 - np.corrcoef(mean_stim))
    # plot_dendrogram(model, truncate_mode='level', p=4)
    # title = "Dendogram, Time stimon, Mouse {}".format(one.list(eid, 'subjects'))
    # plt.title(title)
    # plt.savefig('../../figures/' + title + '.png')
    # plt.close()
    # continue

    cluster_sort_stim = stupid_reorder(model.children_)

    clustering = AgglomerativeClustering(linkage='average', affinity='precomputed', distance_threshold=0, n_clusters=None)
    model = clustering.fit(1 - np.corrcoef(mean_rew))
    # plot_dendrogram(model, truncate_mode='level', p=4)
    # title = "Dendogram, Time stimon, Mouse {}".format(one.list(eid, 'subjects'))
    # plt.title(title)
    # plt.savefig('../../figures/' + title + '.png')
    # plt.close()
    # continue

    cluster_sort_rew = stupid_reorder(model.children_)


    fig = plt.figure(figsize=(11.4, 11))
    ax1 = plt.subplot2grid((20, 23), (0, 0), rowspan=10, colspan=10)
    ax2 = plt.subplot2grid((20, 23), (0, 10), rowspan=10, colspan=10)
    ax3 = plt.subplot2grid((20, 23), (10, 0), rowspan=10, colspan=10)
    ax4 = plt.subplot2grid((20, 23), (10, 10), rowspan=10, colspan=10)
    ax5 = plt.subplot2grid((20, 23), (0, 20), rowspan=10, colspan=1)
    ax6 = plt.subplot2grid((20, 23), (10, 20), rowspan=10, colspan=1)
    ax7 = plt.subplot2grid((20, 23), (5, 21), rowspan=10, colspan=1)
    ax8 = plt.subplot2grid((20, 23), (5, 22), rowspan=10, colspan=1)

    sns.heatmap(np.corrcoef(mean_stim[cluster_sort_stim]), ax=ax1, cbar=False, square=True)
    sns.heatmap(np.corrcoef(mean_rew[cluster_sort_stim]), ax=ax2, cbar=False, square=True)
    sns.heatmap(np.corrcoef(mean_stim[cluster_sort_rew]), ax=ax3, cbar=False, square=True)
    # sns.heatmap(np.corrcoef(np.arange(mean_rew.size).reshape(mean_rew.shape)), ax=ax3, cbar=False, square=True) # ordering seems to work out
    sns.heatmap(np.corrcoef(mean_rew[cluster_sort_rew]), ax=ax4, cbar=False, square=True)

    fs = 15

    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title("Reward Correlations", fontsize=fs)

    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticks(np.arange(0, depths.size + 1, int(depths.size / 12)))
    ax1.set_yticklabels(np.arange(0, depths.size + 1, int(depths.size / 12)))
    ax1.set_ylabel("Stimulus Onset Sorting", fontsize=fs)
    ax1.set_title("Stimulus Onset Correlations", fontsize=fs)

    ax4.set_yticklabels([])
    ax4.set_yticks([])
    ax4.set_xticks(np.arange(0, depths.size + 1, int(depths.size / 12)))
    ax4.set_xticklabels(np.arange(0, depths.size + 1, int(depths.size / 12)))

    ax3.set_yticks(np.arange(0, depths.size + 1, int(depths.size / 12)))
    ax3.set_xticks(np.arange(0, depths.size + 1, int(depths.size / 12)))
    ax3.set_yticklabels(np.arange(0, depths.size + 1, int(depths.size / 12)))
    ax3.set_xticklabels(np.arange(0, depths.size + 1, int(depths.size / 12)))
    ax3.set_ylabel("Reward Sorting", fontsize=fs)

    ax5.imshow(np.repeat(depths[cluster_sort_stim][:, np.newaxis], 5, axis=1), cmap='viridis')
    ax5.set_xticklabels([])
    ax5.set_yticklabels([])
    ax5.set_xticks([])
    ax5.set_yticks([])

    ax6.imshow(np.repeat(depths[cluster_sort_rew][:, np.newaxis], 5, axis=1), cmap='viridis')
    ax6.set_xticklabels([])
    ax6.set_yticklabels([])
    ax6.set_xticks([])
    ax6.set_yticks([])

    ax7.imshow(np.repeat(depths[cluster_sort_stim][:, np.newaxis], 5, axis=1), cmap='viridis')
    ax7.set_xticklabels([])
    ax7.set_yticklabels([])
    ax7.set_xticks([])
    ax7.set_yticks([])

    ax8.imshow(np.repeat(depths[cluster_sort_rew][:, np.newaxis], 5, axis=1), cmap='viridis')
    ax8.set_xticklabels([])
    ax8.set_yticklabels([])
    ax8.set_xticks([])
    ax8.set_yticks([])

    # plt.annotate("test", (0.84, 0.885), fontsize=fs, xycoords='figure fraction')
    title = "Cross sorted Correlations, Mouse {}".format(one.list(eid, 'subjects'))
    plt.suptitle(title, fontsize=fs+3)
    # plt.savefig('../../figures/' + title + '.png')
    # plt.close()
    plt.show()
    quit()
    continue



    sns.heatmap(correlations, square=True)
    title = "Hierarchy sorted correlations, Time stimon (correct & right), Mouse {}".format(one.list(eid, 'subjects'))
    plt.title(title)
    # plt.savefig('../../figures/' + title + '.png')
    # plt.close()
    plt.show()
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

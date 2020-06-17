import numpy as np
import brainbox as bb
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

regions = ['DG-mo', 'VISa6a', 'VISa5', 'LP', 'CA1', 'PO']

eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199', '03cf52f6-fba6-4743-a42e-dd1ac3072343', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']
probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]

bad_eids = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']
good_eids = ['aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', 'ee5c418c-e7fa-431d-8796-b2033e000b75']

folder = "one_data/"


for region in regions:
    activities = []
    activities_right = []
    activities_left = []

    for i, (eid, probe) in enumerate(zip(eids, probes)):
        if eid not in good_eids:
            print('skipped')
            continue
        if eid == '614e1937-4b24-4ad3-9055-c8253d089919':
            probe = 'probe00'

        # Data Processing from here _______________________________________________________________________________________________________________________________________
        channels = pickle.load(open(folder + eid + probe + "_channels.p", "rb"))
        spikes = pickle.load(open(folder + eid + probe + "_spikes.p", "rb"))
        clusters = pickle.load(open(folder + eid + probe + "_clusters.p", "rb"))
        times_stimon = pickle.load(open(folder + eid + probe + "_times_stimon.p", "rb"))
        times_feedback = pickle.load(open(folder + eid + probe + "_times_feedback.p", "rb"))
        feedback = pickle.load(open(folder + eid + probe + "_feedback.p", "rb"))
        signed_contrast = pickle.load(open(folder + eid + probe + "_signed_contrast.p", "rb"))

        times_stim_right_full = times_stimon[signed_contrast == 1]
        times_stim_left_full = times_stimon[signed_contrast == -1]

        cluster_channels = clusters.channels
        cluster_regions = channels.acronym[cluster_channels]

        quality = clusters.metrics.ks2_label == 'good'

        specific_region = cluster_regions == region
        qualified = np.logical_and(quality, specific_region)

        activity, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(cluster_channels))[qualified], times_stimon)
        activity_right, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(cluster_channels))[qualified], times_stim_right_full)
        activity_left, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(cluster_channels))[qualified], times_stim_left_full)
        active = np.sum(activity.means, axis=1) != 0
        activity = activity.means[active]
        # To here _______________________________________________________________________________________________________________________________________________________

        activities.append(activity)
        activities_right.append(activity_right.means[active])
        activities_left.append(activity_left.means[active])

    total_neurons = sum(a.shape[0] for a in activities)

    tscale = np.array([-0.1875, -0.1625, -0.1375, -0.1125, -0.0875, -0.0625, -0.0375,
    -0.0125,  0.0125,  0.0375,  0.0625,  0.0875,  0.1125,  0.1375,
    0.1625,  0.1875,  0.2125,  0.2375,  0.2625,  0.2875,  0.3125,
    0.3375,  0.3625,  0.3875,  0.4125,  0.4375,  0.4625,  0.4875])

    colors = np.zeros(total_neurons)
    mixed_acts = np.zeros((total_neurons, 28))
    so_far = 0
    for i, a in enumerate(activities):
        if a.shape[0] > 8:
            mixed_acts[so_far:so_far + a.shape[0]] = a
            colors[so_far:so_far + a.shape[0]] = i
            so_far += a.shape[0]
            pca = PCA(n_components=2)
            pca.fit(a)
            plt.plot(tscale, pca.components_[0])
            title = "Region {}, mouse {}, first time PC stim onset".format(region, i)
            plt.savefig('figures/' + title + '.png')
            plt.close()

    total_neurons_left = sum(a.shape[0] for a in activities_left)
    mixed_acts_left = np.zeros((total_neurons_left, 28))
    so_far = 0
    for i, a in enumerate(activities_left):
        mixed_acts_left[so_far:so_far + a.shape[0]] = a
        so_far += a.shape[0]

    total_neurons_right = sum(a.shape[0] for a in activities_right)
    mixed_acts_right = np.zeros((total_neurons_right, 28))
    so_far = 0
    for i, a in enumerate(activities_right):
        mixed_acts_right[so_far:so_far + a.shape[0]] = a
        so_far += a.shape[0]


    pca = PCA(n_components=2)
    pca.fit(mixed_acts)
    plt.plot(tscale, pca.components_[0])
    title = "Region {}, all mice, first time PC stim onset".format(region, i)
    plt.savefig('figures/' + title + '.png')
    plt.close()

    continue
    #firstpca = pca.fit(mixed_acts).components_
    #first_rot = pca.fit_transform(mixed_acts)[:, :2]
    pca.fit(np.swapaxes(mixed_acts, 0, 1))
    # first_rot = pca.fit_transform(np.swapaxes(mixed_acts, 0, 1))[:, :2]
    first_rot = pca.transform(np.swapaxes(mixed_acts, 0, 1))[:, :2]

    mark = (first_rot[tscale == -0.0125] + first_rot[tscale == 0.0125]) / 2
    # plt.scatter(first_rot[:, 0], first_rot[:, 1], c=colors)
    plt.scatter(first_rot[:, 0], first_rot[:, 1], c=tscale)
    plt.colorbar()
    plt.plot(mark[0, 0], mark[0, 1], 'kx')
    title = region + "stim_on_PC_dynamics"
    plt.title(title)
    plt.savefig('figures/' + title + '.png')
    plt.close()

    right_rot = pca.transform(np.swapaxes(mixed_acts_right, 0, 1))[:, :2]
    plt.scatter(right_rot[:, 0], right_rot[:, 1], c='r')
    left_rot = pca.transform(np.swapaxes(mixed_acts_left, 0, 1))[:, :2]
    plt.scatter(left_rot[:, 0], left_rot[:, 1], c='b')
    plt.colorbar()
    plt.plot(mark[0, 0], mark[0, 1], 'kx')
    title = region + "stim_on_separated_PC_dynamics"
    plt.title(title)
    plt.savefig('figures/' + title + '.png')
    plt.close()

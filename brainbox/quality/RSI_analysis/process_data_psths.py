import numpy as np
import brainbox as bb
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

regions = ['PO', 'DG-mo', 'VISa6a', 'VISa5', 'LP', 'CA1', ]

eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199', '03cf52f6-fba6-4743-a42e-dd1ac3072343', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']
probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]

bad_eids = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']
good_eids = ['aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', 'ee5c418c-e7fa-431d-8796-b2033e000b75']

folder = "one_data/"

region_to_act = {region: [] for region in regions}
tscale = 0

for i, (eid, probe) in enumerate(zip(eids, probes)):
    if eid not in good_eids:
        print('skipped')
        continue
    if eid == '614e1937-4b24-4ad3-9055-c8253d089919':
        probe = 'probe00'

    channels = pickle.load(open(folder + eid + probe + "_channels.p", "rb"))
    spikes = pickle.load(open(folder + eid + probe + "_spikes.p", "rb"))
    clusters = pickle.load(open(folder + eid + probe + "_clusters.p", "rb"))
    times_stimon = pickle.load(open(folder + eid + probe + "_times_stimon.p", "rb"))
    times_feedback = pickle.load(open(folder + eid + probe + "_times_feedback.p", "rb"))
    feedback = pickle.load(open(folder + eid + probe + "_feedback.p", "rb"))
    signed_contrast = pickle.load(open(folder + eid + probe + "_signed_contrast.p", "rb"))

    cluster_channels = clusters.channels
    cluster_regions = channels.acronym[cluster_channels]

    quality = clusters.metrics.ks2_label == 'good'

    for region in regions:
        specific_region = cluster_regions == region
        qualified = np.logical_and(quality, specific_region)
        region_clusters = qualified[qualified].index
        if region_clusters.shape[0] < 3:
            continue
        valid_spikes = np.isin(spikes.clusters, region_clusters)

        region_spikes, region_clusters = spikes.times[valid_spikes], spikes.clusters[valid_spikes]

        a = bb.task.responsive_units(region_spikes, region_clusters, times_stimon, pre_time=[0.3, 0], post_time=[0, 0.3])

        activity, binned_spikes = bb.singlecell.calculate_peths(region_spikes, region_clusters, a[0], times_stimon, bin_size=0.01)
        tscale = activity.tscale

        region_to_act[region].append(np.mean(binned_spikes, axis=0))

for region in regions:

    all_act = np.concatenate(region_to_act[region])

    fig, axs = plt.subplots(1, 2, figsize=(16, 9), sharex=True, sharey=True)
    axs[0].plot(tscale, np.mean(all_act, axis=0))
    for a in region_to_act[region]:
        axs[1].plot(tscale, np.mean(a, axis=0), label=a.shape[0])

    plt.legend()
    fig.suptitle(region)
    plt.show()

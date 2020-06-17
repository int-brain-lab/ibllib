import numpy as np
import brainbox as bb
import pickle
import scipy

regions = ['any', 'PO', 'DG-mo', 'VIS', 'LP', 'CA1']

eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199', '03cf52f6-fba6-4743-a42e-dd1ac3072343', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']
probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]

bad_eids = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']
good_eids = ['4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']

folder = "one_data/"

region_save = {region: [] for region in regions}

for region in regions:

    for i, (eid, probe) in enumerate(zip(eids, probes)):
        if eid not in good_eids:
            print('skipped')
            continue

        # Data Processing from here _______________________________________________________________________________________________________________________________________
        channels = pickle.load(open(folder + eid + probe + "_channels.p", "rb"))
        spikes = pickle.load(open(folder + eid + probe + "_spikes.p", "rb"))
        clusters = pickle.load(open(folder + eid + probe + "_clusters.p", "rb"))
        times_stimon = pickle.load(open(folder + eid + probe + "_times_stimon.p", "rb"))
        times_feedback = pickle.load(open(folder + eid + probe + "_times_feedback.p", "rb"))
        feedback = pickle.load(open(folder + eid + probe + "_feedback.p", "rb"))
        signed_contrast = pickle.load(open(folder + eid + probe + "_signed_contrast.p", "rb"))
        choices = pickle.load(open(folder + eid + probe + "_choices.p", "rb"))
        choices[choices == -1] = 0


        print(feedback.shape[0] - feedback[np.logical_or(feedback == 1, signed_contrast == 0)].shape[0])
        times_stimon = times_stimon[np.logical_or(feedback == 1, signed_contrast == 0)]
        times_feedback = times_feedback[np.logical_or(feedback == 1, signed_contrast == 0)]
        choices = choices[np.logical_or(feedback == 1, signed_contrast == 0)]
        signed_contrast = signed_contrast[np.logical_or(feedback == 1, signed_contrast == 0)]

        signed_contrast = signed_contrast[~ np.isnan(times_feedback)]
        times_stimon = times_stimon[~ np.isnan(times_feedback)]
        choices = choices[~ np.isnan(times_feedback)]
        times_feedback = times_feedback[~ np.isnan(times_feedback)]

        x = times_feedback - times_stimon
        times_stimon = times_stimon[x < 4]
        times_feedback = times_feedback[x < 4]
        signed_contrast = signed_contrast[x < 4]
        choices = choices[x < 4]
        print(np.sum(x >= 4))

        conts, conts_counts = np.unique(signed_contrast, return_counts=True)

        if np.min(conts_counts) < 10:
            continue

        cluster_channels = clusters.channels
        cluster_regions = channels.acronym[cluster_channels]

        quality = clusters.metrics.ks2_label == 'good'

        if region == 'any':
            specific_region = True
        elif region == 'VIS':
            inds = [i for i, si in enumerate(cluster_regions) if si.startswith('VIS')]
            specific_region = np.zeros(cluster_regions.shape)
            specific_region[inds] = 1
        else:
            specific_region = cluster_regions == region
        qualified = np.logical_and(quality, specific_region)

        activity, binned_spikes = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(cluster_channels))[qualified], times_stimon, pre_time=0.2, post_time=0.7, smoothing=0, bin_size=0.01)

        avs = np.zeros((binned_spikes.shape[1], 5, 2, binned_spikes.shape[2] - 8))
        all_data = np.zeros((np.max(conts_counts), binned_spikes.shape[1], 5, 2, binned_spikes.shape[2] - 8))
        all_data.fill(np.nan)
        cont_to_num = dict(zip([1, 0.25, 0.125, 0.0625, 0], [0, 1, 2, 3, 4]))
        for i in range(binned_spikes.shape[0]):
            for j in range(binned_spikes.shape[1]):
                binned_spikes[i, j] = scipy.ndimage.gaussian_filter(binned_spikes[i, j], sigma=2.4)

        for c in [1, 0.25, 0.125, 0.0625, 0]:
            for n in range(binned_spikes.shape[1]):
                avs[n, cont_to_num[c], 0] = np.mean(binned_spikes[np.logical_and(np.abs(signed_contrast) == c, choices == 0), n], axis=0)[4:-4]
                avs[n, cont_to_num[c], 1] = np.mean(binned_spikes[np.logical_and(np.abs(signed_contrast) == c, choices == 1), n], axis=0)[4:-4]
                trials = binned_spikes[np.logical_and(np.abs(signed_contrast) == c, choices == 0), n].shape[0]
                all_data[:trials, n, cont_to_num[c], 0] = binned_spikes[np.logical_and(np.abs(signed_contrast) == c, choices == 0), n][:, 4: -4]
                trials = binned_spikes[np.logical_and(np.abs(signed_contrast) == c, choices == 1), n].shape[0]
                all_data[:trials, n, cont_to_num[c], 1] = binned_spikes[np.logical_and(np.abs(signed_contrast) == c, choices == 1), n][:, 4: -4]

        region_save[region].append(avs)

        pickle.dump(avs, open(folder + eid + probe + region + "dpca_spikes_dec_stim.p", "wb"))
        pickle.dump(all_data, open(folder + eid + probe + region + "all_data_dec_stim.p", "wb"))

    pickle.dump(np.concatenate(region_save[region]), open(folder + region + "_save_stim.p", "wb"))

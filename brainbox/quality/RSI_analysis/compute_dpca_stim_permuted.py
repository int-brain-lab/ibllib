import numpy as np
from dPCA import dPCA
import pickle
import matplotlib.pyplot as plt

regions = ['CA1', 'any', 'DG-mo', 'VIS', 'LP', 'PO']

eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199', '03cf52f6-fba6-4743-a42e-dd1ac3072343', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']
probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]

bad_eids = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']
good_eids = ['4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']

folder = "one_data/"


tscale = np.linspace(-0.195, 0.695, 90)
tscale = tscale[4:-4]

color_dict = dict(zip(list(range(5)), ['k', 'r', 'b', 'y', 'g']))
label_dict = dict(zip(list(range(5)), ['1 Left', '0.25 Left', '0.125 Left', '0.0625 Left', '0 Left']))
label_dict2 = dict(zip(list(range(5)), ['1 Right', '0.25 Right', '0.125 Right', '0.0625 Right', '0 Right']))

join_dict = {'st': ['s', 'st'], 'dt': ['d', 'dt'], 'sdt': ['sd', 'sdt']}
annotation_xy = (0.07, 0.935)
annotation_fs = 21

for region in regions:

    all_region_data = pickle.load(open(folder + region + "_save_stim_permuted_choice.p", "rb"))

    dpca = dPCA.dPCA(labels='sdt', join=join_dict, n_components=15)#, regularizer=0.001)
    dpca.protect = ['t']
    dpca.n_trials = 6
    all_region_data -= np.mean(all_region_data, (1, 2, 3))[:, None, None, None]
    Z = dpca.fit_transform(all_region_data)


    plt.figure(figsize=(20, 7))
    ax1 = plt.subplot(141)
    S = 5
    D = 2

    for s in range(S):
        plt.plot(tscale, Z['t'][0, s, 0], c=color_dict[s], ls='-', label=label_dict[s])
        plt.plot(tscale, Z['t'][0, s, 1], c=color_dict[s], ls='--', label=label_dict2[s])
    plt.legend(frameon=False, title='Contrasts', fontsize=13, title_fontsize=14)

    ax1.annotate('{:.1f}%'.format(100 * dpca.explained_variance_ratio_['t'][0]), annotation_xy, xycoords='axes fraction', fontsize=annotation_fs, c='r')
    plt.title('1st time dPC', fontsize=21)

    ax2 = plt.subplot(142, sharey=ax1)

    for s in range(S):
        plt.plot(tscale, Z['t'][1, s, 0], c=color_dict[s], ls='-')
        plt.plot(tscale, Z['t'][1, s, 1], c=color_dict[s], ls='--')

    ax2.annotate('{:.1f}%'.format(100 * dpca.explained_variance_ratio_['t'][1]), annotation_xy, xycoords='axes fraction', fontsize=annotation_fs, c='r')
    plt.title('2nd time dPC', fontsize=21)

    ax3 = plt.subplot(143, sharey=ax1)

    for s in range(S):
        plt.plot(tscale, Z['st'][0, s, 0], c=color_dict[s], ls='-')
        plt.plot(tscale, Z['st'][0, s, 1], c=color_dict[s], ls='--')

    ax3.annotate('{:.1f}%'.format(100 * dpca.explained_variance_ratio_['st'][0]), annotation_xy, xycoords='axes fraction', fontsize=annotation_fs, c='r')
    plt.title('1st stimulus dPC', fontsize=21)

    ax4 = plt.subplot(144, sharey=ax1)

    for s in range(S):
        plt.plot(tscale, Z['dt'][0, s, 0], c=color_dict[s], ls='-')
        plt.plot(tscale, Z['dt'][0, s, 1], c=color_dict[s], ls='--')

    ax4.annotate('{:.1f}%'.format(100 * dpca.explained_variance_ratio_['dt'][0]), annotation_xy, xycoords='axes fraction', fontsize=annotation_fs, c='r')
    plt.title('1st decision dPC', fontsize=21)

    # ax5 = plt.subplot(155, sharey=ax1)
    #
    # for s in range(S):
    #     plt.plot(tscale, Z['sdt'][0, s, 0], c=color_dict[s], ls='-')
    #     plt.plot(tscale, Z['sdt'][0, s, 1], c=color_dict[s], ls='--')
    #
    # plt.title('1st interaction dPC, expl. var. {:.4f}'.format(dpca.explained_variance_ratio_['sdt'][0]))

    ax1.set_ylabel('Firing rate', fontsize=24)
    ax1.set_xlabel('Time around stimulus onset (s)', fontsize=24)
    plt.setp(ax2.get_yticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)

    plt.suptitle("{} neurons, {} region".format(all_region_data.shape[0], region), fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figures/dpca/stim_time_all_neurons_choice_permuted' + '_' + region + '.png')
    plt.show()
    continue

    for i, (eid, probe) in enumerate(zip(eids, probes)):
        if eid not in good_eids:
            # print('skipped')
            continue
        if eid == '614e1937-4b24-4ad3-9055-c8253d089919':
            probe = 'probe00'

        # Data Processing from here _______________________________________________________________________________________________________________________________________
        try:
            avs = pickle.load(open(folder + eid + probe + region + "dpca_spikes_dec.p", "rb"))
            all_data = pickle.load(open(folder + eid + probe + region + "all_data_dec.p", "rb"))
        except:
            continue

        print(avs.shape)
        if avs.shape[0] == 0:
            continue
        if avs.shape[0] < 3:
            continue

        dpca = dPCA.dPCA(labels='sdt', regularizer='auto', join=join_dict)
        dpca.protect = ['t']
        dpca.n_trials = 6
        avs -= np.mean(avs, (1, 2, 3))[:, None, None, None]
        Z = dpca.fit_transform(avs, trialX=all_data)

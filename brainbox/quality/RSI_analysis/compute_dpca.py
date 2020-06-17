import numpy as np
from dPCA import dPCA
import pickle
import matplotlib.pyplot as plt

regions = ['any', 'DG-mo', 'VISa6a', 'VISa5', 'LP', 'CA1', 'PO']

eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199', '03cf52f6-fba6-4743-a42e-dd1ac3072343', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']
probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]

bad_eids = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']
good_eids = ['4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']

folder = "one_data/"


tscale = np.array([-0.195, -0.185, -0.175, -0.165, -0.155, -0.145, -0.135, -0.125,
                   -0.115, -0.105, -0.095, -0.085, -0.075, -0.065, -0.055, -0.045,
                   -0.035, -0.025, -0.015, -0.005,  0.005,  0.015,  0.025,  0.035,
                   0.045,  0.055,  0.065,  0.075,  0.085,  0.095,  0.105,  0.115,
                   0.125,  0.135,  0.145,  0.155,  0.165,  0.175,  0.185,  0.195,
                   0.205,  0.215,  0.225,  0.235,  0.245,  0.255,  0.265,  0.275,
                   0.285,  0.295,  0.305,  0.315,  0.325,  0.335,  0.345,  0.355,
                   0.365,  0.375,  0.385,  0.395,  0.405,  0.415,  0.425,  0.435,
                   0.445,  0.455,  0.465,  0.475,  0.485,  0.495,  0.505,  0.515,
                   0.525,  0.535,  0.545,  0.555,  0.565,  0.575,  0.585,  0.595,
                   0.605,  0.615,  0.625,  0.635,  0.645,  0.655,  0.665,  0.675,
                   0.685,  0.695])
tscale = tscale[4:-4]

color_dict = dict(zip(list(range(9)), ['k', 'r', 'b', 'y', 'g', 'y', 'b', 'r', 'k']))
line_dict = dict(zip(list(range(9)), ['-', '-', '-', '-', '-', '--', '--', '--', '--']))
label_dict = dict(zip(list(range(9)), ['1', '0.25', '0.125', '0.0625', '0', '_nolegend_', '_nolegend_', '_nolegend_', '_nolegend_']))


for region in regions:
    activities = []
    activities_right = []
    activities_left = []

    for i, (eid, probe) in enumerate(zip(eids, probes)):
        if eid not in good_eids:
            # print('skipped')
            continue

        # Data Processing from here _______________________________________________________________________________________________________________________________________
        try:
            avs = pickle.load(open(folder + eid + probe + region + "dpca_spikes_dec.p", "rb"))
            all_data = pickle.load(open(folder + eid + probe + region + "all_data_dec.p", "rb"))
        except:
            print('not found')
            continue

        print(avs.shape)
        if avs.shape[0] == 0:
            continue
        if avs.shape[0] < 3:
            continue

        dpca = dPCA.dPCA(labels='st', regularizer='auto')
        dpca.protect = ['t']
        dpca.n_trials = 6
        Z = dpca.fit_transform(avs, trialX=all_data)

        plt.figure(figsize=(16, 7))
        ax1 = plt.subplot(141)
        S = 9

        for s in range(S):
            plt.plot(tscale, Z['t'][0, s], c=color_dict[s], ls=line_dict[s], label=label_dict[s])
        plt.legend()

        plt.title('1st time c, {:.4f}'.format(dpca.explained_variance_ratio_['t'][0]))

        plt.subplot(142, sharey=ax1)

        for s in range(S):
            plt.plot(tscale, Z['t'][1, s], c=color_dict[s], ls=line_dict[s])

        plt.title('2st time c, {:.4f}'.format(dpca.explained_variance_ratio_['t'][1]))

        plt.subplot(143, sharey=ax1)

        for s in range(S):
            plt.plot(tscale, Z['s'][0, s], c=color_dict[s], ls=line_dict[s])

        plt.title('1st stimulus c, {:.4f}'.format(dpca.explained_variance_ratio_['s'][0]))

        plt.subplot(144, sharey=ax1)

        for s in range(S):
            plt.plot(tscale, Z['st'][0, s], c=color_dict[s], ls=line_dict[s])

        plt.title('1st mixing c, {:.4f}'.format(dpca.explained_variance_ratio_['st'][0]))
        plt.suptitle("{} neurons, {} region, {} max trials".format(avs.shape[0], region, all_data.shape[0]))
        #plt.savefig('figures/dpca/correct_regged' + eid + '_' + region + '.png')
        plt.show()

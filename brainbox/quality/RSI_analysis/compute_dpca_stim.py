import numpy as np
from dPCA import dPCA
import pickle
import matplotlib.pyplot as plt

regions = ['CA1', 'any', 'DG-mo', 'VIS', 'LP', 'PO']

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

    all_region_data = pickle.load(open(folder + region + "_save_stim.p", "rb"))

    dpca = dPCA.dPCA(labels='sdt', join=join_dict, n_components=15)#, regularizer=0.001)
    dpca.protect = ['t']
    dpca.n_trials = 6
    all_region_data -= np.mean(all_region_data, (1, 2, 3))[:, None, None, None]
    Z = dpca.fit_transform(all_region_data)

    S = 5
    D = 2

    plt.figure(figsize=(20, 14))
    fs = 15
    rows = 4
    cols = 3

    components = ['t', 'st', 'dt', 'sdt']

    for i in range(3 * 4):
        ax = plt.subplot(rows, cols, i + 1)

        for s in range(S):
            ax.plot(tscale, Z[components[i // cols]][i % cols, s, 0], c=color_dict[s], ls='-', label=label_dict[s])
            ax.plot(tscale, Z[components[i // cols]][i % cols, s, 1], c=color_dict[s], ls='--', label=label_dict2[s])

        plt.ylim(bottom=-0.6, top=0.6)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if i == 0:
            plt.legend(frameon=False, title='Contrasts', fontsize=13, title_fontsize=14)

        ax.annotate('{:.1f}%'.format(100 * dpca.explained_variance_ratio_[components[i // cols]][i % cols]), annotation_xy, xycoords='axes fraction', fontsize=annotation_fs, c='r')
        plt.title('{}. {} dPC'.format(i % cols + 1, components[i // cols]), fontsize=21)

        if i == 0:
            plt.legend(frameon=False, loc='lower right')

        if i / cols < rows - 1:
            plt.xlabel(None)
            plt.xticks([])
        else:
            plt.xlabel('time (s)', fontsize=fs)
        if i % cols != 0:
            plt.ylabel(None)
            plt.yticks([])
        else:
            plt.ylabel('Firing rate', fontsize=fs)

    plt.suptitle("{} neurons, {} region".format(all_region_data.shape[0], region), fontsize=24)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('figures/dpca/stimon_time_all_neurons_complete_' + region + '.png')
    plt.show()

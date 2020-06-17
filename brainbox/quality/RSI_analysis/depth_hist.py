"""
    Analysis of spike data in repeated sites
    Author: Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
import brainbox as bb
import alf.io
from oneibl.one import ONE
from matplotlib import transforms


eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199']
labs = []

print("Will try to compare {} data sets".format(len(eids)))

probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]
bad_eids = ['9a629642-3a9c-42ed-b70a-532db0e86199', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', 'ee40aece-cffd-4edb-a4b6-155f158c666a', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9']
one = ONE()
bad_count = 0

for i, (eid, probe) in enumerate(zip(eids, probes)):
    if eid in bad_eids:
        bad_count += 1
        continue
    print(eid)
    # traj = one.load(eid, dataset_types=['probes.trajectory'])
    # for t in traj[0]:
    #     if t['label'] == probe:
    #         print(t['x'])
    #         print(t['y'])
    #         print(t['z'])
    #         print(t['phi'])
    #         print(t['theta'])
    #spikes, _ = load_spike_sorting(eid, one=one)
    #spikes = spikes[0]

    #spikes, clusters = spikes[probe]['times'], spikes[probe]['clusters']

    depths = np.array(one.load(eid, dataset_types=['clusters.depths']))

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
        if d.shape[0] == np.max(metrics.metrics.cluster_id) + 1: # TODO: just max
            depths = d
            break

    # depths = depths[quality]
    #activity = np.logical_and(np.sum(a_stim.means, axis=1) != 0, np.sum(a_rew.means, axis=1) != 0)
    #depths = depths[activity]
    ax = plt.subplot(3, 5, i + 1 - bad_count)

    ax.hist([depths[quality], depths[~quality]], bins=np.arange(0, 4001, 200), orientation='horizontal', stacked=True, label=['Good', 'mua'])
    #plt.hist(depths, )
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xlim(0, 85)
    ax.set_xticks(np.arange(0, 81, 20))
    ax.set_xticklabels(np.arange(0, 81, 20))
    ax.set_title(one.list(eid, 'subjects'))

    if i + 1 - bad_count == 1:
        ax.set_ylabel('Depth', fontsize=16)
        plt.legend(fontsize=16, frameon=False)
    elif i + 1 - bad_count <= 5:
        ax.get_yaxis().set_ticks([])
    elif (i + 1 - bad_count) % 5 == 1:
        ax.set_ylabel('Depth', fontsize=16)
        ax.get_xaxis().set_ticks([])
    else:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])

plt.show()

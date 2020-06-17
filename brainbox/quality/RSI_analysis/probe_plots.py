"""
    Analysis of spike data in repeated sites
    Author: Sebastian Bruijns
"""

import numpy as np
import matplotlib.pyplot as plt
from oneibl.one import ONE
from brainbox.io.one import load_channel_locations


one = ONE()
eids = ['ee40aece-cffd-4edb-a4b6-155f158c666a', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b', '89f0d6ff-69f4-45bc-b89e-72868abb042a', 'ecb5520d-1358-434c-95ec-93687ecd1396', 'aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', 'a8a8af78-16de-4841-ab07-fde4b5281a03', 'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', '4b00df29-3769-43be-bb40-128b1cba6d35', 'dfd8e7df-dc51-4589-b6ca-7baccfeb94b4', 'db4df448-e449-4a6f-a0e7-288711e7a75a', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c', '0f25376f-2b78-4ddc-8c39-b6cdbe7bf5b9', 'dda5fc59-f09a-4256-9fb5-66c67667a466', '9a629642-3a9c-42ed-b70a-532db0e86199', '03cf52f6-fba6-4743-a42e-dd1ac3072343', 'ee5c418c-e7fa-431d-8796-b2033e000b75', 'e9b57a5a-b06d-476d-ad20-7ec42a16f5f5']
probes = ["probe00", "probe00", "probe00", "probe00", "probe01", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe01", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00", "probe00"]

bad_eids = ['ebe090af-5922-4fcd-8fc6-17b8ba7bad6d', '266a0360-ea0a-4580-8f6a-fe5bad9ed17c']
good_eids = ['aad23144-0e52-4eac-80c5-c4ee2decb198', '572a95d1-39ca-42e1-8424-5c9ffcb2df87', '57fd2325-67f4-4d45-9907-29e77d3043d7', '38d95489-2e82-412a-8c1a-c5377b5f1555', '4153bd83-2168-4bd4-a15c-f7e82f3f73fb', '614e1937-4b24-4ad3-9055-c8253d089919', 'ee5c418c-e7fa-431d-8796-b2033e000b75']

# Code for plots of individual neuropixels
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i, (eid, probe) in enumerate(zip(eids, probes)):
    if eid not in good_eids:
        print('skipped')
        continue
    if eid == '614e1937-4b24-4ad3-9055-c8253d089919':
        probe = 'probe00'
    print(one.list(eid, 'subjects'))

    channels = load_channel_locations(eid, one=one)
    ax.scatter(channels[probe].x, channels[probe].y, channels[probe].z, label=one.list(eid, 'subjects'))

ax.set_xlabel('x in mm')
ax.set_ylabel('y in mm')
ax.set_zlabel('z in mm')
plt.legend()
plt.show()


site_n = 374
xs = np.zeros(site_n * len(good_eids))
ys = np.zeros(site_n * len(good_eids))
zs = np.zeros(site_n * len(good_eids))
names = np.empty(site_n * len(good_eids), dtype='object')
counter = 0

for i, (eid, probe) in enumerate(zip(eids, probes)):
    if eid not in good_eids:
        print('skipped')
        continue
    if eid == '614e1937-4b24-4ad3-9055-c8253d089919':
        probe = 'probe00'
    print(one.list(eid, 'subjects'))

    channels = load_channel_locations(eid, one=one)

    xs[counter * site_n:(counter + 1) * site_n] = channels[probe].x
    ys[counter * site_n:(counter + 1) * site_n] = channels[probe].y
    zs[counter * site_n:(counter + 1) * site_n] = channels[probe].z
    names[counter * site_n:(counter + 1) * site_n] = channels[probe].acronym

    counter += 1


uniq_names = np.unique(names)
names_to_nums = dict(zip(uniq_names, range(len(uniq_names))))
colors = [names_to_nums[x] for x in names]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(xs, ys, zs, c=colors, cmap='hsv')

ax.set_xlabel('x in mm')
ax.set_ylabel('y in mm')
ax.set_zlabel('z in mm')
plt.show()

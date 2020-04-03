'''
Get list of subjects associated to the certification recording project.
TODO not finished: number of subject per lab, number of recording per lab
'''
# Author: Gaelle Chapuis

import numpy
from oneibl.one import ONE
one = ONE(base_url='https://dev.alyx.internationalbrainlab.org')

dataset_types = ['spikes.times',
                 'spikes.clusters']

# eid1, det1 = one.search(project='ibl_certif_neuropix_recording',
#                         dataset_types=dataset_types, details=True)

eid, det = one.search(task_protocol='ephys_certification',
                      dataset_types=dataset_types, details=True)

sub = [p['subject'] for p in det]
# sub_unique = list(set(sub))

lab = [p['lab'] for p in det]
# lab_unique = list(set(lab))

# task = [p['task_protocol'] for p in det]
# task_unique = list(set(task))

# -- How many animals were used per lab
su, ind_su = numpy.unique(sub, return_index=True)
lab_arr = numpy.array(lab)
lu = lab_arr[ind_su]

for i_su in range(0, len(su)):
    # Find how many recordings were made with this animals
    sess_id = numpy.where(numpy.array(sub) == su[i_su])
    # Display
    tr_pl = one.alyx.rest('trajectories', 'list', subject=su[i_su], provenance='Planned')
    tr_tr = one.alyx.rest('trajectories', 'list', subject=su[i_su], provenance='Histology track')
    print(f'Subject: {su[i_su]} - Lab: {lu[i_su]} - N session: {len(sess_id[0])}'
          f' - N planned traces: {len(tr_pl)} - N tracked traces: {len(tr_tr)}')

# TODO -- How many recording sessions were done per lab
# prob_des = one.load(eid, dataset_types=['probes.description'])
# n_probe = len(prob_des[0])

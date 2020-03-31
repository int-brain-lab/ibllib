'''
Get list of subjects associated to the certification recording project.
'''
# Author: Gaelle Chapuis

from oneibl.one import ONE
one = ONE()

dataset_types = ['spikes.times',
                 'spikes.clusters']

# eid, det = one.search(project='ibl_certif_neuropix_recording',
eid, det = one.search(task_protocol='ephys_certification',
                      dataset_types=dataset_types,  details=True)
sub = [p['subject'] for p in det]
sub_unique = list(set(sub))

lab = [p['lab'] for p in det]
lab_unique = list(set(lab))

task = [p['task_protocol'] for p in det]
task_unique = list(set(task))

# How many recording sessions were done per lab

# How many animals were used per lab

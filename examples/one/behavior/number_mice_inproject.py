"""
Quick search through the Alyx database to see all mice/sessions
ever used in training.
"""
# Author: Gaelle Chapuis
import numpy
from oneibl.one import ONE

one = ONE()

eIDs, ses = one.search(task_protocol='world',
                       project='ibl_neuropixel_brainwide_01',
                       details=True)

sub = [p['subject'] for p in ses]
su, ind_su = numpy.unique(sub, return_index=True)

print(f'N subjects: {len(su)} - N sessions: {len(eIDs)}')

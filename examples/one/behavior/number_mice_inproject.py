"""
Quick search through the Alyx database to see all mice/sessions
ever used in training.
"""
# Author: Gaelle Chapuis, Olivier
from oneibl.one import ONE

one = ONE()

ses = one.alyx.rest('sessions', 'list',
                    task_protocol='world',
                    project='ibl_neuropixel_brainwide_01')

subs = one.alyx.rest('subjects', 'list',
                     django=('actions_sessions__task_protocol__icontains,world,'
                             'actions_sessions__project__name,ibl_neuropixel_brainwide_01'))

print(f'N subjects: {len(subs)} - N sessions: {len(ses)}')

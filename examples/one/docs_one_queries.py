"""
List of useful one queries
===========================
"""

from oneibl.one import ONE
one = ONE()

##################################################################################################
# Query session for several subjects
subjects = ['DY_003', 'DY_006']
ses = one.alyx.rest('sessions', 'list', django=f"subject__nickname__in,{subjects}")

##################################################################################################
# Query sessions that have histology available
ses = one.alyx.rest('sessions', 'list', histology=True)
# the generic way
ses = one.alyx.rest('sessions', 'list',
                    django="subject__actions_sessions__procedures__name,Histology")

##################################################################################################
# Query sessions that do not have matlab in the project name
ses = one.alyx.rest('sessions', 'list', django='~project__name__icontains,matlab')

##################################################################################################
# Query sessions that do not contain a given dataset type
ses = one.alyx.rest('sessions', 'list',
                    django='~data_dataset_session_related__dataset_type__name__icontains,wheel')

##################################################################################################
# Query all sessions not labelled as CRITICAL
ses = one.alyx.rest('sessions', 'list', django='qc__lt,50')

##################################################################################################
# Query probe insertions for a given task protocol
ins = one.alyx.rest('insertions', 'list', django='session__task_protocol__icontains,choiceworld')

##################################################################################################
# Query trajectories with channels in given brain region
trajs = one.alyx.rest('trajectories', 'list', django='channels__brain_region__name__icontains,'
                                                     'Entorhinal area medial part dorsal zone '
                                                     'layer 2')

##################################################################################################
# Query spikesorting tasks that have errored in angelaki lab
errored = one.alyx.rest('tasks', 'list', status='Errored', lab='angelakilab',
                        name='SpikeSorting_KS2_Matlab')

##################################################################################################
# Query ephys sessions that have errored tasks
ses = one.alyx.rest('sessions', 'list', task_protocol='ephys', django='tasks__status,40')

##################################################################################################
# Query insertions that have alignment resolved
ins = one.alyx.rest('insertions', 'list', django='json__extended_qc__alignment_resolved,True')

##################################################################################################
# Get names of users who have aligned specified insertion
names = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track',
                      probe_insertion='341ef9bb-25f9-4eeb-8f1d-bdd054b22ba8')[0]['json'].keys()

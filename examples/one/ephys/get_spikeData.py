'''
Get spikes data for all probes in a single session via ONE and brainbox.
# TODO THIS CODE DOES NOT RUN YET
'''
# Author: Gaelle Chapuis

from oneibl.one import ONE
import alf.io as aio
import brainbox as bb

one = ONE()

# To search for specific session, filter using:

dataset_types = ['spikes.times',
                 'spikes.clusters']  # TODO Jai how to download data
'''
task_protocol = '_iblrig_tasks_ephysChoiceWorld6.4.0'
project = 'ibl_neuropixel_brainwide_01'
eIDs = one.search(dataset_types=dataset_types, task_protocol=task_protocol, project=project)
'''
# Example session:
eid = 'da188f2c-553c-4e04-879b-c9ea2d1b9a93'

# Load data
one.load(eid, dataset_types=dataset_types)
session_path = one.path_from_eid(eid)

# Make bunch using brainbox
spks_b = aio.load_object(session_path, 'spikes')
units_b = bb.processing.get_units_bunch(spks_b)

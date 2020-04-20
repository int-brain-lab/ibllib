"""
Test adding of dataset to default.
"""
# Author: Gaelle

from oneibl.one import ONE
from brainbox.io import one as bbone

# get the data from flatiron and the current folder
one = ONE()
eid = one.search(subject='CSHL045', date='2020-02-26', number=1)[0]

# Test added key
spikes, _, _ = bbone.load_ephys_session(eid, one=one, dataset_types=['spikes.depths'])
if 'depths' not in spikes['probe00'].keys():
    raise KeyError

# Test behavior when given random keynames
spikes, _, _ = bbone.load_ephys_session(eid, one=one, dataset_types=['spikes.notexisttest'])
if 'notexisttest' in spikes['probe00'].keys():
    raise KeyError

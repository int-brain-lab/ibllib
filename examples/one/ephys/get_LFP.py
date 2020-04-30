'''
Get LFP data for all probes in a single session via ONE.
'''
# Author: Olivier Winter

from ibllib.io import spikeglx
from oneibl.one import ONE

one = ONE()

# Get a specific session eID
eid = one.search(subject='ZM_2240', date_range='2020-01-22')[0]

# Define and load dataset types of interest
dtypes = ['ephysData.raw.lf', 'ephysData.raw.meta', 'ephysData.raw.ch']
one.load(eid, dataset_types=dtypes, download_only=True)

# Get the files information
session_path = one.path_from_eid(eid)
efiles = [ef for ef in spikeglx.glob_ephys_files(session_path, bin_exists=False) if
          ef.get('lf', None)]

# Read the files and get the data
lfreader = spikeglx.Reader(efiles[0]['lf'])

# Get the first 10000 samples for all traces directly in Volts
dat_volt = lfreader[:10000, :]

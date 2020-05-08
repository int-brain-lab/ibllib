'''
Download raw ephys datasets for all probes in a single session via ONE.
(example written for the LFP, but the download can be done for AP
files similarly by replacing 'lf' with 'ap')
'''
# Author: Olivier, Gaelle

from ibllib.io import spikeglx
from oneibl.one import ONE

# === Option 1 === Download a dataset of interest
one = ONE()

# Get a specific session eID
eid = one.search(subject='ZM_2240', date_range='2020-01-22')[0]

# Define and load dataset types of interest
dtypes = ['ephysData.raw.lf',  # lf : LFP
          'ephysData.raw.meta',
          'ephysData.raw.ch',
          'ephysData.raw.sync']  # Used for synchronisation
one.load(eid, dataset_types=dtypes, download_only=True)

# Get the files information
session_path = one.path_from_eid(eid)
efiles = [ef for ef in spikeglx.glob_ephys_files(session_path, bin_exists=False) if
          ef.get('lf', None)]
efile = efiles[0]['lf']  # Example: access to the first file

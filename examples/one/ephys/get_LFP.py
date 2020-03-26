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

# -- Option 1 -- Get the data directly in Volts
dat_volt = lfreader.read(nsel=slice(0, 1000, None))

# -- Option 2 -- Get the data in samples
dat_samp = lfreader.data[:10000, :]

# Get the conversion factor and check it matches
s2mv = lfreader.channel_conversion_sample2v['lf'][0]  # Convert sample to Volts

if lfreader._raw[55, 5] * s2mv == lfreader[55, 5]:  # TODO OLIVIER CHECK TEST
    ValueError

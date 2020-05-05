'''
Get LFP data for all probes in a single session via ONE.
For client code (such as Matlab spike sorting) to run, it is
necessary to decompress the file. Such decompression is not needed
to access the data otherwise.
'''
# Author: Olivier, Gaelle

from ibllib.io import spikeglx
from oneibl.one import ONE
# from pathlib import Path

# === Option 1 === Download a dataset of interest
one = ONE()

# Get a specific session eID
eid = one.search(subject='ZM_2240', date_range='2020-01-22')[0]

# Define and load dataset types of interest
dtypes = ['ephysData.raw.lf', 'ephysData.raw.meta', 'ephysData.raw.ch']
one.load(eid, dataset_types=dtypes, download_only=True)

# Get the files information
session_path = one.path_from_eid(eid)
efiles_op1 = [ef for ef in spikeglx.glob_ephys_files(session_path, bin_exists=False) if
              ef.get('lf', None)]

# === Option 2 === You can also input a file locally, e.g.
# efile_op2 = Path("/probe00/_spikeglx_ephysData_g0_t0.imec.lf.cbin")
# TODO test run
# NB the .ch file matching the cbin file name must exit

# === Read the files and get the data ===
# Enough to do analysis
sr_op1 = spikeglx.Reader(efiles_op1[0]['lf'])

# Get the first 10000 samples for all traces directly in Volts
dat_volt = sr_op1[:10000, :]

# === Decompress the data ===
# Used by client code, e.g. Matlab for spike sorting
# Give new path output name
sr_op1.decompress_file(keep_original=True,
                       out=session_path.joinpath('efile_lfp_decompressed'))

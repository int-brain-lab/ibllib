"""
Download and decompress raw ephys data
======================================

This code demonstrates how to decompress raw ephys (binary) data - This is necessary for some
client codes (such as Matlab spike sorting KS2 algorithm) to run

(example taken for nidq.cbin file, but also applicable for lf.cbin and ap.cbin files)
"""

# Author: Olivier, Gaelle, Mayo

from ibllib.io import spikeglx
from oneibl.one import ONE

one = ONE()
# Download a dataset of interest
eid = one.search(subject='ZM_2240', date_range='2020-01-22')[0]

dtypes = ['ephysData.raw.nidq',  # change this to ephysData.raw.lf or ephysData.raw.ap for
          'ephysData.raw.ch',
          'ephysData.raw.meta']


_ = one.load(eid, dataset_types=dtypes, download_only=True)

# Get file paths of interest
raw_ephys_path = one.path_from_eid(eid).joinpath('raw_ephys_data')
efile = raw_ephys_path.joinpath('_spikeglx_ephysData_g0_t0.nidq.cbin')

# Read the files and get the data
# Enough to do analysis
sr = spikeglx.Reader(efile)

# Decompress the data
# Used by client code, e.g. Matlab for spike sorting
# Give new path output name
sr.decompress_file(keep_original=True, overwrite=True)  # Keep the original file and overwrite any
# previously decompressed file

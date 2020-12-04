'''
This code demonstrates how to decompress raw ephys
(binary) data - This is necessary for some client codes
(such as Matlab spike sorting KS2 algorithm) to run

(example taken of the LFP, but can be done with AP files similarly)
'''
# Author: Olivier, Gaelle

from ibllib.io import spikeglx
from pathlib import Path
from oneibl.one import ONE

# === Option 1 === Download a dataset of interest
# See download example.

# === Option 2 === Input a file locally, e.g.
# NB the .ch file matching the cbin file name must exit in the same folder
efile = Path("/Users/gaelle/Downloads/FlatIron/examples/ephys/"
             "mainenlab/Subjects/ZM_2240/2020-01-22/001/raw_ephys_data/probe00/"
             "_spikeglx_ephysData_g0_t0.imec0.lf.cbin")

EXAMPLE_OVERWRITE = True  # Put to False when wanting to run the script on your data

# ======== DO NOT EDIT BELOW (used for example testing) ====

if EXAMPLE_OVERWRITE:
    one = ONE()
    # TODO Olivier : Function to download examples folder
    cachepath = Path(one._par.CACHE_DIR)
    efile = cachepath.joinpath('examples', 'ephys',
                               'mainenlab', 'Subjects', 'ZM_2240',
                               '2020-01-22', '001', 'raw_ephys_data', 'probe00',
                               '_spikeglx_ephysData_g0_t0.imec0.lf.cbin')

# === Read the files and get the data ===
# Enough to do analysis
sr = spikeglx.Reader(efile)

# === Decompress the data ===
# Used by client code, e.g. Matlab for spike sorting
# Give new path output name
sr.decompress_file(keep_original=True,  # Keeps the original file
                   overwrite=True)  # Overwrite the out file in case it already exists

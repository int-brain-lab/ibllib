"""
How to decompress an ephys file.

With EXAMPLE_OVERWRITE = True, the script downloads an example dataset and runs
the registration (used for automatic testing of the example).

"""
# Author: Olivier, Gaelle

from pathlib import Path
from ibllib.io import spikeglx

ephys_file = Path("/probe00/_spikeglx_ephysData_g0_t0.imec.lf.cbin")

EXAMPLE_OVERWRITE = True  # Put to False when wanting to run the script on your data

# -- Example (for testing)
if EXAMPLE_OVERWRITE:
    # TODO Olivier : Function to download examples folder
    cachepath = Path(one._par.CACHE_DIR)
    ephys_file = cachepath.joinpath('examples', 'ephys',
                                    'probe00', '_spikeglx_ephysData_g0_t0.imec.lf.cbin')

# -- NB the .ch file matching the cbin file name must exit
assert ephys_file.with_suffix('ch').exists()
sr = spikeglx.Reader(ephys_file)
sr.decompress_file(keep_original=True)

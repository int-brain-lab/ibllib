"""
How to decompress an ephys file.
"""
# Author: Olivier

from pathlib import Path
from ibllib.io import spikeglx

ephys_file = Path("/probe00/_spikeglx_ephysData_g0_t0.imec.lf.cbin")
# NB the .ch file matching the cbin file name must exit
assert ephys_file.with_suffix('ch').exists()
sr = spikeglx.Reader(ephys_file)
sr.decompress_file(keep_original=True)

"""
ALF ephys data files.
"""

import logging
from pathlib import Path
import shutil

# import numpy as np
logger = logging.getLogger(__name__)


def _move_if_possible(path, new_path):
    if not Path(path).exists():
        logger.info("File %s does not exist, skip moving.", path)
        return
    if Path(new_path).exists():
        raise FileExistsError()
    shutil.move(path, new_path)


def _copy_if_possible(path, new_path):
    assert Path(path).exists()
    if Path(new_path).exists():
        raise FileExistsError()
    shutil.copy(path, new_path)


_FILE_RENAMES = (
    ('spike_times.npy', 'spikes.times.npy'),
    ('spike_clusters.npy', 'spikes.clusters.npy'),
    ('amplitudes.npy', 'spikes.amps.npy'),
    ('channel_positions.npy', 'channels.sitePositions.npy'),
    ('templates.npy', 'clusters.templateWaveforms.npy'),
    ('cluster_Amplitude.tsv', 'clusters.amps.tsv'),
    ('channel_map.npy', 'channels.site.npy'),
)


"""

## ALF files to generate

spikes.depths
clusters.meanWaveforms
clusters.peakChannel
clusters.depths
clusters.waveformDuration


## Unused (for now) KS2 files

spike_templates.npy
templates_ind.npy
cluster_ContamPct.tsv
cluster_group.tsv
cluster_KSLabel.tsv


## Further ALF files

lfp.raw
lfp.timestamps
clusters._phy_annotation
clusters.probes
probes.insertion
probes.description
probes.sitePositions
probes.rawFilename
channels.probe
channels.brainLocation
channels.rawRow

"""


def rename_to_alf(dirpath, rawfile=None):
    dirpath = Path(dirpath)

    # File renames.
    for oldfn, newfn in _FILE_RENAMES:
        _move_if_possible(dirpath / oldfn, dirpath / newfn)

    # Rename the raw data file.
    if rawfile:
        rawfile = dirpath / rawfile
        assert rawfile.exists()
        _move_if_possible(rawfile, dirpath / ('ephys.raw' + rawfile.suffix))

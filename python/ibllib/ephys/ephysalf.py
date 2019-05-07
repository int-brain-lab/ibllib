"""
ALF ephys data files.
"""

import logging
from pathlib import Path
import shutil

import numpy as np

from phylib.utils._misc import _read_tsv
from ibllib.misc.misc import logger_config


# import numpy as np
logger = logging.getLogger(__name__)
logger_config()


def _move_if_possible(path, new_path):
    if not Path(path).exists():
        logger.info("File %s does not exist, skip moving.", path)
        return
    if Path(new_path).exists():
        raise FileExistsError()
    # Create the target directory hierarchy if needed.
    new_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(path, new_path)
    logger.info("Moved %s to %s.", path, new_path)


def _copy_if_possible(path, new_path):
    assert Path(path).exists()
    if Path(new_path).exists():
        raise FileExistsError()
    shutil.copy(path, new_path)
    logger.info("Copied %s to %s.", path, new_path)


def _find_file_with_ext(path, ext):
    """Find a file with a given extension in a directory.
    Raises an exception if there are more than one file.
    Return None if there is no such file.
    """
    p = Path(path)
    assert p.is_dir()
    files = list(p.glob('*' + ext))
    if not files:
        return
    elif len(files) == 1:
        return files[0]
    raise RuntimeError(
        "%d files with the extension %s were found in %s.",
        len(files), ext, path
    )


def _load(path):
    path = str(path)
    if path.endswith('.npy'):
        return np.load(path)
    elif path.endswith(('.csv', '.tsv')):
        return _read_tsv(path)[1]  # the function returns a tuple (field, data)
    elif path.endswith('.bin'):
        # TODO: configurable dtype
        return np.fromfile(path, np.int16)


_FILE_RENAMES = (
    ('spike_times.npy', 'spikes.times.npy'),
    ('spike_clusters.npy', 'spikes.clusters.npy'),
    ('amplitudes.npy', 'spikes.amps.npy'),
    ('channel_positions.npy', 'channels.sitePositions.npy'),
    ('templates.npy', 'clusters.templateWaveforms.npy'),
    ('cluster_Amplitude.tsv', 'clusters.amps.tsv'),
    ('channel_map.npy', 'channels.rawRow.npy'),
    ('spike_templates.npy', 'ks2/spikes.clusters.npy'),
    ('cluster_ContamPct.tsv', 'ks2/clusters.ContamPct.tsv'),
    ('cluster_group.tsv', 'ks2/clusters.phyAnnotation.tsv'),
    ('cluster_KSLabel.tsv', 'ks2/clusters.group.tsv'),
)


"""
## Generate

spikes.depths
clusters.meanWaveforms
clusters.peakChannel
clusters.depths
clusters.waveformDuration


## Todo later

clusters.probes
probes.insertion
probes.description
probes.sitePositions
probes.rawFilename
channels.probe
channels.brainLocation
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

    # Rename the LFP file if it exists.
    lf = _find_file_with_ext(dirpath, '.lf.bin')
    if lf:
        _move_if_possible(lf, dirpath / 'lfp.raw.bin')

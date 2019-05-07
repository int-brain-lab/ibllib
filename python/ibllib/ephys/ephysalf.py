"""
ALF ephys data files.
"""

import logging
from pathlib import Path
import shutil

import numpy as np

from phylib.utils._misc import _read_tsv
from phylib.io.array import _spikes_per_cluster, select_spikes, _unique, grouped_mean, _index_of
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


_FILE_RENAMES = [
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
]


"""
## Todo later

clusters.probes
probes.insertion
probes.description
probes.sitePositions
probes.rawFilename
channels.probe
channels.brainLocation
"""


def _rename_to_alf(dir_path, renames=None):
    dir_path = Path(dir_path)

    # File renames.
    for oldfn, newfn in renames or _FILE_RENAMES:
        _move_if_possible(dir_path / oldfn, dir_path / newfn)


def _rename_raw_file(raw_path):
    raw_path = Path(raw_path)
    assert raw_path.exists()
    dir_path = raw_path.parent
    _move_if_possible(raw_path, dir_path / ('ephys.raw' + raw_path.suffix))


def _rename_lfp(dir_path):
    # Rename the LFP file if it exists.
    lf = _find_file_with_ext(dir_path, '.lf.bin')
    if lf:
        _move_if_possible(lf, dir_path / 'lfp.raw.bin')
    return lf.name


class EphysAlfCreator(object):
    def __init__(self, model):
        self.model = model
        self.dir_path = Path(model.dir_path)
        self.spc = _spikes_per_cluster(model.spike_clusters)
        self.renames = _FILE_RENAMES

    def convert(self):
        """Rename the files and generate new ALF files."""
        # Renames.
        _rename_to_alf(self.dir_path)
        _rename_raw_file(self.model.dat_path)
        lfpname = _rename_lfp(self.dir_path)
        self.renames.append((lfpname, 'lfp.raw.bin'))

        # New files.
        self.make_cluster_waveforms()
        self.make_depths()
        self.make_mean_waveforms()

    def rollback(self):
        """Rollback the renames."""
        inverse_renames = [(new, old) for (old, new) in self.renames]
        _rename_to_alf(self.dir_path, renames=inverse_renames)

    def make_cluster_waveforms(self):
        """Return the channel index with the highest template amplitude, for
        every template."""
        p = self.dir_path
        tmp = self.model.sparse_templates.data

        peak_channel_path = p / 'clusters.peakChannel.npy'
        if not peak_channel_path.exists():
            # Create the cluster channels file.
            n_templates, n_samples, n_channels = tmp.shape

            # Compute the peak channels for each template.
            template_peak_channels = np.argmax(tmp.max(axis=1) - tmp.min(axis=1), axis=1)
            assert template_peak_channels.shape == (n_templates,)
            np.save(peak_channel_path, template_peak_channels)
        else:
            template_peak_channels = np.load(peak_channel_path)

        waveform_duration_path = p / 'clusters.waveformDuration.npy'
        if not waveform_duration_path.exists():
            # Compute the peak channel waveform for each template.
            waveforms = tmp[:, :, template_peak_channels]
            durations = waveforms.argmax(axis=1) - waveforms.argmin(axis=1)
            np.save(waveform_duration_path, durations)

    def make_depths(self):
        """Make spikes.depths.npy and clusters.depths.npy."""
        p = self.dir_path

        channel_positions = self.model.channel_positions
        assert channel_positions.ndim == 2

        cluster_channels = np.load(p / 'clusters.peakChannel.npy')
        assert cluster_channels.ndim == 1
        n_clusters = cluster_channels.shape[0]

        spike_clusters = self.model.spike_clusters
        assert spike_clusters.ndim == 1
        n_spikes = spike_clusters.shape[0]
        self.cluster_ids = _unique(self.model.spike_clusters)

        clusters_depths = channel_positions[cluster_channels, 1]
        assert clusters_depths.shape == (n_clusters,)

        spike_clusters_rel = _index_of(spike_clusters, self.cluster_ids)
        assert spike_clusters_rel.max() < clusters_depths.shape[0]
        spikes_depths = clusters_depths[spike_clusters_rel]
        assert spikes_depths.shape == (n_spikes,)

        np.save(p / 'spikes.depths.npy', spikes_depths)
        np.save(p / 'clusters.depths.npy', clusters_depths)

    def make_mean_waveforms(self):
        p = self.dir_path

        spike_ids = select_spikes(
            cluster_ids=self.cluster_ids,
            max_n_spikes_per_cluster=100,
            spikes_per_cluster=lambda clu: self.spc[clu],
            subset='random')
        waveforms = self.model.get_waveforms(spike_ids, np.arange(self.model.n_channels))
        mean_waveforms = grouped_mean(waveforms, self.model.spike_clusters[spike_ids])

        np.save(p / 'clusters.meanWaveforms.npy', mean_waveforms)

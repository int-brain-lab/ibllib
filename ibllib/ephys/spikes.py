from pathlib import Path
import logging
import shutil

import numpy as np
from scipy.interpolate import interp1d

from phylib.io import alf, model, merge
from phylib.io.array import _index_of, _unique

from ibllib.io import spikeglx
from ibllib.io.extractors.ephys_fpga import glob_ephys_files

_logger = logging.getLogger('ibllib')


def merge_probes(ses_path):
    """
    Merge spike sorting output from 2 probes and output in the session ALF folder the combined
    output in IBL format
    :param ses_path: session containig probes to be merged
    :return: None
    """
    ses_path = Path(ses_path)
    out_dir = ses_path.joinpath('alf').joinpath('tmp_merge')
    ephys_files = glob_ephys_files(ses_path)
    subdirs, labels, ephys_files_sorted = zip(*sorted([(ep.ap.parent, ep.label, ep)
                                                       for ep in ephys_files if ep.get('ap')]))

    # if there is only one file, just convert the output to IBL format et basta
    if len(subdirs) == 1:
        ks2_to_alf(subdirs[0], ses_path / 'alf')
        return
    md = spikeglx.read_meta_data(ephys_files[0].get('ap').with_suffix('.meta'))
    sampling_rate = spikeglx._get_fs_from_meta(md)
    mt = merge.probes(subdirs=subdirs, out_dir=out_dir, labels=labels, sampling_rate=sampling_rate)
    # Create the cluster channels file, this should go in the model template as 2 methods
    tmp = mt.sparse_templates.data
    n_templates, n_samples, n_channels = tmp.shape
    template_peak_channels = np.argmax(tmp.max(axis=1) - tmp.min(axis=1), axis=1)
    cluster_probes = mt.channel_probes[template_peak_channels]
    spike_clusters_rel = _index_of(mt.spike_clusters, _unique(mt.spike_clusters))
    spike_probes = cluster_probes[spike_clusters_rel]

    # sync spikes according to the probes
    # how do you make sure they match the files: todo add probe description in the template
    for ind, probe in enumerate(ephys_files_sorted):
        assert(labels[ind] == probe.label)  # paranoid, make sure they are sorted
        if not probe.get('ap'):
            continue
        sync_file = probe.ap.parent / probe.ap.name.replace('ap.bin', 'sync.npy')
        if not sync_file.exists():
            error_msg = f'No synchronisation file for {sync_file}'
            _logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        sync_points = np.load(sync_file)
        fcn = interp1d(sync_points[:, 0] * sampling_rate,
                       sync_points[:, 1], fill_value='extrapolate')
        mt.spike_times[spike_probes == ind] = fcn(mt.spike_times[spike_probes == ind])

    # And convert to ALF
    ac = alf.EphysAlfCreator(mt)
    ac.convert(ses_path / 'alf')

    # remove the temporary directory
    shutil.rmtree(out_dir)


def ks2_to_alf(ks_path, out_path):
    """
    Convert Kilosort 2 output to ALF dataset for single probe data
    :param ks_path:
    :param out_path:
    :return:
    """
    # Todo get sampling rate properly from meta data file
    # efiles = glob_ephys_files(ks_path)
    m = model.TemplateModel(dir_path=ks_path,
                            dat_path=[],
                            sample_rate=30000,
                            n_channels_dat=385)
    ac = alf.EphysAlfCreator(m)
    ac.convert(out_path)

from pathlib import Path
import logging
import shutil

import numpy as np
from scipy.interpolate import interp1d

from phylib.io import alf, model

from ibllib.io import spikeglx
from ibllib.io.extractors.ephys_fpga import glob_ephys_files

_logger = logging.getLogger('ibllib')


def sync_spike_sortings(ses_path):
    """
    Merge spike sorting output from 2 probes and output in the session ALF folder the combined
    output in IBL format
    :param ses_path: session containing probes to be merged
    :return: None
    """
    def _sr(ap_file):
        md = spikeglx.read_meta_data(ap_file.with_suffix('.meta'))
        return spikeglx._get_fs_from_meta(md)

    ses_path = Path(ses_path)
    ephys_files = glob_ephys_files(ses_path)
    subdirs, labels, efiles_sorted, srates = zip(
        *sorted([(ep.ap.parent, ep.label, ep, _sr(ep.ap)) for ep in ephys_files if ep.get('ap')]))

    # if there is only one file, just convert the output to IBL format et basta
    _logger.info('converting  spike-sorting outputs to ALF')
    for subdir, label, ef, sr in zip(subdirs, labels, efiles_sorted, srates):
        ks2alf_path = subdir / 'ks2_alf'
        if ks2alf_path.exists():
            shutil.rmtree(ks2alf_path, ignore_errors=True)
        ks2_to_alf(subdir, ses_path / 'alf', label=label, sr=sr, force=True)
        # synchronize the spike sorted times
        sync_file = ef.ap.parent.joinpath(ef.ap.name.replace('.ap.', '.sync.')).with_suffix('.npy')
        if not sync_file.exists():
            error_msg = f'No synchronisation file for {sync_file}'
            _logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        sync_points = np.load(sync_file)
        fcn = interp1d(sync_points[:, 0] * sr,
                       sync_points[:, 1], fill_value='extrapolate')
        # patch the files manually
        st_file = ses_path.joinpath('alf', f'spikes.times.{label}.npy')
        interp_times = fcn(np.load(st_file))
        np.save(st_file, interp_times)


def ks2_to_alf(ks_path, out_path, sr=30000, nchannels=385, label=None, force=True):
    """
    Convert Kilosort 2 output to ALF dataset for single probe data
    :param ks_path:
    :param out_path:
    :return:
    """
    m = model.TemplateModel(dir_path=ks_path,
                            dat_path=[],
                            sample_rate=sr,
                            n_channels_dat=nchannels)
    ac = alf.EphysAlfCreator(m)
    ac.convert(out_path, label=label, force=force)

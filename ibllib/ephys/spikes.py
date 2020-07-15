from pathlib import Path
import logging
import json
import shutil

import numpy as np

from phylib.io import alf
from ibllib.ephys.sync_probes import apply_sync
import ibllib.ephys.ephysqc as ephysqc
from ibllib.io import spikeglx, raw_data_loaders

_logger = logging.getLogger('ibllib')


def probes_description(ses_path, bin_exists=True):
    """
    Aggregate probes information into ALF files
    Input:
        raw_ephys_data/probeXX/
    Output:
        alf/probes.description.npy
        alf/probes.trajecory.npy
    """

    ses_path = Path(ses_path)
    ephys_files = spikeglx.glob_ephys_files(ses_path, bin_exists=bin_exists)
    subdirs, labels, efiles_sorted = zip(
        *sorted([(ep.ap.parent, ep.label, ep) for ep in ephys_files if ep.get('ap')]))

    """Ouputs the probes description file"""
    probe_description = []
    for label, ef in zip(labels, efiles_sorted):
        md = spikeglx.read_meta_data(ef.ap.with_suffix('.meta'))
        probe_description.append({'label': label,
                                  'model': md.neuropixelVersion,
                                  'serial': int(md.serial),
                                  'raw_file_name': md.fileName,
                                  })
    alf_path = ses_path.joinpath('alf')
    alf_path.mkdir(exist_ok=True, parents=True)
    probe_description_file = alf_path.joinpath('probes.description.json')
    with open(probe_description_file, 'w+') as fid:
        fid.write(json.dumps(probe_description))

    """Ouputs the probes trajectory file"""
    bpod_meta = raw_data_loaders.load_settings(ses_path)
    if not bpod_meta.get('PROBE_DATA'):
        _logger.error('No probe information in settings JSON. Skipping probes.trajectory')
        return

    def prb2alf(prb, label):
        return {'label': label, 'x': prb['X'], 'y': prb['Y'], 'z': prb['Z'], 'phi': prb['A'],
                'theta': prb['P'], 'depth': prb['D'], 'beta': prb['T']}

    # the labels may not match, in which case throw a warning and work in alphabetical order
    if labels != ('probe00', 'probe01'):
        _logger.warning("Probe names do not match the json settings files. Will match coordinates"
                        " per alphabetical order !")
        _ = [_logger.warning(f"  probe0{i} ----------  {lab} ") for i, lab in enumerate(labels)]
    trajs = []
    keys = sorted(bpod_meta['PROBE_DATA'].keys())
    for i, k in enumerate(keys):
        if i >= len(labels):
            break
        trajs.append(prb2alf(bpod_meta['PROBE_DATA'][f'probe0{i}'], labels[i]))
    probe_trajectory_file = alf_path.joinpath('probes.trajectory.json')
    with open(probe_trajectory_file, 'w+') as fid:
        fid.write(json.dumps(trajs))
    return [probe_trajectory_file, probe_description_file]


def sync_spike_sortings(ses_path):
    """
    Converts the KS2 outputs for each probe in ALF format. Creates:
    alf/probeXX/spikes.*
    alf/probeXX/clusters.*
    alf/probeXX/templates.*
    :param ses_path: session containing probes to be merged
    :return: None
    """
    def _sr(ap_file):
        # gets sampling rate from data
        md = spikeglx.read_meta_data(ap_file.with_suffix('.meta'))
        return spikeglx._get_fs_from_meta(md)

    def _sample2v(ap_file):
        md = spikeglx.read_meta_data(ap_file.with_suffix('.meta'))
        s2v = spikeglx._conversion_sample2v_from_meta(md)
        return s2v['ap'][0]

    ses_path = Path(ses_path)
    ephys_files = spikeglx.glob_ephys_files(ses_path)
    subdirs, labels, efiles_sorted, srates = zip(
        *sorted([(ep.ap.parent, ep.label, ep, _sr(ep.ap)) for ep in ephys_files if ep.get('ap')]))

    _logger.info('converting  spike-sorting outputs to ALF')
    out_files = []
    for subdir, label, ef, sr in zip(subdirs, labels, efiles_sorted, srates):
        if not subdir.joinpath('spike_times.npy').exists():
            _logger.warning(f"No KS2 spike sorting found in {subdir}, skipping probe !")
            continue
        probe_out_path = ses_path.joinpath('alf', label)
        probe_out_path.mkdir(parents=True, exist_ok=True)
        # handles the probes synchronization
        sync_file = ef.ap.parent.joinpath(ef.ap.name.replace('.ap.', '.sync.')
                                          ).with_suffix('.npy')
        if not sync_file.exists():
            """
            if there is no sync file it means something went wrong. Outputs the spike sorting
            in time according the the probe by following ALF convention on the times objects
            """
            error_msg = f'No synchronisation file for {label}: {sync_file}. The spike-' \
                        f'sorting is not synchronized and data not uploaded on Flat-Iron'
            _logger.error(error_msg)
            # remove the alf folder if the sync failed
            shutil.rmtree(probe_out_path)
            continue
        # converts the folder to ALF
        ks2_to_alf(subdir, probe_out_path, ampfactor=_sample2v(ef.ap), label=None, force=True)
        # patch the spikes.times files manually
        st_file = ses_path.joinpath(probe_out_path, 'spikes.times.npy')
        spike_samples = np.load(ses_path.joinpath(probe_out_path, 'spikes.samples.npy'))
        interp_times = apply_sync(sync_file, spike_samples / sr, forward=True)
        np.save(st_file, interp_times)
        # get the list of output files
        out_files.extend([f for f in ses_path.joinpath(probe_out_path).glob("*.*") if
                          f.name.startswith(('channels.', 'clusters.', 'spikes.', 'templates.',
                                             '_kilosort_', '_phy_spikes_subset'))])
    return out_files


def ks2_to_alf(ks_path, out_path, ampfactor=1, label=None, force=True):
    """
    Convert Kilosort 2 output to ALF dataset for single probe data
    :param ks_path:
    :param out_path:
    :return:

    TODO change/remove `ampfactor` after seeing how amplitudes are calculated by ks2
    """
    m = ephysqc.phy_model_from_ks2_path(ks_path)
    ephysqc.unit_metrics_ks2(ks_path, m, save=True)
    ac = alf.EphysAlfCreator(m)
    ac.convert(out_path, label=label, force=force, ampfactor=ampfactor)

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


def probes_description(ses_path, one=None, bin_exists=True):
    """
    Aggregate probes information into ALF files
    Register alyx probe insertions and Micro-manipulator trajectories
    Input:
        raw_ephys_data/probeXX/
    Output:
        alf/probes.description.npy
        alf/probes.trajectory.npy
    """

    eid = one.eid_from_path(ses_path)
    ses_path = Path(ses_path)
    ephys_files = spikeglx.glob_ephys_files(ses_path, bin_exists=bin_exists)
    subdirs, labels, efiles_sorted = zip(
        *sorted([(ep.ap.parent, ep.label, ep) for ep in ephys_files if ep.get('ap')]))

    # Ouputs the probes description file
    probe_description = []
    alyx_insertions = []
    for label, ef in zip(labels, efiles_sorted):
        md = spikeglx.read_meta_data(ef.ap.with_suffix('.meta'))
        probe_description.append({'label': label,
                                  'model': md.neuropixelVersion,
                                  'serial': int(md.serial),
                                  'raw_file_name': md.fileName,
                                  })
        # create or update alyx probe insertions
        alyx_insertion = {'session': eid, 'model': md.neuropixelVersion,
                          'serial': md.serial, 'name': label}
        pi = one.alyx.rest('insertions', 'list', session=eid, name=label)
        if len(pi) == 0:
            alyx_insertions.append(one.alyx.rest('insertions', 'create', data=alyx_insertion))
        else:
            alyx_insertions.append(
                one.alyx.rest('insertions', 'partial_update', data=alyx_insertion, id=pi[0]['id']))

    alf_path = ses_path.joinpath('alf')
    alf_path.mkdir(exist_ok=True, parents=True)
    probe_description_file = alf_path.joinpath('probes.description.json')
    with open(probe_description_file, 'w+') as fid:
        fid.write(json.dumps(probe_description))

    # Ouputs the probes trajectory file
    bpod_meta = raw_data_loaders.load_settings(ses_path)
    if not bpod_meta.get('PROBE_DATA'):
        _logger.error('No probe information in settings JSON. Skipping probes.trajectory')
        return

    def prb2alf(prb, label):
        return {'label': label, 'x': prb['X'], 'y': prb['Y'], 'z': prb['Z'], 'phi': prb['A'],
                'theta': prb['P'], 'depth': prb['D'], 'beta': prb['T']}

    def prb2alyx(prb, probe_insertion):
        return {'probe_insertion': probe_insertion, 'x': prb['X'], 'y': prb['Y'], 'z': prb['Z'],
                'phi': prb['A'], 'theta': prb['P'], 'depth': prb['D'], 'roll': prb['T'],
                'provenance': 'Micro-manipulator', 'coordinate_system': 'Needles-Allen'}

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
        pdict = bpod_meta['PROBE_DATA'][f'probe0{i}']
        trajs.append(prb2alf(pdict, labels[i]))
        pid = next((ai['id'] for ai in alyx_insertions if ai['name'] == k), None)
        if pid:
            # here we don't update the micro-manipulator coordinates if the trajectory already
            # exists as it may have been entered manually through admin interface
            trj = one.alyx.rest('trajectories', 'list', probe_insertion=pid,
                                provenance='Micro-manipulator')
            if len(trj) == 0:
                one.alyx.rest('trajectories', 'create', data=prb2alyx(pdict, pid))

    probe_trajectory_file = alf_path.joinpath('probes.trajectory.json')
    with open(probe_trajectory_file, 'w+') as fid:
        fid.write(json.dumps(trajs))
    return [probe_trajectory_file, probe_description_file]


def sync_spike_sortings(session_path):
    """
    Converts the KS2 outputs for each probe in ALF format. Creates:
    alf/probeXX/spikes.*
    alf/probeXX/clusters.*
    alf/probeXX/templates.*
    :param session_path: session containing probes to be merged
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

    session_path = Path(session_path)
    ephys_files = spikeglx.glob_ephys_files(session_path)
    bin_data_dirs, labels, efiles_sorted, srates = zip(
        *sorted([(ep.ap.parent, ep.label, ep, _sr(ep.ap)) for ep in ephys_files if ep.get('ap')]))

    _logger.info('converting  spike-sorting outputs to ALF')
    out_files = []
    # label: the probe name: "probe00"
    # bin_data_dir: the directory with raw ephys: "{session_path}/raw_ephys_data/probe00"
    # ks2_dir: spike sorted results: "{session_path}/spike_sorters/ks2_matlab/probe00"
    for bin_data_dir, label, ef, sr in zip(bin_data_dirs, labels, efiles_sorted, srates):
        ks2_dir = session_path.joinpath('spike_sorters', 'ks2_matlab', label)
        if not ks2_dir.joinpath('spike_times.npy').exists():
            ks2_dir = bin_data_dir
        if not ks2_dir.joinpath('spike_times.npy').exists():
            _logger.warning(f"No KS2 spike sorting found in {bin_data_dir}, skipping probe !")
            continue
        probe_out_path = session_path.joinpath('alf', label)
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
        ks2_to_alf(ks2_dir, bin_data_dir, probe_out_path, bin_file=ef.ap,
                   ampfactor=_sample2v(ef.ap), label=None, force=True)
        # patch the spikes.times files manually
        st_file = session_path.joinpath(probe_out_path, 'spikes.times.npy')
        spike_samples = np.load(session_path.joinpath(probe_out_path, 'spikes.samples.npy'))
        interp_times = apply_sync(sync_file, spike_samples / sr, forward=True)
        np.save(st_file, interp_times)
        # get the list of output files
        out_files.extend([f for f in session_path.joinpath(probe_out_path).glob("*.*") if
                          f.name.startswith(('channels.', 'clusters.', 'spikes.', 'templates.',
                                             '_kilosort_', '_phy_spikes_subset'))])
    return out_files


def ks2_to_alf(ks_path, bin_path, out_path, bin_file=None, ampfactor=1, label=None, force=True):
    """
    Convert Kilosort 2 output to ALF dataset for single probe data
    :param ks_path:
    :param bin_path: path of raw data
    :param out_path:
    :return:
    """
    m = ephysqc.phy_model_from_ks2_path(ks2_path=ks_path, bin_path=bin_path, bin_file=bin_file)
    ephysqc.unit_metrics_ks2(ks_path, m, save=True)
    ac = alf.EphysAlfCreator(m)
    ac.convert(out_path, label=label, force=force, ampfactor=ampfactor)

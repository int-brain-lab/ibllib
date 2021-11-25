from pathlib import Path
import logging
import json
import shutil
import tarfile

import numpy as np
from one.alf.files import get_session_path

from iblutil.util import Bunch
import phylib.io.alf
from ibllib.ephys.sync_probes import apply_sync
import ibllib.ephys.ephysqc as ephysqc
from ibllib.ephys import sync_probes
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

    eid = one.path2eid(ses_path, query_type='remote')
    ses_path = Path(ses_path)
    ephys_files = spikeglx.glob_ephys_files(ses_path, ext='meta')
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
            qc_dict = {'qc': 'NOT_SET', 'extended_qc': {}}
            alyx_insertion.update({'json': qc_dict})
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
        return []

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


def sync_spike_sorting(ap_file, out_path):
    """
    Synchronizes the spike.times using the previously computed sync files
    :param ap_file: raw binary data file for the probe insertion
    :param out_path: probe output path (usually {session_path}/alf/{probe_label})
    """

    def _sr(ap_file):
        # gets sampling rate from data
        md = spikeglx.read_meta_data(ap_file.with_suffix('.meta'))
        return spikeglx._get_fs_from_meta(md)

    out_files = []
    label = ap_file.parts[-1]  # now the bin file is always in a folder bearing the name of probe
    sync_file = ap_file.parent.joinpath(
        ap_file.name.replace('.ap.', '.sync.')).with_suffix('.npy')
    # try to get probe sync if it doesn't exist
    if not sync_file.exists():
        _, sync_files = sync_probes.sync(get_session_path(ap_file))
        out_files.extend(sync_files)
    # if it still not there, full blown error
    if not sync_file.exists():
        # if there is no sync file it means something went wrong. Outputs the spike sorting
        # in time according the the probe by following ALF convention on the times objects
        error_msg = f'No synchronisation file for {label}: {sync_file}. The spike-' \
                    f'sorting is not synchronized and data not uploaded on Flat-Iron'
        _logger.error(error_msg)
        # remove the alf folder if the sync failed
        shutil.rmtree(out_path)
        return None, 1
    # patch the spikes.times files manually
    st_file = out_path.joinpath('spikes.times.npy')
    spike_samples = np.load(out_path.joinpath('spikes.samples.npy'))
    interp_times = apply_sync(sync_file, spike_samples / _sr(ap_file), forward=True)
    np.save(st_file, interp_times)
    # get the list of output files
    out_files.extend([f for f in out_path.glob("*.*") if
                      f.name.startswith(('channels.', 'drift', 'clusters.', 'spikes.', 'templates.',
                                         '_kilosort_', '_phy_spikes_subset', '_ibl_log.info'))])
    # the QC files computed during spike sorting stay within the raw ephys data folder
    out_files.extend(list(ap_file.parent.glob('_iblqc_*AP.*.npy')))
    return out_files, 0


def ks2_to_alf(ks_path, bin_path, out_path, bin_file=None, ampfactor=1, label=None, force=True):
    """
    Convert Kilosort 2 output to ALF dataset for single probe data
    :param ks_path:
    :param bin_path: path of raw data
    :param out_path:
    :return:
    """
    m = ephysqc.phy_model_from_ks2_path(ks2_path=ks_path, bin_path=bin_path, bin_file=bin_file)
    ac = phylib.io.alf.EphysAlfCreator(m)
    ac.convert(out_path, label=label, force=force, ampfactor=ampfactor)


def ks2_to_tar(ks_path, out_path, force=False):
    """
    Compress output from kilosort 2 into tar file in order to register to flatiron and move to
    spikesorters/ks2_matlab/probexx path. Output file to register

    :param ks_path: path to kilosort output
    :param out_path: path to keep the
    :return
    path to tar ks output

    To extract files from the tar file can use this code
    Example:
        save_path = Path('folder you want to extract to')
        with tarfile.open('_kilosort_output.tar', 'r') as tar_dir:
            tar_dir.extractall(path=save_path)

    """
    ks2_output = ['amplitudes.npy',
                  'channel_map.npy',
                  'channel_positions.npy',
                  'cluster_Amplitude.tsv',
                  'cluster_ContamPct.tsv',
                  'cluster_group.tsv',
                  'cluster_KSLabel.tsv',
                  'params.py',
                  'pc_feature_ind.npy',
                  'pc_features.npy',
                  'similar_templates.npy',
                  'spike_clusters.npy',
                  'spike_sorting_ks2.log',
                  'spike_templates.npy',
                  'spike_times.npy',
                  'template_feature_ind.npy',
                  'template_features.npy',
                  'templates.npy',
                  'templates_ind.npy',
                  'whitening_mat.npy',
                  'whitening_mat_inv.npy']

    out_file = Path(out_path).joinpath('_kilosort_raw.output.tar')
    if out_file.exists() and not force:
        _logger.info(f"Already converted ks2 to tar: for {ks_path}, skipping.")
        return [out_file]

    with tarfile.open(out_file, 'w') as tar_dir:
        for file in Path(ks_path).iterdir():
            if file.name in ks2_output:
                tar_dir.add(file, file.name)

    return [out_file]


def detection(data, fs, h, detect_threshold=-4, time_tol=.002, distance_threshold_um=70):
    """
    Detects and de-duplicates negative voltage spikes based on voltage thresholding.
    The de-duplication step locks in maximum amplitude events. To account for collisions the amplitude
    is assumed to be decaying from the peak. If this is a multipeak event, each is labeled as a spike.

    :param data: 2D numpy array nsamples x nchannels
    :param fs: sampling frequency (Hz)
    :param h: dictionary with neuropixel geometry header: see. ibllib.ephys.neuropixel.trace_header
    :param detect_threshold: negative value below which the voltage is considered to be a spike
    :param time_tol: time in seconds for which samples before and after are assumed to be part of the spike
    :param distance_threshold_um: distance for which exceeding threshold values are assumed to part of the same spike
    :return: spikes dictionary of vectors with keys "time", "trace", "amp" and "ispike"
    """
    multipeak = False
    time_bracket = np.array([-1, 1]) * time_tol
    inds, indtr = np.where(data < detect_threshold)
    picks = Bunch(time=inds / fs, trace=indtr, amp=data[inds, indtr], ispike=np.zeros(inds.size))
    amp_order = np.argsort(picks.amp)

    hxy = h['x'] + 1j * h['y']

    spike_id = 1
    while np.any(picks.ispike == 0):
        # find the first unassigned spike with the highest amplitude
        iamp = np.where(picks.ispike[amp_order] == 0)[0][0]
        imax = amp_order[iamp]
        # look only within the time range
        itlims = np.searchsorted(picks.time, picks.time[imax] + time_bracket)
        itlims = np.arange(itlims[0], itlims[1])

        offset = np.abs(hxy[picks.trace[itlims]] - hxy[picks.trace[imax]])
        iit = np.where(offset < distance_threshold_um)[0]

        picks.ispike[itlims[iit]] = -1
        picks.ispike[imax] = spike_id
        # handles collision with a simple amplitude decay model: if amplitude doesn't decay
        # as a function of offset, then it's a collision and another spike is set
        if multipeak:  # noqa
            iii = np.lexsort((picks.amp[itlims[iit]], offset[iit]))
            sorted_amps_db = 20 * np.log10(np.abs(picks.amp[itlims[iit][iii]]))
            idetect = np.r_[0, np.where(np.diff(sorted_amps_db) > 12)[0] + 1]
            picks.ispike[itlims[iit[iii[idetect]]]] = np.arange(idetect.size) + spike_id
            spike_id += idetect.size
        else:
            spike_id += 1

    detects = Bunch({k: picks[k][picks.ispike > 0] for k in picks})
    return detects

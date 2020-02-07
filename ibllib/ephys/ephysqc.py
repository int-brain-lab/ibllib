"""
Quality control of raw Neuropixel electrophysiology data.
"""
from pathlib import Path
import logging

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d

import alf.io
from brainbox.core import Bunch
from brainbox.processing import bincount2D
from ibllib.ephys import sync_probes
from ibllib.io import spikeglx
import ibllib.dsp as dsp
import ibllib.io.extractors.ephys_fpga as fpga
from ibllib.misc import print_progress, log2session_static
from phylib.io import model


_logger = logging.getLogger('ibllib')

RMS_WIN_LENGTH_SECS = 3
WELCH_WIN_LENGTH_SAMPLES = 1024

METRICS_PARAMS = {
    'presence_bin_length_secs': 20,
    "isi_threshold": 0.0015,
    "min_isi": 0.000166,
    "num_channels_to_compare": 13,
    "max_spikes_for_unit": 500,
    "max_spikes_for_nn": 10000,
    "n_neighbors": 4,
    'n_silhouette': 10000,
    "quality_metrics_output_file": "metrics.csv",
    "drift_metrics_interval_s": 51,
    "drift_metrics_min_spikes_per_interval": 10
}


def rmsmap(fbin):
    """
    Computes RMS map in time domain and spectra for each channel of Neuropixel probe

    :param fbin: binary file in spike glx format (will look for attached metatdata)
    :type fbin: str or pathlib.Path
    :return: a dictionary with amplitudes in channeltime space, channelfrequency space, time
     and frequency scales
    """
    if not isinstance(fbin, spikeglx.Reader):
        sglx = spikeglx.Reader(fbin)
    rms_win_length_samples = 2 ** np.ceil(np.log2(sglx.fs * RMS_WIN_LENGTH_SECS))
    # the window generator will generates window indices
    wingen = dsp.WindowGenerator(ns=sglx.ns, nswin=rms_win_length_samples, overlap=0)
    # pre-allocate output dictionary of numpy arrays
    win = {'TRMS': np.zeros((wingen.nwin, sglx.nc)),
           'nsamples': np.zeros((wingen.nwin,)),
           'fscale': dsp.fscale(WELCH_WIN_LENGTH_SAMPLES, 1 / sglx.fs, one_sided=True),
           'tscale': wingen.tscale(fs=sglx.fs)}
    win['spectral_density'] = np.zeros((len(win['fscale']), sglx.nc))
    # loop through the whole session
    for first, last in wingen.firstlast:
        D = sglx.read_samples(first_sample=first, last_sample=last)[0].transpose()
        # remove low frequency noise below 1 Hz
        D = dsp.hp(D, 1 / sglx.fs, [0, 1])
        iw = wingen.iw
        win['TRMS'][iw, :] = dsp.rms(D)
        win['nsamples'][iw] = D.shape[1]
        # the last window may be smaller than what is needed for welch
        if last - first < WELCH_WIN_LENGTH_SAMPLES:
            continue
        # compute a smoothed spectrum using welch method
        _, w = signal.welch(D, fs=sglx.fs, window='hanning', nperseg=WELCH_WIN_LENGTH_SAMPLES,
                            detrend='constant', return_onesided=True, scaling='density', axis=-1)
        win['spectral_density'] += w.T
        # print at least every 20 windows
        if (iw % min(20, max(int(np.floor(wingen.nwin / 75)), 1))) == 0:
            print_progress(iw, wingen.nwin)
    return win


def extract_rmsmap(fbin, out_folder=None, force=False):
    """
    Wrapper for rmsmap that outputs _ibl_ephysRmsMap and _ibl_ephysSpectra ALF files

    :param fbin: binary file in spike glx format (will look for attached metatdata)
    :param out_folder: folder in which to store output ALF files. Default uses the folder in which
     the `fbin` file lives.
    :param force: do not re-extract if all ALF files already exist
    :param label: string or list of strings that will be appended to the filename before extension
    :return: None
    """
    _logger.info(str(fbin))
    sglx = spikeglx.Reader(fbin)
    # check if output ALF files exist already:
    if out_folder is None:
        out_folder = Path(fbin).parent
    else:
        out_folder = Path(out_folder)
    alf_object_time = f'_iblqc_ephysTimeRms{sglx.type.upper()}'
    alf_object_freq = f'_iblqc_ephysSpectralDensity{sglx.type.upper()}'
    if alf.io.exists(out_folder, alf_object_time) and \
            alf.io.exists(out_folder, alf_object_freq) and not force:
        _logger.warning(f'{fbin.name} QC already exists, skipping. Use force option to override')
        return
    # crunch numbers
    rms = rmsmap(fbin)
    # output ALF files, single precision with the optional label as suffix before extension
    if not out_folder.exists():
        out_folder.mkdir()
    tdict = {'rms': rms['TRMS'].astype(np.single), 'timestamps': rms['tscale'].astype(np.single)}
    fdict = {'power': rms['spectral_density'].astype(np.single),
             'freqs': rms['fscale'].astype(np.single)}
    out_time = alf.io.save_object_npy(out_folder, object=alf_object_time, dico=tdict)
    out_freq = alf.io.save_object_npy(out_folder, object=alf_object_freq, dico=fdict)
    return out_time + out_freq


@log2session_static('ephys')
def raw_qc_session(session_path, dry=False, force=False):
    """
    Wrapper that exectutes QC from a session folder and outputs the results whithin the same folder
    as the original raw data.
    :param session_path: path of the session (Subject/yyyy-mm-dd/number
    :param dry: bool (False) Dry run if True
    :param force: bool (False) Force means overwriting an existing QC file
    :return: None
    """
    efiles = spikeglx.glob_ephys_files(session_path)
    for efile in efiles:
        if efile.get('ap') and efile.ap.exists():
            print(efile.get('ap'))
            if not dry:
                extract_rmsmap(efile.ap, out_folder=None, force=force)
        if efile.get('lf') and efile.lf.exists():
            print(efile.get('lf'))
            if not dry:
                extract_rmsmap(efile.lf, out_folder=None, force=force)


def validate_ttl_test(ses_path, display=False):
    """
    For a mock session on the Ephys Choice world task, check the sync channels for all
    device properly connected and perform a synchronization if dual probes to check that
    all channels are recorded properly
    :param ses_path: session path
    :param display: show the probe synchronization plot if several probes
    :return: True if tests pass, errors otherwise
    """

    def _single_test(assertion, str_ok, str_ko):
        if assertion:
            _logger.info(str_ok)
            return True
        else:
            _logger.error(str_ko)
            return False

    EXPECTED_RATES_HZ = {'left_camera': 60, 'right_camera': 150, 'body_camera': 30}
    SYNC_RATE_HZ = 1
    MIN_TRIALS_NB = 6

    ok = True
    ses_path = Path(ses_path)
    if not ses_path.exists():
        return False
    rawsync, sync_map = fpga._get_main_probe_sync(ses_path)
    last_time = rawsync['times'][-1]

    # get upgoing fronts for each
    sync = Bunch({})
    for k in sync_map:
        fronts = fpga._get_sync_fronts(rawsync, sync_map[k])
        sync[k] = fronts['times'][fronts['polarities'] == 1]
    wheel = fpga.extract_wheel_sync(rawsync, chmap=sync_map, save=False)

    frame_rates = {'right_camera': np.round(1 / np.median(np.diff(sync.right_camera))),
                   'left_camera': np.round(1 / np.median(np.diff(sync.left_camera))),
                   'body_camera': np.round(1 / np.median(np.diff(sync.body_camera)))}

    # check the camera frame rates
    for lab in frame_rates:
        expect = EXPECTED_RATES_HZ[lab]
        ok &= _single_test(assertion=abs((1 - frame_rates[lab] / expect)) < 0.1,
                           str_ok=f'PASS: {lab} frame rate: {frame_rates[lab]} = {expect} Hz',
                           str_ko=f'FAILED: {lab} frame rate: {frame_rates[lab]} != {expect} Hz')

    # check that the wheel has a minimum rate of activity on both channels
    re_test = abs(1 - sync.rotary_encoder_1.size / sync.rotary_encoder_0.size) < 0.1
    re_test &= len(wheel['re_pos']) / last_time > 5
    ok &= _single_test(assertion=re_test,
                       str_ok="PASS: Rotary encoder", str_ko="FAILED: Rotary encoder")
    # check that the frame 2 ttls has a minimum rate of activity
    ok &= _single_test(assertion=len(sync.frame2ttl) / last_time > 0.2,
                       str_ok="PASS: Frame2TTL", str_ko="FAILED: Frame2TTL")
    # the audio has to have at least one event per trial
    ok &= _single_test(assertion=len(sync.bpod) > len(sync.audio) > MIN_TRIALS_NB,
                       str_ok="PASS: audio", str_ko="FAILED: audio")
    # the bpod has to have at least twice the amount of min trial pulses
    ok &= _single_test(assertion=len(sync.bpod) > MIN_TRIALS_NB * 2,
                       str_ok="PASS: Bpod", str_ko="FAILED: Bpod")
    try:
        # note: tried to depend as little as possible on the extraction code but for the valve...
        behaviour = fpga.extract_behaviour_sync(rawsync, save=False, chmap=sync_map)
        res = behaviour.valve_open.size > 1
    except AssertionError:
        res = False
    # check that the reward valve is actionned at least once
    ok &= _single_test(assertion=res,
                       str_ok="PASS: Valve open", str_ko="FAILED: Valve open not detected")
    _logger.info('ALL CHECKS PASSED !')

    # the imec sync is for 3B Probes only
    if sync.get('imec_sync') is not None:
        ok &= _single_test(assertion=np.all(1 - SYNC_RATE_HZ * np.diff(sync.imec_sync) < 0.1),
                           str_ok="PASS: imec sync", str_ko="FAILED: imec sync")

    # second step is to test that we can make the sync. Assertions are whithin the synch code
    if sync.get('imec_sync') is not None:
        sync_result = sync_probes.version3B(ses_path, display=display)
    else:
        sync_result = sync_probes.version3A(ses_path, display=display)

    ok &= _single_test(assertion=sync_result, str_ok="PASS: synchronisation",
                       str_ko="FAILED: probe synchronizations threshold exceeded")

    if not ok:
        raise ValueError('FAILED TTL test')
    return ok


def _spike_sorting_metrics_ks2(ks2_path, save=True):
    """
    Given a path containing kilosort 2 output, compute quality metrics and optionally save them
    to a clusters_metric.csv file
    :param ks2_path:
    :param save
    :return:
    """

    m = phy_model_from_ks2_path(ks2_path)
    r = spike_sorting_metrics(m.spike_times, m.spike_clusters, m.amplitudes, params=METRICS_PARAMS)
    #  includes the ks2 contamination
    file_contamination = ks2_path.joinpath('cluster_ContamPct.tsv')
    if file_contamination.exists():
        contam = pd.read_csv(file_contamination, sep='\t')
        contam.rename(columns={'ContamPct': 'ks2_contamination_pct'}, inplace=True)
        r = r.set_index('cluster_id', drop=False).join(contam.set_index('cluster_id'))

    #  includes the ks2 labeling
    file_labels = ks2_path.joinpath('cluster_KSLabel.tsv')
    if file_labels.exists():
        ks2_labels = pd.read_csv(file_labels, sep='\t')
        ks2_labels.rename(columns={'KSLabel': 'ks2_label'}, inplace=True)
        r = r.set_index('cluster_id', drop=False).join(ks2_labels.set_index('cluster_id'))

    if save:
        #  the file name contains the label of the probe (directory name in this case)
        r.to_csv(ks2_path.joinpath(f'cluster_metrics.csv'))

    return r


def spike_sorting_metrics(spike_times, spike_clusters, spike_amplitudes,
                          params=METRICS_PARAMS, epochs=None):
    """ Spike sorting QC metrics """
    cluster_ids = np.arange(np.max(spike_clusters) + 1)
    nclust = cluster_ids.size
    r = Bunch({
        'cluster_id': cluster_ids,
        'num_spikes': np.zeros(nclust, ) + np.nan,
        'firing_rate': np.zeros(nclust, ) + np.nan,
        'presence_ratio': np.zeros(nclust, ) + np.nan,
        'presence_ratio_std': np.zeros(nclust, ) + np.nan,
        'isi_viol': np.zeros(nclust, ) + np.nan,
        'amplitude_cutoff': np.zeros(nclust, ) + np.nan,
        'amplitude_std': np.zeros(nclust, ) + np.nan,
        # 'isolation_distance': np.zeros(nclust, ) + np.nan,
        # 'l_ratio': np.zeros(nclust, ) + np.nan,
        # 'd_prime': np.zeros(nclust, ) + np.nan,
        # 'nn_hit_rate': np.zeros(nclust, ) + np.nan,
        # 'nn_miss_rate': np.zeros(nclust, ) + np.nan,
        # 'silhouette_score': np.zeros(nclust, ) + np.nan,
        # 'max_drift': np.zeros(nclust, ) + np.nan,
        # 'cumulative_drift': np.zeros(nclust, ) + np.nan,
        'epoch_name': np.zeros(nclust, dtype='object'),
    })

    tmin = 0
    tmax = spike_times[-1]

    """computes basic metrics such as spike rate and presence ratio"""
    presence_ratio = bincount2D(spike_times, spike_clusters,
                                xbin=params['presence_bin_length_secs'],
                                ybin=cluster_ids, xlim=[tmin, tmax])[0]
    r.num_spikes = np.sum(presence_ratio > 0, axis=1)
    r.firing_rate = r.num_spikes / params['presence_bin_length_secs']
    r.presence_ratio = np.sum(presence_ratio > 0, axis=1) / presence_ratio.shape[1]
    r.presence_ratio_std = np.std(presence_ratio, axis=1)

    # loop over each cluster
    for ic in np.arange(nclust):
        # slice the spike_times array
        ispikes = spike_clusters == cluster_ids[ic]
        if np.all(~ispikes):
            continue
        st = spike_times[ispikes]
        sa = spike_amplitudes[ispikes]
        # compute metrics
        r.isi_viol[ic], _ = isi_violations(st, tmin, tmax,
                                           isi_threshold=params['isi_threshold'],
                                           min_isi=params['min_isi'])
        r.amplitude_cutoff[ic] = amplitude_cutoff(amplitudes=sa)
        r.amplitude_std[ic] = np.std(sa)

    return pd.DataFrame(r)


def isi_violations(spike_train, min_time, max_time, isi_threshold, min_isi=0):
    """Calculate ISI violations for a spike train.

    Based on metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    modified by Dan Denman from cortex-lab/sortingQuality GitHub by Nick Steinmetz

    Inputs:
    -------
    spike_train : array of spike times
    min_time : minimum time for potential spikes
    max_time : maximum time for potential spikes
    isi_threshold : threshold for isi violation
    min_isi : threshold for duplicate spikes

    Outputs:
    --------
    fpRate : rate of contaminating spikes as a fraction of overall rate
        A perfect unit has a fpRate = 0
        A unit with some contamination has a fpRate < 0.5
        A unit with lots of contamination has a fpRate > 1.0
    num_violations : total number of violations

    """

    duplicate_spikes = np.where(np.diff(spike_train) <= min_isi)[0]

    spike_train = np.delete(spike_train, duplicate_spikes + 1)
    isis = np.diff(spike_train)

    num_spikes = spike_train.size
    num_violations = np.sum(isis < isi_threshold)
    violation_time = 2 * num_spikes * (isi_threshold - min_isi)
    total_rate = spike_train.size / (max_time - min_time)
    violation_rate = num_violations / violation_time
    fpRate = violation_rate / total_rate

    return fpRate, num_violations


def amplitude_cutoff(amplitudes, num_histogram_bins=500, histogram_smoothing_value=3):
    """ Calculate approximate fraction of spikes missing from a distribution of amplitudes

    Assumes the amplitude histogram is symmetric (not valid in the presence of drift)

    Inspired by metric described in Hill et al. (2011) J Neurosci 31: 8699-8705

    Input:
    ------
    amplitudes : numpy.ndarray
        Array of amplitudes (don't need to be in physical units)

    Output:
    -------
    fraction_missing : float
        Fraction of missing spikes (0-0.5)
        If more than 50% of spikes are missing, an accurate estimate isn't possible

    """

    h, b = np.histogram(amplitudes, num_histogram_bins, density=True)

    pdf = gaussian_filter1d(h, histogram_smoothing_value)
    support = b[:-1]

    peak_index = np.argmax(pdf)
    G = np.argmin(np.abs(pdf[peak_index:] - pdf[0])) + peak_index

    bin_size = np.mean(np.diff(support))
    fraction_missing = np.sum(pdf[G:]) * bin_size

    fraction_missing = np.min([fraction_missing, 0.5])

    return fraction_missing


def phy_model_from_ks2_path(ks2_path):
    params_file = ks2_path.joinpath('params.py')
    if params_file.exists():
        m = model.load_model(params_file)
    else:
        meta_file = next(ks2_path.rglob('*.ap.meta'), None)
        if meta_file and meta_file.exists():
            meta = spikeglx.read_meta_data(meta_file)
            fs = spikeglx._get_fs_from_meta(meta)
            nch = (spikeglx._get_nchannels_from_meta(meta) -
                   len(spikeglx._get_sync_trace_indices_from_meta(meta)))
        else:
            fs = 30000
            nch = 384
        m = model.TemplateModel(dir_path=ks2_path,
                                dat_path=[],
                                sample_rate=fs,
                                n_channels_dat=nch)
    return m


# Make a bunch gathering all trial QC
def qc_fpga_task(fpga_trials, alf_trials):
    """
    :fpga_task is the dictionary output of
    ibllib.io.extractors.ephys_fpga.extract_behaviour_sync
    : bpod_trials is the dictionary output of ibllib.io.extractors.ephys_trials.extract_all
    : alf_trials is the ALF _ibl_trials object after extraction (alf.io.load_object)
    :return: qc_session, qc_trials, True means QC passes while False indicates a failure
    """

    GOCUE_STIMON_DELAY = 0.01
    FEEDBACK_STIMFREEZE_DELAY = 0.01
    VALVE_STIM_OFF_DELAY = 1
    VALVE_STIM_OFF_JITTER = 0.1
    ITI_IN_STIM_OFF_JITTER = 0.1
    ERROR_STIM_OFF_DELAY = 2
    ERROR_STIM_OFF_JITTER = 0.1
    RESPONSE_FEEDBACK_DELAY = 0.0005

    def strictly_after(t0, t1, threshold):
        """ returns isafter, iswithinthreshold"""
        return (t1 - t0) > 0, np.abs((t1 - t0)) <= threshold

    ntrials = fpga_trials['stimOn_times'].size
    qc_trials = Bunch({})

    """
    First Check consistency of the dataset: whithin each trial, all events happen after trial
    start should not be NaNs and increasing. This is not a QC but an assertion.
    """
    status = True
    for k in ['goCueTrigger_times_bpod', 'response_times', 'stimOn_times', 'response_times_bpod',
              'goCueTrigger_times', 'goCue_times', 'feedback_times']:
        if k.endswith('_bpod'):
            tstart = alf_trials['intervals_bpod'][:, 0]
        else:
            tstart = alf_trials['intervals'][:, 0]
        selection = ~np.isnan(alf_trials[k])
        status &= np.all(alf_trials[k][selection] - tstart[selection] > 0)
        status &= np.all(np.diff(alf_trials[k][selection]) > 0)
    assert status

    """
    This part of the function uses only fpga_trials information
    """
    # check number of feedbacks: should always be one
    qc_trials['n_feedback'] = (np.uint32(~np.isnan(fpga_trials['valve_open'])) +
                               np.uint32(~np.isnan(fpga_trials['error_tone_in'])))

    # check for non-Nans
    qc_trials['stimOn_times_nan'] = ~np.isnan(fpga_trials['stimOn_times'])
    qc_trials['goCue_times_nan'] = ~np.isnan(fpga_trials['goCue_times'])

    # stimOn before goCue
    qc_trials['stimOn_times_before_goCue_times'], qc_trials['stimOn_times_goCue_times_delay'] =\
        strictly_after(fpga_trials['stimOn_times'], fpga_trials['goCue_times'], GOCUE_STIMON_DELAY)

    # stimFreeze before feedback
    qc_trials['stim_freeze_before_feedback'], qc_trials['stim_freeze_feedback_delay'] = \
        strictly_after(fpga_trials['stim_freeze'], fpga_trials['feedback_times'],
                       FEEDBACK_STIMFREEZE_DELAY)

    # stimOff 1 sec after valve, with 0.1 as acceptable jitter
    qc_trials['stimOff_delay_valve'] = np.less(
        np.abs(fpga_trials['stimOff_times'] - fpga_trials['valve_open'] - VALVE_STIM_OFF_DELAY),
        VALVE_STIM_OFF_JITTER, out=np.ones(ntrials, dtype=np.bool),
        where=~np.isnan(fpga_trials['valve_open']))

    # iti_in whithin 0.01 sec of stimOff
    qc_trials['iti_in_delay_stim_off'] = \
        np.abs(fpga_trials['stimOff_times'] - fpga_trials['iti_in']) < ITI_IN_STIM_OFF_JITTER

    # stimOff 2 secs after error_tone_in with jitter
    # noise off happens 2 secs after stimm, with 0.1 as acceptable jitter
    qc_trials['stimOff_delay_noise'] = np.less(
        np.abs(fpga_trials['stimOff_times'] - fpga_trials['error_tone_in'] - ERROR_STIM_OFF_DELAY),
        ERROR_STIM_OFF_JITTER, out=np.ones(ntrials, dtype=np.bool),
        where=~np.isnan(fpga_trials['error_tone_in']))

    """
    This part uses only alf_trials information
    """
    # TEST  Response times (from session start) should be increasing continuously
    #       Note: RT are not durations but time stamps from session start
    #       1. check for non-Nans
    qc_trials['response_times_nan'] = ~np.isnan(alf_trials['response_times'])
    #       2. check for positive increase
    qc_trials['response_times_increase'] = \
        np.diff(np.append([0], alf_trials['response_times'])) > 0
    # TEST  Response times (from goCue) should be positive
    qc_trials['response_times_goCue_times_diff'] = \
        alf_trials['response_times'] - alf_trials['goCue_times'] > 0
    # TEST  1. Response_times should be before feedback
    qc_trials['response_before_feedback'] = \
        alf_trials['feedback_times'] - alf_trials['response_times'] > 0
    #       2. Delay between wheel reaches threshold (response time) and
    #       feedback is 100us, acceptable jitter 500 us
    qc_trials['response_feedback_delay'] = \
        alf_trials['feedback_times'] - alf_trials['response_times'] < RESPONSE_FEEDBACK_DELAY

    # Test output at session level
    qc_session = {k: np.all(qc_trials[k]) for k in qc_trials}

    return qc_session, qc_trials

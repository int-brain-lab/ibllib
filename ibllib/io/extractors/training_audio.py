#!/usr/bin/env python
# -*- coding:utf-8 -*-
from pathlib import Path
import logging

import numpy as np
from scipy import signal
from scipy.io import wavfile

from ibllib import dsp
import ibllib.io.raw_data_loaders as ioraw
import ibllib.io.extractors.training_trials

logger_ = logging.getLogger('ibllib')

NS_WIN = 2 ** 18  # 2 ** np.ceil(np.log2(1 * fs))
OVERLAP = NS_WIN / 2
NS_WELCH = 512
FTONE = 5000
UNIT = 'dBFS'  # dBFS or dbSPL


def _running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def _detect_ready_tone(w, fs):
    # get envelope of DC free signal and envelope of BP signal around freq of interest
    h = np.abs(signal.hilbert(w - np.median(w)))
    fh = np.abs(signal.hilbert(dsp.bp(w, si=1 / fs, b=FTONE * np.array([0.9, 0.95, 1.15, 1.1]))))
    dtect = _running_mean(fh / (h + 1e-3), int(fs * 0.1)) > 0.8
    return np.where(np.diff(dtect.astype(int)) == 1)[0]
    # tone = np.sin(2 * np.pi * FTONE * np.arange(0, fs * 0.1) / fs)
    # tone = tone / np.sum(tone ** 2)
    # xc = np.abs(signal.hilbert(signal.correlate(w - np.mean(w), tone)))


def _get_conversion_factor():
    # 3 approaches here (not exclusive):
    # a- get the mic sensitivity, the preamp gain and DAC parameters and do the math
    # b- treat the whole thing as a black box and do a calibration run (cf. people at Renard's lab)
    # c- use calibrated ready tone
    # The reference of acoustic pressure is 0dBSPL @ 1kHz which is threshold of hearing (20 μPa).
    # Usual calibration is 1 Pa (94 dBSPL) at 1 kHz
    # c) here we know that the ready tone is 55dB SPL at 5kHz, assuming a flat spectrum between
    # 1 and 5 kHz, and observing the peak value on the 5k at the microphone.
    if UNIT == 'dBFS':
        return 1.0
    distance_to_the_mic = .155
    peak_value_observed = 60
    rms_value_observed = np.sqrt(2) / 2 * peak_value_observed
    fac = 10 ** ((55 - 20 * np.log10(rms_value_observed)) / 20) * distance_to_the_mic
    return fac


def welchogram(fs, wav, nswin=NS_WIN, overlap=OVERLAP, nperseg=NS_WELCH):
    """
    Computes a spectrogram on a very large audio file.

    :param fs: sampling frequency (Hz)
    :param wav: wav signal (vector or memmap)
    :param nswin: n samples of the sliding window
    :param overlap: n samples of the overlap between windows
    :param nperseg: n samples for the computation of the spectrogram
    :return: tscale, fscale, downsampled_spectrogram
    """
    ns = wav.shape[0]
    window_generator = dsp.WindowGenerator(ns=ns, nswin=nswin, overlap=overlap)
    nwin = window_generator.nwin
    fscale = dsp.fscale(nperseg, 1 / fs, one_sided=True)
    W = np.zeros((nwin, len(fscale)))
    tscale = window_generator.tscale(fs=fs)
    detect = []
    for first, last in window_generator.firstlast:
        # load the current window into memory
        w = np.float64(wav[first:last]) * _get_conversion_factor()
        # detection of ready tones
        a = [d + first for d in _detect_ready_tone(w, fs)]
        if len(a):
            detect += a
        # the last window may not allow a pwelch
        if (last - first) < nperseg:
            continue
        # compute PSD estimate for the current window
        iw = window_generator.iw
        _, W[iw, :] = signal.welch(w, fs=fs, window='hanning', nperseg=nperseg, axis=-1,
                                   detrend='constant', return_onesided=True, scaling='density')
        if (iw % 50) == 0:
            window_generator.print_progress()
    # the onset detection may have duplicates with sliding window, average them and remove
    detect = np.sort(np.array(detect)) / fs
    ind = np.where(np.diff(detect) < 0.1)[0]
    detect[ind] = (detect[ind] + detect[ind + 1]) / 2
    detect = np.delete(detect, ind + 1)
    return tscale, fscale, W, detect


def extract_sound(ses_path, save=True, force=False):
    """
    Simple audio features extraction for ambient sound characterization.
    From a wav file, generates several ALF files to be registered on Alyx

    :param ses_path: ALF full session path: (/mysubject001/YYYY-MM-DD/001)
    :return: None
    """
    ses_path = Path(ses_path)
    wav_file = ses_path / 'raw_behavior_data' / '_iblrig_micData.raw.wav'
    if not wav_file.exists():
        return None
    files_out = {'power': ses_path / 'alf' / '_ibl_audioSpectrogram.power.npy',
                 'frequencies': ses_path / 'alf' / '_ibl_audioSpectrogram.frequencies.npy',
                 'onset_times': ses_path / 'alf' / '_ibl_audioOnsetGoCue.times_microphone.npy',
                 'times_microphone': Path(ses_path) / 'alf' /
                                     '_ibl_audioSpectrogram.times_microphone.npy',
                 'times': Path(ses_path) / 'alf' / '_ibl_audioSpectrogram.times.npy'
                 }
    # if they exist and the option Force is set to false, do not recompute and exit
    if all([files_out[f].exists() for f in files_out]) and not force:
        logger_.warning('Output exists. Skipping ' + str(wav_file) + ' Use force flag to override')
        return
    # crunch the wav file
    fs, wav = wavfile.read(wav_file, mmap=False)
    if len(wav) == 0:
        logger_.error('WAV Header Indicates empty file. Abort.' + str(wav_file))
        return
    tscale, fscale, W, detect = welchogram(fs, wav)
    # save files
    alf_folder = ses_path / 'alf'
    if not alf_folder.exists():
        alf_folder.mkdir()
    np.save(file=files_out['power'], arr=W.astype(np.single))
    np.save(file=files_out['frequencies'], arr=fscale[None, :].astype(np.single))
    np.save(file=files_out['onset_times'], arr=detect)
    np.save(file=files_out['times_microphone'], arr=tscale[:, None].astype(np.single))

    # for the time scale, attempt to synchronize using onset sound detection and task data
    data = ioraw.load_data(ses_path)
    if data is None:  # if no session data, we're done
        return
    tgocue = np.array(ibllib.io.extractors.training_trials.get_goCueOnset_times(
        None, save=False, data=data))
    ilast = min(len(tgocue), len(detect))
    dt = tgocue[:ilast] - detect[: ilast]
    # only save if dt is consistent for the whole session
    if np.std(dt) < 0.2:
        tscale += np.median(dt)
        np.save(file=files_out['times'], arr=tscale[:, None].astype(np.single))

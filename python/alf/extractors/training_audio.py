import numpy as np
from scipy import signal
from scipy.io import wavfile

from ibllib import dsp

NS_WIN = 2 ** 18 # 2 ** np.ceil(np.log2(1 * fs))
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
    # a- get the mic sensitivity, the preamp gain and DAC parameters and do the maths
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
    rms_value_observed = np.sqrt(2)/2 * peak_value_observed
    fac = 10 ** ((55 - 20.*np.log10(rms_value_observed)) / 20) * distance_to_the_mic
    return fac


def welchogram(wav_file, save=True):
    fs, wav = wavfile.read(wav_file, mmap=True)
    ns = wav.shape[0]
    window_generator = dsp.WindowGenerator(ns=ns, nswin=NS_WIN, overlap=OVERLAP)
    nwin = window_generator.nwin
    fscale = dsp.fscale(NS_WELCH, 1 / fs, one_sided=True)
    W = np.zeros((nwin, len(fscale)))

    detect = []
    for first, last in window_generator.slices:
        # the last window may not allow a pwelch
        if (last - first) < NS_WELCH:
            continue
        # compute PSD estimate for the current window
        w = np.float64(wav[first:last]) * _get_conversion_factor()
        iw = window_generator.iw
        _, W[iw, :] = signal.welch(w, fs=fs, window='hanning', nperseg=NS_WELCH, detrend='constant',
                                   return_onesided=True, scaling='density', axis=-1)
        # detection of ready tones
        a = [d + first for d in _detect_ready_tone(w, fs)]
        if len(a):
            detect +=  a
        if (iw % 100) == 0:
            print(iw, nwin, first, last)

    return fscale, W


def extract_sound(wav_file, save=True):
    """
    Simple audio features extraction for ambient sound characterization.
    From a wav file, generates several ALF files to be registered on Alyx
    :param wav_file: sound file
    :return: None
    """
    fs, wav = wavfile.read(wav_file, mmap=True)
    ns = wav.shape[0]
    window_generator = dsp.WindowGenerator(ns=ns, nswin=NS_WIN, overlap=OVERLAP)
    nwin = window_generator.nwin

    RES = {'rms':         np.zeros((nwin,) ),
           'nsamples':    np.zeros((nwin,) ),
           'first_last':  np.zeros((nwin, 2) ),
           'fscale': dsp.fscale(NS_WELCH, 1 / fs, one_sided=True)}
    W = np.zeros((nwin, len(RES['fscale'])))

    detect = []
    for first, last in window_generator.slices:
        # the last window may not allow a pwelch
        if (last - first) < NS_WELCH:
            continue
        # compute PSD estimate for the current window
        w = np.float64(wav[first:last]) * _get_conversion_factor()
        iw = window_generator.iw
        _, W[iw, :] = signal.welch(w, fs=fs, window='hanning', nperseg=NS_WELCH, detrend='constant',
                                   return_onesided=True, scaling='density', axis=-1)
        # detection of ready tones
        a = [d + first for d in _detect_ready_tone(w, fs)]
        if len(a):
            detect +=  a
        if (iw % 100) == 0:
            print(iw, nwin, first, last)

    # need to save:
    # - welchogram for the full session
    # - periodogram per trial
    # - periodogram full session



# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:57:53 2020

Functions to analyse LFP signals

@author: Guido Meijer
"""

from scipy.signal import welch, csd, filtfilt, butter
import numpy as np


def butter_filter(signal, highpass_freq=None, lowpass_freq=None, order=4, fs=2500):

    # The filter type is determined according to the values of cut-off frequencies
    Fn = fs / 2.
    if lowpass_freq and highpass_freq:
        if highpass_freq < lowpass_freq:
            Wn = (highpass_freq / Fn, lowpass_freq / Fn)
            btype = 'bandpass'
        else:
            Wn = (lowpass_freq / Fn, highpass_freq / Fn)
            btype = 'bandstop'
    elif lowpass_freq:
        Wn = lowpass_freq / Fn
        btype = 'lowpass'
    elif highpass_freq:
        Wn = highpass_freq / Fn
        btype = 'highpass'
    else:
        raise ValueError("Either highpass_freq or lowpass_freq must be given")

    # Filter signal
    b, a = butter(order, Wn, btype=btype, output='ba')
    filtered_data = filtfilt(b=b, a=a, x=signal, axis=1)

    return filtered_data


def power_spectrum(signal, fs=2500, segment_length=0.5, segment_overlap=0.5, scaling='density'):
    """
    Calculate the power spectrum of an LFP signal

    Parameters
    ----------
    signal : 2D array
        LFP signal from different channels in V with dimensions (channels X samples)
    fs : int
        Sampling frequency
    segment_length : float
        Length of the segments for which the spectral density is calcualted in seconds
    segment_overlap : float
        Fraction of overlap between the segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap)

    Returns
    ----------
    freqs : 1D array
        Frequencies for which the spectral density is calculated
    psd : 2D array
        Power spectrum in V^2 with dimensions (channels X frequencies)

    """

    # Transform segment from seconds to samples
    segment_samples = int(fs * segment_length)
    overlap_samples = int(segment_overlap * segment_samples)

    # Calculate power spectrum
    freqs, psd = welch(signal, fs=fs, nperseg=segment_samples, noverlap=overlap_samples,
                       scaling=scaling)
    return freqs, psd


def coherence(signal_a, signal_b, fs=2500, segment_length=1, segment_overlap=0.5):
    """
    Calculate the coherence between two LFP signals

    Parameters
    ----------
    signal_a : 1D array
        LFP signal from different channels with dimensions (channels X samples)
    fs : int
        Sampling frequency
    segment_length : float
        Length of the segments for which the spectral density is calcualted in seconds
    segment_overlap : float
        Fraction of overlap between the segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap)

    Returns
    ----------
    freqs : 1D array
        Frequencies for which the coherence is calculated
    coherence : 1D array
        Coherence takes a value between 0 and 1, with 0 or 1 representing no or perfect coherence,
        respectively
    phase_lag : 1D array
        Estimate of phase lag in radian between the input time series for each frequency

    """

    # Transform segment from seconds to samples
    segment_samples = int(fs * segment_length)
    overlap_samples = int(segment_overlap * segment_samples)

    # Calculate coherence
    freqs, Pxx = welch(signal_a, fs=fs, nperseg=segment_samples, noverlap=overlap_samples)
    _, Pyy = welch(signal_b, fs=fs, nperseg=segment_samples, noverlap=overlap_samples)
    _, Pxy = csd(signal_a, signal_b, fs=fs, nperseg=segment_samples, noverlap=overlap_samples)
    coherence = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    phase_lag = np.angle(Pxy)

    return freqs, coherence, phase_lag

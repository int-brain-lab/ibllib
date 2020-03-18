# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:57:53 2020

Functions to analyse LFP

@author: guido
"""

from ._elephant import welch_psd


def power_spectrum(signal, fs=2500, segment_length=5, segment_overlap=0.5):
    """
    Calculate the power spectrum of an LFP signal

    Parameters
    ----------
    signal : 2D array
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
        Frequencies for which the spectral density is calculated
    psd : 2D array
        Power spectral density in V^2/Hz with dimensions (channels X frequencies)

    """
    segment_samples = fs * segment_length
    freqs, psd = welch_psd(signal, fs=fs, len_seg=segment_samples, overlap=segment_overlap,
                           scaling='spectrum')
    return freqs, psd

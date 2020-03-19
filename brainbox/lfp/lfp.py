# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:57:53 2020

Functions to analyse LFP, this is basically a wrapper for functions from Elephant

@author: Guido Meijer
"""

import neo
import quantities as pq
from ._elephant import welch_psd, welch_cohere
from ._elephant import spike_triggered_average as spk_tr_av
from ._elephant import spike_field_coherence as spk_fd_coh


def power_spectrum(signal, fs=2500, segment_length=1, segment_overlap=0.5):
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
    segment_samples = fs * segment_length

    # Calculate power spectrum
    freqs, psd = welch_psd(signal, fs=fs,
                           len_seg=segment_samples, overlap=segment_overlap,
                           scaling='spectrum')
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

    # Transform segment length from seconds to samples
    segment_samples = fs * segment_length

    # Calculate coherence
    freqs, coherence, phase_lag = welch_cohere(signal_a, signal_b, fs=fs,
                                               len_seg=segment_samples,
                                               overlap=segment_overlap,
                                               scaling='spectrum')
    return freqs, coherence, phase_lag


def spike_triggered_average(signal, spiketrain, spike_window=[-5, 10], fs=2500):
    """
    Calculate the spike triggered average

    Parameters
    ----------
    signal : 1D or 2D array
        LFP signal from either one channel or multiple channels
        dimensions (channels X samples)
    spiketrain : 1D array
        Timestamps of spikes in seconds
    spike_window : 2 element vector
        Lenght of window around each spike to average in ms
    fs : int
        Sampling frequency of LFP channel

    Returns
    ----------
    sta : 1D or 2D array
        Array of spike triggered averages in uV
        dimensions (channels x samples)
    time : 1D array
        Time vector in ms
    """

    assert len(spiketrain.shape) == 1

    # Transform into neo objects
    signal_obj = neo.AnalogSignal(signal.T, units='V', sampling_rate=fs * pq.Hz)
    short_spike_train = spiketrain[((spiketrain > signal_obj.t_start)
                                    & (spiketrain < signal_obj.t_stop))]
    spiketrain_obj = neo.SpikeTrain(short_spike_train * pq.s, signal_obj.t_stop)

    # Calculate spike triggered average
    result_sta = spk_tr_av(signal_obj, spiketrain_obj, (spike_window[0] * pq.ms,
                                                        spike_window[1] * pq.ms))
    sta = result_sta.rescale('uV').magnitude
    sta = sta.T[0]
    time = result_sta.times.rescale('ms').magnitude

    return sta, time

"""
Elephant - Electrophysiology Analysis Toolkit
Liscence: BSD 3-Clause

Copyright (c) 2014-2019, Elephant authors and contributors
All rights reserved.
"""

from __future__ import division, print_function, unicode_literals

import neo
import numpy as np
import scipy.signal
import quantities as pq
from neo.core import AnalogSignal, SpikeTrain
import warnings
from ._conversion import BinnedSpikeTrain


def welch_psd(signal, num_seg=8, len_seg=None, freq_res=None, overlap=0.5,
              fs=1.0, window='hanning', nfft=None, detrend='constant',
              return_onesided=True, scaling='density', axis=-1):
    """
    Estimates power spectrum density (PSD) of a given `neo.AnalogSignal`
    using Welch's method.
    The PSD is obtained through the following steps:
    1. Cut the given data into several overlapping segments. The degree of
       overlap can be specified by parameter `overlap` (default is 0.5,
       i.e. segments are overlapped by the half of their length).
       The number and the length of the segments are determined according
       to the parameters `num_seg`, `len_seg` or `freq_res`. By default, the
       data is cut into 8 segments;
    2. Apply a window function to each segment. Hanning window is used by
       default. This can be changed by giving a window function or an
       array as parameter `window` (see Notes [2]);
    3. Compute the periodogram of each segment;
    4. Average the obtained periodograms to yield PSD estimate.
    Parameters
    ----------
    signal : neo.AnalogSignal or pq.Quantity or np.ndarray
        Time series data, of which PSD is estimated. When `signal` is
        `pq.Quantity` or `np.ndarray`, sampling frequency should be given
        through the keyword argument `fs`. Otherwise, the default value is
        used (`fs` = 1.0).
    num_seg : int, optional
        Number of segments. The length of segments is adjusted so that
        overlapping segments cover the entire stretch of the given data. This
        parameter is ignored if `len_seg` or `freq_res` is given.
        Default: 8.
    len_seg : int, optional
        Length of segments. This parameter is ignored if `freq_res` is given.
        If None, it will be determined from other parameters.
        Default: None.
    freq_res : pq.Quantity or float, optional
        Desired frequency resolution of the obtained PSD estimate in terms of
        the interval between adjacent frequency bins. When given as a `float`,
        it is taken as frequency in Hz.
        If None, it will be determined from other parameters.
        Default: None.
    overlap : float, optional
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap).
        Default: 0.5 (half-overlapped).
    fs : pq.Quantity or float, optional
        Specifies the sampling frequency of the input time series. When the
        input is given as a `neo.AnalogSignal`, the sampling frequency is
        taken from its attribute and this parameter is ignored.
        Default: 1.0.
    window : str or tuple or np.ndarray, optional
        Desired window to use.
        See Notes [2].
        Default: 'hanning'.
    nfft : int, optional
        Length of the FFT used.
        See Notes [2].
        Default: None.
    detrend : str or function or False, optional
        Specifies how to detrend each segment.
        See Notes [2].
        Default: 'constant'.
    return_onesided : bool, optional
        If True, return a one-sided spectrum for real data.
        If False return a two-sided spectrum.
        See Notes [2].
        Default: True.
    scaling : {'density', 'spectrum'}, optional
        If 'density', computes the power spectral density where Pxx has units
        of V**2/Hz. If 'spectrum', computes the power spectrum where Pxx has
        units of V**2, if `signal` is measured in V and `fs` is measured in
        Hz.
        See Notes [2].
        Default: 'density'.
    axis : int, optional
        Axis along which the periodogram is computed.
        See Notes [2].
        Default: last axis (-1).
    Returns
    -------
    freqs : pq.Quantity or np.ndarray
        Frequencies associated with the power estimates in `psd`.
        `freqs` is always a vector irrespective of the shape of the input
        data in `signal`.
        If `signal` is `neo.AnalogSignal` or `pq.Quantity`, a `pq.Quantity`
        array is returned.
        Otherwise, a `np.ndarray` containing frequency in Hz is returned.
    psd : pq.Quantity or np.ndarray
        PSD estimates of the time series in `signal`.
        If `signal` is `neo.AnalogSignal`, a `pq.Quantity` array is returned.
        Otherwise, the return is a `np.ndarray`.
    Raises
    ------
    ValueError
        If `overlap` is not in the interval [0, 1).
        If `freq_res` is not positive.
        If `freq_res` is too high for the given data size.
        If `freq_res` is None and `len_seg` is not a positive number.
        If `freq_res` is None and `len_seg` is greater than the length of data
        on `axis`.
        If both `freq_res` and `len_seg` are None and `num_seg` is not a
        positive number.
        If both `freq_res` and `len_seg` are None and `num_seg` is greater
        than the length of data on `axis`.
    Notes
    -----
    1. The computation steps used in this function are implemented in
       `scipy.signal` module, and this function is a wrapper which provides
       a proper set of parameters to `scipy.signal.welch` function.
    2. The parameters `window`, `nfft`, `detrend`, `return_onesided`,
       `scaling`, and `axis` are directly passed to the `scipy.signal.welch`
       function. See the respective descriptions in the docstring of
       `scipy.signal.welch` for usage.
    3. When only `num_seg` is given, parameter `nperseg` of
       `scipy.signal.welch` function is determined according to the expression
       `signal.shape[axis]` / (`num_seg` - `overlap` * (`num_seg` - 1))
       converted to integer.
    See Also
    --------
    scipy.signal.welch
    """

    # initialize a parameter dict (to be given to scipy.signal.welch()) with
    # the parameters directly passed on to scipy.signal.welch()
    params = {'window': window, 'nfft': nfft,
              'detrend': detrend, 'return_onesided': return_onesided,
              'scaling': scaling, 'axis': axis}

    # add the input data to params. When the input is AnalogSignal, the
    # data is added after rolling the axis for time index to the last
    data = np.asarray(signal)
    if isinstance(signal, neo.AnalogSignal):
        data = np.rollaxis(data, 0, len(data.shape))
    params['x'] = data

    # if the data is given as AnalogSignal, use its attribute to specify
    # the sampling frequency
    if hasattr(signal, 'sampling_rate'):
        params['fs'] = signal.sampling_rate.rescale('Hz').magnitude
    else:
        params['fs'] = fs

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    elif 1 <= overlap:
        raise ValueError("overlap must be less then 1")

    # determine the length of segments (i.e. *nperseg*) according to given
    # parameters
    if freq_res is not None:
        if freq_res <= 0:
            raise ValueError("freq_res must be positive")
        dF = freq_res.rescale('Hz').magnitude \
            if isinstance(freq_res, pq.quantity.Quantity) else freq_res
        nperseg = int(params['fs'] / dF)
        if nperseg > data.shape[axis]:
            raise ValueError("freq_res is too high for the given data size")
    elif len_seg is not None:
        if len_seg <= 0:
            raise ValueError("len_seg must be a positive number")
        elif data.shape[axis] < len_seg:
            raise ValueError("len_seg must be shorter than the data length")
        nperseg = len_seg
    else:
        if num_seg <= 0:
            raise ValueError("num_seg must be a positive number")
        elif data.shape[axis] < num_seg:
            raise ValueError("num_seg must be smaller than the data length")
        # when only *num_seg* is given, *nperseg* is determined by solving the
        # following equation:
        #  num_seg * nperseg - (num_seg-1) * overlap * nperseg = data.shape[-1]
        #  -----------------   ===============================   ^^^^^^^^^^^
        # summed segment lengths        total overlap            data length
        nperseg = int(data.shape[axis] / (num_seg - overlap * (num_seg - 1)))
    params['nperseg'] = nperseg
    params['noverlap'] = int(nperseg * overlap)

    freqs, psd = scipy.signal.welch(**params)

    # attach proper units to return values
    if isinstance(signal, pq.quantity.Quantity):
        if 'scaling' in params and params['scaling'] == 'spectrum':
            psd = psd * signal.units * signal.units
        else:
            psd = psd * signal.units * signal.units / pq.Hz
        freqs = freqs * pq.Hz

    return freqs, psd


def welch_cohere(x, y, num_seg=8, len_seg=None, freq_res=None, overlap=0.5,
                 fs=1.0, window='hanning', nfft=None, detrend='constant',
                 scaling='density', axis=-1):
    r"""
    Estimates coherence between a given pair of analog signals.
    The estimation is performed with Welch's method: the given pair of data
    are cut into short segments, cross-spectra are calculated for each pair of
    segments, and the cross-spectra are averaged and normalized by respective
    auto-spectra.
    By default, the data are cut into 8 segments with 50% overlap between
    neighboring segments. These numbers can be changed through respective
    parameters.
    Parameters
    ----------
    x : neo.AnalogSignal or pq.Quantity or np.ndarray
        First time series data of the pair between which coherence is
        computed.
    y : neo.AnalogSignal or pq.Quantity or np.ndarray
        Second time series data of the pair between which coherence is
        computed.
        The shapes and the sampling frequencies of `x` and `y` must be
        identical. When `x` and `y` are not `neo.AnalogSignal`, sampling
        frequency should be specified through the keyword argument `fs`.
        Otherwise, the default value is used (`fs` = 1.0).
    num_seg : int, optional
        Number of segments. The length of segments is adjusted so that
        overlapping segments cover the entire stretch of the given data. This
        parameter is ignored if `len_seg` or `freq_res` is given.
        Default: 8.
    len_seg : int, optional
        Length of segments. This parameter is ignored if `freq_res` is given.
        If None, it is determined from other parameters.
        Default: None.
    freq_res : pq.Quantity or float, optional
        Desired frequency resolution of the obtained coherence estimate in
        terms of the interval between adjacent frequency bins. When given as a
        `float`, it is taken as frequency in Hz.
        If None, it is determined from other parameters.
        Default: None.
    overlap : float, optional
        Overlap between segments represented as a float number between 0 (no
        overlap) and 1 (complete overlap).
        Default: 0.5 (half-overlapped).
    fs : pq.Quantity or float, optional
        Specifies the sampling frequency of the input time series. When the
        input time series are given as `neo.AnalogSignal`, the sampling
        frequency is taken from their attribute and this parameter is ignored.
        Default: 1.0.
    window : str or tuple or np.ndarray, optional
        Desired window to use.
        See Notes [1].
        Default: 'hanning'.
    nfft : int, optional
        Length of the FFT used.
        See Notes [1].
        Default: None.
    detrend : str or function or False, optional
        Specifies how to detrend each segment.
        See Notes [1].
        Default: 'constant'.
    scaling : {'density', 'spectrum'}, optional
        If 'density', computes the power spectral density where Pxx has units
        of V**2/Hz. If 'spectrum', computes the power spectrum where Pxx has
        units of V**2, if `signal` is measured in V and `fs` is measured in
        Hz.
        See Notes [1].
        Default: 'density'.
    axis : int, optional
        Axis along which the periodogram is computed.
        See Notes [1].
        Default: last axis (-1).
    Returns
    -------
    freqs : pq.Quantity or np.ndarray
        Frequencies associated with the estimates of coherency and phase lag.
        `freqs` is always a vector irrespective of the shape of the input
        data. If `x` and `y` are `neo.AnalogSignal` or `pq.Quantity`, a
        `pq.Quantity` array is returned. Otherwise, a `np.ndarray` containing
        frequency in Hz is returned.
    coherency : np.ndarray
        Estimate of coherency between the input time series. For each
        frequency, coherency takes a value between 0 and 1, with 0 or 1
        representing no or perfect coherence, respectively.
        When the input arrays `x` and `y` are multi-dimensional, `coherency`
        is of the same shape as the inputs, and the frequency is indexed
        depending on the type of the input. If the input is
        `neo.AnalogSignal`, the first axis indexes frequency. Otherwise,
        frequency is indexed by the last axis.
    phase_lag : pq.Quantity or np.ndarray
        Estimate of phase lag in radian between the input time series. For
        each frequency, phase lag takes a value between :math:`-\pi` and
        :math:`\pi`, with positive values meaning phase precession of `x`
        ahead of `y`, and vice versa. If `x` and `y` are `neo.AnalogSignal` or
        `pq.Quantity`, a `pq.Quantity` array is returned. Otherwise, a
        `np.ndarray` containing phase lag in radian is returned.
        The axis for frequency index is determined in the same way as for
        `coherency`.
    Raises
    ------
    ValueError
        If `overlap` is not in the interval [0, 1).
        If `freq_res` is not positive.
        If `freq_res` is too high for the given data size.
        If `freq_res` is None and `len_seg` is not a positive number.
        If `freq_res` is None and `len_seg` is greater than the length of data
        on `axis`.
        If both `freq_res` and `len_seg` are None and `num_seg` is not a
        positive number.
        If both `freq_res` and `len_seg` are None and `num_seg` is greater
        than the length of data on `axis`.
    Notes
    -----
    1. The parameters `window`, `nfft`, `detrend`, `scaling`, and `axis` are
       directly passed to the helper function `_welch`. See the
       respective descriptions in the docstring of `_welch` for usage.
    2. When only `num_seg` is given, parameter `nperseg` for `_welch` function
       is determined according to the expression
       `x.shape[axis]` / (`num_seg` - `overlap` * (`num_seg` - 1))
       converted to integer.
    See Also
    --------
    spectral._welch
    """

    # initialize a parameter dict for scipy.signal.csd()
    params = {'window': window, 'nfft': nfft,
              'detrend': detrend, 'scaling': scaling, 'axis': axis}

    # When the input is AnalogSignal, the axis for time index is rolled to
    # the last
    xdata = np.asarray(x)
    ydata = np.asarray(y)
    if isinstance(x, neo.AnalogSignal):
        xdata = np.rollaxis(xdata, 0, len(xdata.shape))
        ydata = np.rollaxis(ydata, 0, len(ydata.shape))

    # if the data is given as AnalogSignal, use its attribute to specify
    # the sampling frequency
    if hasattr(x, 'sampling_rate'):
        params['fs'] = x.sampling_rate.rescale('Hz').magnitude
    else:
        params['fs'] = fs

    if overlap < 0:
        raise ValueError("overlap must be greater than or equal to 0")
    elif 1 <= overlap:
        raise ValueError("overlap must be less then 1")

    # determine the length of segments (i.e. *nperseg*) according to given
    # parameters
    if freq_res is not None:
        if freq_res <= 0:
            raise ValueError("freq_res must be positive")
        dF = freq_res.rescale('Hz').magnitude \
            if isinstance(freq_res, pq.quantity.Quantity) else freq_res
        nperseg = int(params['fs'] / dF)
        if nperseg > xdata.shape[axis]:
            raise ValueError("freq_res is too high for the given data size")
    elif len_seg is not None:
        if len_seg <= 0:
            raise ValueError("len_seg must be a positive number")
        elif xdata.shape[axis] < len_seg:
            raise ValueError("len_seg must be shorter than the data length")
        nperseg = len_seg
    else:
        if num_seg <= 0:
            raise ValueError("num_seg must be a positive number")
        elif xdata.shape[axis] < num_seg:
            raise ValueError("num_seg must be smaller than the data length")
        # when only *num_seg* is given, *nperseg* is determined by solving the
        # following equation:
        #  num_seg * nperseg - (num_seg-1) * overlap * nperseg = data.shape[-1]
        #  -----------------   ===============================   ^^^^^^^^^^^
        # summed segment lengths        total overlap            data length
        nperseg = int(xdata.shape[axis] / (num_seg - overlap * (num_seg - 1)))
    params['nperseg'] = nperseg
    params['noverlap'] = int(nperseg * overlap)

    freqs, Pxx = scipy.signal.welch(xdata, **params)
    _, Pyy = scipy.signal.welch(ydata, **params)
    _, Pxy = scipy.signal.csd(xdata, ydata, **params)

    coherency = np.abs(Pxy) ** 2 / (Pxx * Pyy)
    phase_lag = np.angle(Pxy)

    # attach proper units to return values
    if isinstance(x, pq.quantity.Quantity):
        freqs = freqs * pq.Hz
        phase_lag = phase_lag * pq.rad

    # When the input is AnalogSignal, the axis for frequency index is
    # rolled to the first to comply with the Neo convention about time axis
    if isinstance(x, neo.AnalogSignal):
        coherency = np.rollaxis(coherency, -1)
        phase_lag = np.rollaxis(phase_lag, -1)

    return freqs, coherency, phase_lag


def spike_triggered_average(signal, spiketrains, window):
    """
    Calculates the spike-triggered averages of analog signals in a time window
    relative to the spike times of a corresponding spiketrain for multiple
    signals each. The function receives n analog signals and either one or
    n spiketrains. In case it is one spiketrain this one is muliplied n-fold
    and used for each of the n analog signals.
    Parameters
    ----------
    signal : neo AnalogSignal object
        'signal' contains n analog signals.
    spiketrains : one SpikeTrain or one numpy ndarray or a list of n of either of these.
        'spiketrains' contains the times of the spikes in the spiketrains.
    window : tuple of 2 Quantity objects with dimensions of time.
        'window' is the start time and the stop time, relative to a spike, of
        the time interval for signal averaging.
        If the window size is not a multiple of the sampling interval of the
        signal the window will be extended to the next multiple.
    Returns
    -------
    result_sta : neo AnalogSignal object
        'result_sta' contains the spike-triggered averages of each of the
        analog signals with respect to the spikes in the corresponding
        spiketrains. The length of 'result_sta' is calculated as the number
        of bins from the given start and stop time of the averaging interval
        and the sampling rate of the analog signal. If for an analog signal
        no spike was either given or all given spikes had to be ignored
        because of a too large averaging interval, the corresponding returned
        analog signal has all entries as nan. The number of used spikes and
        unused spikes for each analog signal are returned as annotations to
        the returned AnalogSignal object.
    Examples
    --------
    >>> signal = neo.AnalogSignal(np.array([signal1, signal2]).T, units='mV',
    ...                                sampling_rate=10/ms)
    >>> stavg = spike_triggered_average(signal, [spiketrain1, spiketrain2],
    ...                                 (-5 * ms, 10 * ms))
    """

    # checking compatibility of data and data types
    # window_starttime: time to specify the start time of the averaging
    # interval relative to a spike
    # window_stoptime: time to specify the stop time of the averaging
    # interval relative to a spike
    window_starttime, window_stoptime = window
    if not (isinstance(window_starttime, pq.quantity.Quantity) and
            window_starttime.dimensionality.simplified ==
            pq.Quantity(1, "s").dimensionality):
        raise TypeError("The start time of the window (window[0]) "
                        "must be a time quantity.")
    if not (isinstance(window_stoptime, pq.quantity.Quantity) and
            window_stoptime.dimensionality.simplified ==
            pq.Quantity(1, "s").dimensionality):
        raise TypeError("The stop time of the window (window[1]) "
                        "must be a time quantity.")
    if window_stoptime <= window_starttime:
        raise ValueError("The start time of the window (window[0]) must be "
                         "earlier than the stop time of the window (window[1]).")

    # checks on signal
    if not isinstance(signal, AnalogSignal):
        raise TypeError(
            "Signal must be an AnalogSignal, not %s." % type(signal))
    if len(signal.shape) > 1:
        # num_signals: number of analog signals
        num_signals = signal.shape[1]
    else:
        raise ValueError("Empty analog signal, hence no averaging possible.")
    if window_stoptime - window_starttime > signal.t_stop - signal.t_start:
        raise ValueError("The chosen time window is larger than the "
                         "time duration of the signal.")

    # spiketrains type check
    if isinstance(spiketrains, (np.ndarray, SpikeTrain)):
        spiketrains = [spiketrains]
    elif isinstance(spiketrains, list):
        for st in spiketrains:
            if not isinstance(st, (np.ndarray, SpikeTrain)):
                raise TypeError(
                    "spiketrains must be a SpikeTrain, a numpy ndarray, or a "
                    "list of one of those, not %s." % type(spiketrains))
    else:
        raise TypeError(
            "spiketrains must be a SpikeTrain, a numpy ndarray, or a list of "
            "one of those, not %s." % type(spiketrains))

    # multiplying spiketrain in case only a single spiketrain is given
    if len(spiketrains) == 1 and num_signals != 1:
        template = spiketrains[0]
        spiketrains = []
        for i in range(num_signals):
            spiketrains.append(template)

    # checking for matching numbers of signals and spiketrains
    if num_signals != len(spiketrains):
        raise ValueError(
            "The number of signals and spiketrains has to be the same.")

    # checking the times of signal and spiketrains
    for i in range(num_signals):
        if spiketrains[i].t_start < signal.t_start:
            raise ValueError(
                "The spiketrain indexed by %i starts earlier than "
                "the analog signal." % i)
        if spiketrains[i].t_stop > signal.t_stop:
            raise ValueError(
                "The spiketrain indexed by %i stops later than "
                "the analog signal." % i)

    # *** Main algorithm: ***

    # window_bins: number of bins of the chosen averaging interval
    window_bins = int(np.ceil(((window_stoptime - window_starttime) *
        signal.sampling_rate).simplified))
    # result_sta: array containing finally the spike-triggered averaged signal
    result_sta = AnalogSignal(np.zeros((window_bins, num_signals)),
        sampling_rate=signal.sampling_rate, units=signal.units)
    # setting of correct times of the spike-triggered average
    # relative to the spike
    result_sta.t_start = window_starttime
    used_spikes = np.zeros(num_signals, dtype=int)
    unused_spikes = np.zeros(num_signals, dtype=int)
    total_used_spikes = 0

    for i in range(num_signals):
        # summing over all respective signal intervals around spiketimes
        for spiketime in spiketrains[i]:
            # checks for sufficient signal data around spiketime
            if (spiketime + window_starttime >= signal.t_start and
                    spiketime + window_stoptime <= signal.t_stop):
                # calculating the startbin in the analog signal of the
                # averaging window for spike
                startbin = int(np.floor(((spiketime + window_starttime -
                    signal.t_start) * signal.sampling_rate).simplified))
                # adds the signal in selected interval relative to the spike
                result_sta[:, i] += signal[
                    startbin: startbin + window_bins, i]
                # counting of the used spikes
                used_spikes[i] += 1
            else:
                # counting of the unused spikes
                unused_spikes[i] += 1

        # normalization
        result_sta[:, i] = result_sta[:, i] / used_spikes[i]

        total_used_spikes += used_spikes[i]

    if total_used_spikes == 0:
        warnings.warn(
            "No spike at all was either found or used for averaging")
    result_sta.annotate(used_spikes=used_spikes, unused_spikes=unused_spikes)

    return result_sta


def spike_field_coherence(signal, spiketrain, **kwargs):
    """
    Calculates the spike-field coherence between a analog signal(s) and a
    (binned) spike train.
    The current implementation makes use of scipy.signal.coherence(). Additional
    kwargs will will be directly forwarded to scipy.signal.coherence(),
    except for the axis parameter and the sampling frequency, which will be
    extracted from the input signals.
    The spike_field_coherence function receives an analog signal array and
    either a binned spike train or a spike train containing the original spike
    times. In case of original spike times the spike train is binned according
    to the sampling rate of the analog signal array.
    The AnalogSignal object can contain one or multiple signal traces. In case
    of multiple signal traces, the spike field coherence is calculated
    individually for each signal trace and the spike train.
    Parameters
    ----------
    signal : neo AnalogSignal object
        'signal' contains n analog signals.
    spiketrain : SpikeTrain or BinnedSpikeTrain
        Single spike train to perform the analysis on. The binsize of the
        binned spike train must match the sampling_rate of signal.
    KWArgs
    ------
    All KWArgs are passed to scipy.signal.coherence().
    Returns
    -------
    coherence : complex Quantity array
        contains the coherence values calculated for each analog signal trace
        in combination with the spike train. The first dimension corresponds to
        the frequency, the second to the number of the signal trace.
    frequencies : Quantity array
        contains the frequency values corresponding to the first dimension of
        the 'coherence' array
    Example
    -------
    Plot the SFC between a regular spike train at 20 Hz, and two sinusoidal
    time series at 20 Hz and 23 Hz, respectively.
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from quantities import s, ms, mV, Hz, kHz
    >>> import neo, elephant
    >>> t = pq.Quantity(range(10000),units='ms')
    >>> f1, f2 = 20. * Hz, 23. * Hz
    >>> signal = neo.AnalogSignal(np.array([
            np.sin(f1 * 2. * np.pi * t.rescale(s)),
            np.sin(f2 * 2. * np.pi * t.rescale(s))]).T,
            units=pq.mV, sampling_rate=1. * kHz)
    >>> spiketrain = neo.SpikeTrain(
        range(t[0], t[-1], 50), units='ms',
        t_start=t[0], t_stop=t[-1])
    >>> sfc, freqs = elephant.sta.spike_field_coherence(
        signal, spiketrain, window='boxcar')
    >>> plt.plot(freqs, sfc[:,0])
    >>> plt.plot(freqs, sfc[:,1])
    >>> plt.xlabel('Frequency [Hz]')
    >>> plt.ylabel('SFC')
    >>> plt.xlim((0, 60))
    >>> plt.show()
    """

    if not hasattr(scipy.signal, 'coherence'):
        raise AttributeError('scipy.signal.coherence is not available. The sfc '
                             'function uses scipy.signal.coherence for '
                             'the coherence calculation. This function is '
                             'available for scipy version 0.16 or newer. '
                             'Please update you scipy version.')

    # spiketrains type check
    if not isinstance(spiketrain, (SpikeTrain, BinnedSpikeTrain)):
        raise TypeError(
            "spiketrain must be of type SpikeTrain or BinnedSpikeTrain, "
            "not %s." % type(spiketrain))

    # checks on analogsignal
    if not isinstance(signal, AnalogSignal):
        raise TypeError(
            "Signal must be an AnalogSignal, not %s." % type(signal))
    if len(signal.shape) > 1:
        # num_signals: number of individual traces in the analog signal
        num_signals = signal.shape[1]
    elif len(signal.shape) == 1:
        num_signals = 1
    else:
        raise ValueError("Empty analog signal.")
    len_signals = signal.shape[0]

    # bin spiketrain if necessary
    if isinstance(spiketrain, SpikeTrain):
        spiketrain = BinnedSpikeTrain(
            spiketrain, binsize=signal.sampling_period)

    # check the start and stop times of signal and spike trains
    if spiketrain.t_start < signal.t_start:
        raise ValueError(
            "The spiketrain starts earlier than the analog signal.")
    if spiketrain.t_stop > signal.t_stop:
        raise ValueError(
            "The spiketrain stops later than the analog signal.")

    # check equal time resolution for both signals
    if spiketrain.binsize != signal.sampling_period:
        raise ValueError(
            "The spiketrain and signal must have a "
            "common sampling frequency / binsize")

    # calculate how many bins to add on the left of the binned spike train
    delta_t = spiketrain.t_start - signal.t_start
    if delta_t % spiketrain.binsize == 0:
        left_edge = int((delta_t / spiketrain.binsize).magnitude)
    else:
        raise ValueError("Incompatible binning of spike train and LFP")
    right_edge = int(left_edge + spiketrain.num_bins)

    # duplicate spike trains
    spiketrain_array = np.zeros((1, len_signals))
    spiketrain_array[0, left_edge:right_edge] = spiketrain.to_array()
    spiketrains_array = np.repeat(spiketrain_array, repeats=num_signals, axis=0).transpose()

    # calculate coherence
    frequencies, sfc = scipy.signal.coherence(
        spiketrains_array, signal.magnitude,
        fs=signal.sampling_rate.rescale('Hz').magnitude,
        axis=0, **kwargs)

    return (pq.Quantity(sfc, units=pq.dimensionless),
            pq.Quantity(frequencies, units=pq.Hz))

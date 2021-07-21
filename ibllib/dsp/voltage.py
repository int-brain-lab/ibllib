"""
Module to work with raw voltage traces. Spike sorting pre-processing functions.
"""
from pathlib import Path

import numpy as np
import scipy.signal
from tqdm import tqdm

from ibllib.io import spikeglx
import ibllib.dsp.fourier as fdsp
from ibllib.dsp import fshift
from ibllib.ephys import neuropixel


def reject_channels(x, fs, butt_kwargs=None, threshold=0.6, trx=1):
    """
    Computes the
    :param x: demultiplexed array (ntraces, nsample)
    :param fs: sampling frequency (Hz)
    :param trx: number of traces each side (1)
    :param butt kwargs (optional, None), butt_kwargs = {'N': 4, 'Wn': 0.05, 'btype': 'lp'}
    :param threshold: r value below which a channel is rejected
    :return:
    """
    ntr, ns = x.shape
    # mirror padding by taking care of not repeating first/last trace
    x = np.r_[x[1:trx + 1, :], x, x[-2 - trx:-2, :]]
    # apply butterworth
    if butt_kwargs is not None:
        sos = scipy.signal.butter(**butt_kwargs, output='sos')
        x = scipy.signal.sosfiltfilt(sos, x)
    r = np.zeros(ntr)
    for ix in np.arange(trx, ntr + trx):
        ref = np.median(x[ix - trx: ix + trx + 1, :], axis=0)
        r[ix - trx] = np.corrcoef(x[ix, :], ref)[1, 0]
    return r >= threshold, r


def agc(x, wl=.5, si=.002, epsilon=1e-8):
    """
    Automatic gain control
    :param x: seismic array (sample last dimension)
    :param wl: window length (secs)
    :param si: sampling interval (secs)
    :param epsilon: whitening (useful mainly for synthetic data)
    :return:
    """
    ns_win = np.round(wl / si / 2) * 2 + 1
    w = np.hanning(ns_win)
    w /= np.sum(w)
    gain = np.sqrt(fdsp.convolve(x ** 2, w, mode='same'))
    gain += (np.sum(gain, axis=1) * epsilon / x.shape[-1])[:, np.newaxis]
    gain = 1 / gain
    return x * gain, gain


def fk(x, si=.002, dx=1, vbounds=None, btype='highpass', ntr_pad=0, ntr_tap=None, lagc=.5,
       collection=None, kfilt=None):
    """Frequency-wavenumber filter: filters apparent plane-waves velocity
    :param x: the input array to be filtered. dimension, the filtering is considering
    axis=0: spatial dimension, axis=1 temporal dimension. (ntraces, ns)
    :param si: sampling interval (secs)
    :param dx: spatial interval (usually meters)
    :param vbounds: velocity high pass [v1, v2], cosine taper from 0 to 1 between v1 and v2
    :param btype: {‘lowpass’, ‘highpass’}, velocity filter : defaults to highpass
    :param ntr_pad: padding will add ntr_padd mirrored traces to each side
    :param ntr_tap: taper (if None, set to ntr_pad)
    :param lagc: length of agc in seconds. If set to None or 0, no agc
    :param kfilt: optional (None) if kfilter is applied, parameters as dict (bounds are in m-1
    according to the dx parameter) kfilt = {'bounds': [0.05, 0.1], 'btype', 'highpass'}
    :param collection: vector length ntraces. Each unique value set of traces is a collection
    on which the FK filter will run separately (shot gaters, receiver gathers)
    :return:
    """
    if collection is not None:
        xout = np.zeros_like(x)
        for c in np.unique(collection):
            sel = collection == c
            xout[sel, :] = fk(x[sel, :], si=si, dx=dx, vbounds=vbounds, ntr_pad=ntr_pad,
                              ntr_tap=ntr_tap, lagc=lagc, collection=None)
        return xout

    assert vbounds
    nx, nt = x.shape

    # lateral padding left and right
    ntr_pad = int(ntr_pad)
    ntr_tap = ntr_pad if ntr_tap is None else ntr_tap
    nxp = nx + ntr_pad * 2

    # compute frequency wavenumber scales and deduce the velocity filter
    fscale = fdsp.fscale(nt, si)
    kscale = fdsp.fscale(nxp, dx)
    kscale[0] = 1e-6
    v = fscale[np.newaxis, :] / kscale[:, np.newaxis]
    if btype.lower() in ['highpass', 'hp']:
        fk_att = fdsp.fcn_cosine(vbounds)(np.abs(v))
    elif btype.lower() in ['lowpass', 'lp']:
        fk_att = (1 - fdsp.fcn_cosine(vbounds)(np.abs(v)))

    # if a k-filter is also provided, apply it
    if kfilt is not None:
        katt = fdsp._freq_vector(np.abs(kscale), kfilt['bounds'], typ=kfilt['btype'])
        fk_att *= katt[:, np.newaxis]

    # import matplotlib.pyplot as plt
    # plt.imshow(np.fft.fftshift(np.abs(v), axes=0).T, aspect='auto', vmin=0, vmax=1e5,
    #            extent=[np.min(kscale), np.max(kscale), 0, np.max(fscale) * 2])
    # plt.imshow(np.fft.fftshift(np.abs(fk_att), axes=0).T, aspect='auto', vmin=0, vmax=1,
    #            extent=[np.min(kscale), np.max(kscale), 0, np.max(fscale) * 2])

    # apply the attenuation in fk-domain
    if not lagc:
        xf = np.copy(x)
        gain = 1
    else:
        xf, gain = agc(x, wl=lagc, si=si)
    if ntr_pad > 0:
        # pad the array with a mirrored version of itself and apply a cosine taper
        xf = np.r_[np.flipud(xf[:ntr_pad]), xf, np.flipud(xf[-ntr_pad:])]
    if ntr_tap > 0:
        taper = fdsp.fcn_cosine([0, ntr_tap])(np.arange(nxp))  # taper up
        taper *= 1 - fdsp.fcn_cosine([nxp - ntr_tap, nxp])(np.arange(nxp))   # taper down
        xf = xf * taper[:, np.newaxis]
    xf = np.real(np.fft.ifft2(fk_att * np.fft.fft2(xf)))

    if ntr_pad > 0:
        xf = xf[ntr_pad:-ntr_pad, :]
    return xf / gain


def kfilt(x, collection=None, ntr_pad=0, ntr_tap=None, lagc=300, butter_kwargs=None):
    """
    Applies a butterworth filter on the 0-axis with tapering / padding
    :param x: the input array to be filtered. dimension, the filtering is considering
    axis=0: spatial dimension, axis=1 temporal dimension. (ntraces, ns)
    :param collection:
    :param ntr_pad: traces added to each side (mirrored)
    :param ntr_tap: n traces for apodizatin on each side
    :param lagc: window size for time domain automatic gain control (no agc otherwise)
    :param butter_kwargs: filtering parameters: defaults: {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}
    :return:
    """
    if butter_kwargs is None:
        butter_kwargs = {'N': 3, 'Wn': 0.1, 'btype': 'highpass'}
    if collection is not None:
        xout = np.zeros_like(x)
        for c in np.unique(collection):
            sel = collection == c
            xout[sel, :] = kfilt(x=x[sel, :], ntr_pad=0, ntr_tap=None, collection=None,
                                 butter_kwargs=butter_kwargs)
        return xout
    nx, nt = x.shape

    # lateral padding left and right
    ntr_pad = int(ntr_pad)
    ntr_tap = ntr_pad if ntr_tap is None else ntr_tap
    nxp = nx + ntr_pad * 2

    # apply agc and keep the gain in handy
    if not lagc:
        xf = np.copy(x)
        gain = 1
    else:
        xf, gain = agc(x, wl=lagc, si=1.0)
    if ntr_pad > 0:
        # pad the array with a mirrored version of itself and apply a cosine taper
        xf = np.r_[np.flipud(xf[:ntr_pad]), xf, np.flipud(xf[-ntr_pad:])]
    if ntr_tap > 0:
        taper = fdsp.fcn_cosine([0, ntr_tap])(np.arange(nxp))  # taper up
        taper *= 1 - fdsp.fcn_cosine([nxp - ntr_tap, nxp])(np.arange(nxp))   # taper down
        xf = xf * taper[:, np.newaxis]
    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    xf = scipy.signal.sosfiltfilt(sos, xf, axis=0)

    if ntr_pad > 0:
        xf = xf[ntr_pad:-ntr_pad, :]
    return xf / gain


def destripe(x, fs, tr_sel=None, neuropixel_version=1, butter_kwargs=None, k_kwargs=None):
    """Super Car (super slow also...) - far from being set in stone but a good workflow example
    :param x: demultiplexed array (ntraces, nsample)
    :param fs: sampling frequency
    :param neuropixel_version (optional): 1 or 2. Useful for the ADC shift correction. If None,
     no correction is applied
    :param tr_sel: index array for the first axis of x indicating the selected traces.
     On a full workflow, one should scan sparingly the full file to get a robust estimate of the
     selection. If None, and estimation is done using only the current batch is provided for
     convenience but should be avoided in production.
    :param butter_kwargs: (optional, None) butterworth params, see the code for the defaults dict
    :param k_kwargs: (optional, None) K-filter params, see the code for the defaults dict
    :return: x, filtered array
    """
    if butter_kwargs is None:
        butter_kwargs = {'N': 3, 'Wn': 300 / fs / 2, 'btype': 'highpass'}
    if k_kwargs is None:
        k_kwargs = {'ntr_pad': 60, 'ntr_tap': 0, 'lagc': 3000,
                    'butter_kwargs': {'N': 3, 'Wn': 0.01, 'btype': 'highpass'}}
    h = neuropixel.trace_header(version=neuropixel_version)
    # butterworth
    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    x = scipy.signal.sosfiltfilt(sos, x)
    # apply ADC shift
    if neuropixel_version is not None:
        x = fshift(x, h['sample_shift'], axis=1)
    # apply spatial filter on good channel selection only
    x_ = kfilt(x, **k_kwargs)
    return x_


def decompress_destripe_cbin(sr, output_file=None, h=None):
    """
    From a spikeglx Reader object, decompresses and apply ADC
    Production version with optimized FFTs - requires pyfftw
    :param sr: seismic reader object (spikeglx.Reader)
    :param output_file: (optional, defaults to .bin extension of the compressed bin file)
    :param h: (optional)
    :return:
    """
    import pyfftw

    SAMPLES_TAPER = 128
    NBATCH = 65536
    # handles input parameters
    if isinstance(sr, str) or isinstance(sr, Path):
        sr = spikeglx.Reader(sr, open=True)
    butter_kwargs = {'N': 3, 'Wn': 300 / sr.fs / 2, 'btype': 'highpass'}
    k_kwargs = {'ntr_pad': 60, 'ntr_tap': 0, 'lagc': 3000,
                'butter_kwargs': {'N': 3, 'Wn': 0.01, 'btype': 'highpass'}}
    h = neuropixel.trace_header(version=1) if h is None else h
    output_file = sr.file_bin.with_suffix('.bin') if output_file is None else output_file
    assert output_file != sr.file_bin
    taper = np.r_[0, scipy.signal.windows.cosine((SAMPLES_TAPER - 1) * 2), 0]
    # create the FFT stencils
    ncv = h['x'].size  # number of channels
    # compute LP filter coefficients
    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    # compute fft stencil for batchsize
    win = pyfftw.empty_aligned((ncv, NBATCH), dtype='float32')
    WIN = pyfftw.empty_aligned((ncv, int(NBATCH / 2 + 1)), dtype='complex64')
    fft_object = pyfftw.FFTW(win, WIN, axes=(1,), direction='FFTW_FORWARD', threads=4)
    ifft_object = pyfftw.FFTW(WIN, win, axes=(1,), direction='FFTW_BACKWARD', threads=4)
    dephas = np.zeros((ncv, NBATCH), dtype=np.float32)
    dephas[:, 1] = 1.
    DEPHAS = np.exp(1j * np.angle(fft_object(dephas)) * h['sample_shift'][:, np.newaxis])

    pbar = tqdm(total=sr.ns / sr.fs)
    with open(output_file, 'wb') as fid:
        first_s = 0
        while True:
            last_s = np.minimum(NBATCH + first_s, sr.ns)
            # transpose to get faster processing for all trace based process
            chunk = sr[first_s:last_s, :ncv].T
            chunk[:, :SAMPLES_TAPER] *= taper[:SAMPLES_TAPER]
            chunk[:, -SAMPLES_TAPER:] *= taper[SAMPLES_TAPER:]
            # apply butterworth
            chunk = scipy.signal.sosfiltfilt(sos, chunk)
            # apply adc
            ind2save = [SAMPLES_TAPER, NBATCH - SAMPLES_TAPER]
            if last_s == sr.ns:
                # for the last batch just use the normal fft as the stencil doesn't fit
                chunk = fshift(chunk, s=h['sample_shift'])
                ind2save[1] = NBATCH
            else:
                # apply precomputed fshift of the proper length
                chunk = ifft_object(fft_object(chunk) * DEPHAS)
            if first_s == 0:
                # for the first batch save the start with taper applied
                ind2save[0] = 0
            # apply K-filter
            chunk = kfilt(chunk, **k_kwargs)
            # add back sync trace and save
            chunk = np.r_[chunk, sr[first_s:last_s, ncv:].T].T
            (chunk[slice(*ind2save), :] / sr.channel_conversion_sample2v['ap']
             ).astype(np.int16).tofile(fid)
            first_s += NBATCH - SAMPLES_TAPER * 2
            pbar.update(NBATCH / sr.fs)
            if last_s == sr.ns:
                break
    pbar.close()

"""
Module to work with raw voltage traces. Spike sorting pre-processing functions.
"""
import numpy as np
import scipy.signal

import ibllib.dsp.fourier as fdsp
from ibllib.dsp import fshift, voltage
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


def destripe(x, fs, tr_sel=None, neuropixel_version=1, butter_kwargs=None, fk_kwargs=None):
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
    :param fk_kwargs: (optional, None) FK params, see the code for the defaults dict
    :return: x, filtered array
    """
    if butter_kwargs is None:
        butter_kwargs = {'N': 3, 'Wn': 300 / fs / 2, 'btype': 'highpass'}
    if fk_kwargs is None:
        fk_kwargs = {'dx': 1, 'vbounds': [0, 1e6], 'ntr_pad': 60, 'ntr_tap': 0,
                     'lagc': .01, 'btype': 'lowpass'}
    h = neuropixel.trace_header(version=neuropixel_version)
    # butterworth
    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    x = scipy.signal.sosfiltfilt(sos, x)
    # apply ADC shift
    if neuropixel_version is not None:
        x = fshift(x, h['sample_shift'], axis=1)
    # detect faulty channels if single batch
    if tr_sel is None:
        reject_channel_kwargs = {'butt_kwargs': {'N': 4, 'Wn': 0.05, 'btype': 'lp'}, 'trx': 1}
        tr_sel, _ = reject_channels(x, fs, **reject_channel_kwargs)
    # apply spatial filter on good channel selection only
    x_ = np.zeros_like(x)
    x_[tr_sel, :] = voltage.fk(x[tr_sel, :], si=1 / fs, **fk_kwargs)
    return x_

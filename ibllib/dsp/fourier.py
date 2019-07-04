"""
Low-level functions to work in frequency domain for n-dim arrays
"""

import numpy as np


def fscale(ns, si=1, one_sided=False):
    """
    numpy.fft.fftfreq returns Nyquist as a negative frequency so we propose this instead

    :param ns: number of samples
    :param si: sampling interval in seconds
    :param one_sided: if True, returns only positive frequencies
    :return: fscale: numpy vector containing frequencies in Hertz
    """
    fsc = np.arange(0, np.floor(ns / 2) + 1) / ns / si  # sample the frequency scale
    if one_sided:
        return fsc
    else:
        return np.concatenate((fsc, -fsc[slice(-2 + (ns % 2), 0, -1)]), axis=0)


def freduce(x, axis=None):
    """
    Reduces a spectrum to positive frequencies only
    Works on the last dimension (contiguous in c-stored array)

    :param x: numpy.ndarray
    :param axis: axis along which to perform reduction (last axis by default)
    :return: numpy.ndarray
    """
    if axis is None:
        axis = x.ndim - 1
    siz = list(x.shape)
    siz[axis] = int(np.floor(siz[axis] / 2 + 1))
    return np.take(x, np.arange(0, siz[axis]), axis=axis)


def fexpand(x, ns=1, axis=None):
    """
    Reconstructs full spectrum from positive frequencies
    Works on the last dimension (contiguous in c-stored array)

    :param x: numpy.ndarray
    :param axis: axis along which to perform reduction (last axis by default)
    :return: numpy.ndarray
    """
    if axis is None:
        axis = x.ndim - 1
    # dec = int(ns % 2) * 2 - 1
    # xcomp = np.conj(np.flip(x[..., 1:x.shape[-1] + dec], axis=axis))
    ilast = int((ns + (ns % 2)) / 2)
    xcomp = np.conj(np.flip(np.take(x, np.arange(1, ilast), axis=axis), axis=axis))
    return np.concatenate((x, xcomp), axis=axis)


def bp(ts, si, b, axis=None):
    """
    Band-pass filter in frequency domain

    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 4 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ='bp')


def lp(ts, si, b, axis=None):
    """
    Low-pass filter in frequency domain

    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 2 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ='lp')


def hp(ts, si, b, axis=None):
    """
    High-pass filter in frequency domain

    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 2 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ='hp')


def _freq_filter(ts, si, b, axis=None, typ='lp'):
    """
        Wrapper for hp/lp/bp filters
    """
    if axis is None:
        axis = ts.ndim - 1
    ns = ts.shape[axis]
    f = fscale(ns, si=si, one_sided=True)
    if typ == 'bp':
        filc = _freq_vector(f, b[0:2], typ='hp') * _freq_vector(f, b[2:4], typ='lp')
    else:
        filc = _freq_vector(f, b, typ=typ)
    if axis < (ts.ndim - 1):
        filc = filc[:, np.newaxis]
    return np.real(np.fft.ifft(np.fft.fft(ts, axis=axis) * fexpand(filc, ns, axis=0), axis=axis))


def _freq_vector(f, b, typ='lp'):
    """
        Returns a frequency modulated vector for filtering

        :param f: frequency vector, uniform and monotonic
        :param b: 2 bounds array
        :return: amplitude modulated frequency vector
    """
    filc = ((f <= b[0]).astype(float) +
            np.bitwise_and(f > b[0], f < b[1]).astype(float) *
            (0.5 * (1 + np.sin(np.pi * (f - ((b[0] + b[1]) / 2)) /
             (b[0] - b[1])))))
    if typ == 'hp':
        return 1 - filc
    elif typ == 'lp':
        return filc

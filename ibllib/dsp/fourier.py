"""
Low-level functions to work in frequency domain for n-dim arrays
"""

import numpy as np
import scipy
from ibllib.dsp.utils import fcn_cosine


def convolve(x, w, mode='full'):
    """
    Frequency domain convolution along the last dimension (2d arrays)
    Will broadcast if a matrix is convolved with a vector
    :param x:
    :param w:
    :return: convolution
    """
    nsx = x.shape[-1]
    nsw = w.shape[-1]
    ns = ns_optim_fft(nsx + nsw)
    x_ = np.concatenate((x, np.zeros([*x.shape[:-1], ns - nsx], dtype=x.dtype)), axis=-1)
    w_ = np.concatenate((w, np.zeros([*w.shape[:-1], ns - nsw], dtype=w.dtype)), axis=-1)
    xw = np.fft.irfft(np.fft.rfft(x_, axis=-1) * np.fft.rfft(w_, axis=-1), axis=-1)
    xw = xw[..., :(nsx + nsw)]  # remove 0 padding
    if mode == 'full':
        return xw
    elif mode == 'same':
        first = int(np.floor(nsw / 2)) - ((nsw + 1) % 2)
        last = int(np.ceil(nsw / 2)) + ((nsw + 1) % 2)
        return xw[..., first:-last]


def ns_optim_fft(ns):
    """
    Gets the next higher combination of factors of 2 and 3 than ns to compute efficient ffts
    :param ns:
    :return: nsoptim
    """
    p2, p3 = np.meshgrid(2 ** np.arange(25), 3 ** np.arange(15))
    sz = np.unique((p2 * p3).flatten())
    return sz[np.searchsorted(sz, ns)]


def dephas(w, phase, axis=-1):
    """
    dephas a signal by a given angle in degrees
    :param w:
    :param phase: phase in degrees
    :param axis:
    :return:
    """
    ns = w.shape[axis]
    W = freduce(np.fft.fft(w, axis=axis), axis=axis) * np.exp(- 1j * phase / 180 * np.pi)
    return np.real(np.fft.ifft(fexpand(W, ns=ns, axis=axis), axis=axis))


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
    filc = fcn_cosine(b)(f)
    if typ == 'hp':
        return filc
    elif typ == 'lp':
        return 1 - filc


def shift(w, s, axis=-1):
    """
    Shifts a signal in frequency domain, to allow for accurate non-integer shifts
    :param w: input signal
    :param s: shift in samples, positive shifts forward
    :param axis: axis along which to shift
    :return: w
    """
    ns = np.array(w.shape) * 0 + 1
    ns[axis] = w.shape[axis]
    dephas = np.zeros(ns)
    np.put(dephas, 1, 1)
    W = freduce(scipy.fft.fft(w, axis=axis), axis=axis)
    dephas = freduce(scipy.fft.fft(dephas, axis=axis), axis=axis)
    dephas = np.exp(1j * np.angle(dephas) * s)
    return np.real(scipy.fft.ifft(fexpand(W * dephas, ns[axis], axis=axis), axis=axis))


def fit_phase(w, si=1, fmin=0, fmax=None, axis=-1):
    """
    Performs a linear regression on the unwrapped phase of a wavelet to obtain a time-delay
    :param w: wavelet (usually a cross-correlation)
    :param si: sampling interval
    :param fmin: sampling interval
    :param fnax: sampling interval
    :param axis:
    :return: dt
    """
    if fmax is None:
        fmax = 1 / si / 2
    ns = w.shape[axis]
    freqs = freduce(fscale(ns, si=si))
    phi = np.unwrap(np.angle(freduce(np.fft.fft(w, axis=axis), axis=axis)))
    indf = np.logical_and(fmin < freqs, freqs < fmax)
    dt = - np.polyfit(freqs[indf],
                      np.swapaxes(phi.compress(indf, axis=axis), axis, 0), 1)[0] / np.pi / 2
    return dt


def dft(x, xscale=None, axis=-1, kscale=None):
    """
    1D discrete fourier transform. Vectorized.
    :param x: 1D numpy array to be transformed
    :param xscale: time or spatial index of each sample
    :param axis: for multidimensional arrays, axis along which the ft is computed
    :param kscale: (optional) fourier coefficient. All if complex input, positive if real
    :return: 1D complex numpy array
    """
    ns = x.shape[axis]
    if xscale is None:
        xscale = np.arange(ns)
    if kscale is None:
        nk = ns if np.any(np.iscomplex(x)) else np.ceil((ns + 1) / 2)
        kscale = np.arange(nk)
    else:
        nk = kscale.size
    if axis != 0:
        # the axis of the transform always needs to be the first
        x = np.swapaxes(x, axis, 0)
    shape = np.array(x.shape)
    x = np.reshape(x, (ns, int(np.prod(x.shape) / ns)))
    # compute fourier coefficients
    exp = np.exp(- 1j * 2 * np.pi / ns * xscale * kscale[:, np.newaxis])
    X = np.matmul(exp, x)
    shape[0] = int(nk)
    X = X.reshape(shape)
    if axis != 0:
        X = np.swapaxes(X, axis, 0)
    return X


def dft2(x, r, c, nk, nl):
    """
    Irregularly sampled 2D dft by projecting into sines/cosines. Vectorized.
    :param x: vector or 2d matrix of shape (nrc, nt)
    :param r: vector (nrc) of normalized positions along the k dimension (axis 0)
    :param c: vector (nrc) of normalized positions along the l dimension (axis 1)
    :param nk: output size along axis 0
    :param nl: output size along axis 1
    :return: Matrix X (nk, nl, nt)
    """
    # it would be interesting to compare performance with numba straight loops (easier to write)
    # GPU/C implementation should implement straight loops
    nt = x.shape[-1]
    k, h = [v.flatten() for v in np.meshgrid(np.arange(nk), np.arange(nl), indexing='ij')]
    # exp has dimension (kh, rc)
    exp = np.exp(- 1j * 2 * np.pi * (r[np.newaxis] * k[:, np.newaxis] +
                                     c[np.newaxis] * h[:, np.newaxis]))
    return np.matmul(exp, x).reshape((nk, nl, nt))

import numpy as np


def fscale(ns, si=1, half_sided=False):
    """
    numpy.fft.fftfreq returns Nyquist as a negative frequency so we propose this instead
    :param ns: number of samples
    :param si: sampling interval in seconds
    :param half_sided: if True, returns only positive frequencies
    :return: fscale: numpy vector containing frequencies in Hertz
    """
    fsc = np.arange(0, np.floor(ns / 2) + 1) / ns / si  # sample the frequency scale
    if half_sided:
        return fsc
    else:
        return np.concatenate((fsc, -fsc[slice(-2 + (ns % 2), 0, -1)]), axis=0)


def freduce(x):
    """
    Reduces a spectrum to positive frequencies only
    Works on the last dimension (contiguous in c-stored array)
    :param x: numpy.ndarray
    :return: numpy.ndarray
    """
    siz = list(x.shape)
    siz[-1] = int(np.floor(siz[-1] / 2 + 1))
    return x[..., :siz[-1]]


def fexpand(x, ns=1):
    """
    Reconstructs full spectrum from positive frequencies
    Works on the last dimension (contiguous in c-stored array)
    :param x: numpy.ndarray
    :return: numpy.ndarray
    """
    dec = int(ns % 2) * 2 - 1
    xcomp = np.conj(np.flip(x[..., 1:x.shape[-1] + dec], axis=x.ndim - 1))
    return np.concatenate((x, xcomp), axis=x.ndim - 1)


def lp(ts, si, b):
    """
    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 2 elements vector or list
    :return: filtered time serie
    """
    # TODO: multidimensional support using broadcast when needed
    ns = ts.shape[-1]
    f = fscale(ns, si=si, half_sided=True)
    filc = ((f <= b[0]).astype(float) +
            np.bitwise_and(f > b[0], f < b[1]).astype(float) *
            (0.5 * (1 + np.sin(np.pi * (f - ((b[0] + b[1]) / 2)) /
             (b[0] - b[1])))))
    return np.real(np.fft.ifft(np.fft.fft(ts) * fexpand(filc, ns)))

"""
Window generator, front detections, rms
"""
import numpy as np
from scipy import interpolate
from ibllib.misc import print_progress


def sync_timestamps(tsa, tsb, tbin=0.1, return_indices=False):
    """
    Sync two arrays of time stamps
    :param tsa: vector of timestamps
    :param tsb: vector of timestamps
    :param tbin: time bin length
    :param return_indices (bool), if True returns 2 sets of indices for tsa and tsb with
    identified matches
    :return:
     function: interpolation function such as fnc(tsa) = tsb
     float: drift in ppm
     numpy array: of indices ia
     numpy array: of indices ib
    """

    def _interp_fcn(tsa, tsb, ib):
        # now compute the bpod/fpga drift and precise time shift
        drift_ppm = np.polyfit(tsa[ib >= 0], tsb[ib[ib >= 0]] - tsa[ib >= 0], 1)[0] * 1e6
        fcn_a2b = interpolate.interp1d(tsa[ib >= 0], tsb[ib[ib >= 0]], fill_value="extrapolate")
        return fcn_a2b, drift_ppm

    # assert sorted inputs
    tmin = np.min([np.min(tsa), np.min(tsb)])
    tmax = np.max([np.max(tsa), np.max(tsb)])
    # brute force correlation to get an estimate of the delta_t between series
    x = np.zeros(int(np.ceil(tmax - tmin) / tbin))
    y = np.zeros_like(x)
    x[np.int32(np.floor((tsa - tmin) / tbin))] = 1
    y[np.int32(np.floor((tsb - tmin) / tbin))] = 1
    delta_t = (parabolic_max(np.correlate(x, y, mode='full'))[0] - x.shape[0] + 1) * tbin

    # do a first assignment at a DT threshold
    ib = np.zeros(tsa.shape, dtype=np.int32) - 1
    threshold = tbin
    for m in np.arange(tsa.shape[0]):
        dt = np.abs(tsa[m] - delta_t - tsb)
        inds = np.where(dt < threshold)[0]
        if inds.size == 1:
            ib[m] = inds[0]
        elif inds.size > 1:
            candidates = inds[~np.isin(inds, ib[:m])]
            if candidates.size == 1:
                ib[m] = candidates[0]
            elif candidates.size > 1:
                ib[m] = inds[np.argmin(dt[inds])]

    fcn_a2b, _ = _interp_fcn(tsa, tsb, ib)
    # do a second assignment - this time a full matrix of candidate matches is computed
    # the most obvious matches are assigned first and then one by one
    iamiss = np.where(ib < 0)[0]
    ibmiss = np.setxor1d(np.arange(tsb.size), ib[ib >= 0])
    dt = np.abs(fcn_a2b(tsa[iamiss]) - tsb[ibmiss][:, np.newaxis])
    dt[dt > tbin] = np.nan
    while ~np.all(np.isnan(dt)):
        _b, _a = np.unravel_index(np.nanargmin(dt), dt.shape)
        ib[iamiss[_a]] = ibmiss[_b]
        dt[:, _a] = np.nan
        dt[_b, :] = np.nan
    fcn_a2b, drift_ppm = _interp_fcn(tsa, tsb, ib)

    if return_indices:
        return fcn_a2b, drift_ppm, np.where(ib >= 0)[0], ib[ib >= 0]
    else:
        return fcn_a2b, drift_ppm


def parabolic_max(x):
    """
    Maximum picking with parabolic interpolation around the maxima
    :param x: 1d or 2d array
    :return: interpolated max index, interpolated max
    """
    # for 2D arrays, operate along the last dimension
    ns = x.shape[-1]
    axis = -1
    imax = np.argmax(x, axis=axis)

    if x.ndim == 1:
        v010 = x[np.maximum(np.minimum(imax + np.array([-1, 0, 1]), ns - 1), 0)]
        v010 = v010[:, np.newaxis]
    else:
        v010 = np.vstack((x[..., np.arange(x.shape[0]), np.maximum(imax - 1, 0)],
                          x[..., np.arange(x.shape[0]), imax],
                          x[..., np.arange(x.shape[0]), np.minimum(imax + 1, ns - 1)]))
    poly = np.matmul(.5 * np.array([[1, -2, 1], [-1, 0, 1], [0, 2, 0]]), v010)
    ipeak = - poly[1] / (poly[0] + np.double(poly[0] == 0)) / 2
    maxi = poly[2] + ipeak * poly[1] + ipeak ** 2. * poly[0]
    ipeak += imax
    # handle edges
    iedges = np.logical_or(imax == 0, imax == ns - 1)
    if x.ndim == 1:
        maxi = v010[1, 0] if iedges else maxi[0]
        ipeak = imax if iedges else ipeak[0]
    else:
        maxi[iedges] = v010[1, iedges]
        ipeak[iedges] = imax[iedges]
    return ipeak, maxi


def _fcn_extrap(x, f, bounds):
    """
    Extrapolates a flat value before and after bounds
    x: array to be filtered
    f: function to be applied between bounds (cf. fcn_cosine below)
    bounds: 2 elements list or np.array
    """
    y = f(x)
    y[x < bounds[0]] = f(bounds[0])
    y[x > bounds[1]] = f(bounds[1])
    return y


def fcn_cosine(bounds):
    """
    Returns a soft thresholding function with a cosine taper:
    values <= bounds[0]: values
    values < bounds[0] < bounds[1] : cosine taper
    values < bounds[1]: bounds[1]
    :param bounds:
    :return: lambda function
    """
    def _cos(x):
        return (1 - np.cos((x - bounds[0]) / (bounds[1] - bounds[0]) * np.pi)) / 2
    func = lambda x: _fcn_extrap(x, _cos, bounds)  # noqa
    return func


def fronts(x, axis=-1, step=1):
    """
    Detects Rising and Falling edges of a voltage signal, returns indices and

    :param x: array on which to compute RMS
    :param axis: (optional, -1) negative value
    :param step: (optional, 1) value of the step to detect
    :return: numpy array of indices, numpy array of rises (1) and falls (-1)
    """
    d = np.diff(x, axis=axis)
    ind = np.array(np.where(np.abs(d) >= step))
    sign = d[tuple(ind)]
    ind[axis] += 1
    if len(ind) == 1:
        return ind[0], sign
    else:
        return ind, sign


def falls(x, axis=-1, step=-1):
    """
    Detects Falling edges of a voltage signal, returns indices

    :param x: array on which to compute RMS
    :param axis: (optional, -1) negative value
    :param step: (optional, -1) value of the step to detect
    :return: numpy array
    """
    return rises(-x, axis=axis, step=-step)


def rises(x, axis=-1, step=1):
    """
    Detect Rising edges of a voltage signal, returns indices

    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :param step: (optional, 1) amplitude of the step to detect
    :return: numpy array
    """
    ind = np.array(np.where(np.diff(x, axis=axis) >= step))
    ind[axis] += 1
    if len(ind) == 1:
        return ind[0]
    else:
        return ind


def rms(x, axis=-1):
    """
    Root mean square of array along axis

    :param x: array on which to compute RMS
    :param axis: (optional, -1)
    :return: numpy array
    """
    return np.sqrt(np.mean(x ** 2, axis=axis))


class WindowGenerator(object):
    """
    `wg = WindowGenerator(ns, nswin, overlap)`

    Provide sliding windows indices generator for signal processing applications.
    For straightforward spectrogram / periodogram implementation, prefer scipy methods !

    Example of implementations in test_dsp.py.
    """
    def __init__(self, ns, nswin, overlap):
        """
        :param ns: number of sample of the signal along the direction to be windowed
        :param nswin: number of samples of the window
        :return: dsp.WindowGenerator object:
        """
        self.ns = int(ns)
        self.nswin = int(nswin)
        self.overlap = int(overlap)
        self.nwin = int(np.ceil(float(ns - nswin) / float(nswin - overlap))) + 1
        self.iw = None

    @property
    def firstlast(self):
        """
        Generator that yields first and last index of windows

        :return: tuple of [first_index, last_index] of the window
        """
        self.iw = 0
        first = 0
        while True:
            last = first + self.nswin
            last = min(last, self.ns)
            yield (first, last)
            if last == self.ns:
                break
            first += self.nswin - self.overlap
            self.iw += 1

    @property
    def slice(self):
        """
        Generator that yields slices of windows

        :return: a slice of the window
        """
        for first, last in self.firstlast:
            yield slice(first, last)

    def slice_array(self, sig, axis=-1):
        """
        Provided an array or sliceable object, generator that yields
        slices corresponding to windows. Especially useful when working on memmpaps

        :param sig: array
        :param axis: (optional, -1) dimension along which to provide the slice
        :return: array slice Generator
        """
        for first, last in self.firstlast:
            yield np.take(sig, np.arange(first, last), axis=axis)

    def tscale(self, fs):
        """
        Returns the time scale associated with Window slicing (middle of window)
        :param fs: sampling frequency (Hz)
        :return: time axis scale
        """
        return np.array([(first + (last - first - 1) / 2) / fs for first, last in self.firstlast])

    def print_progress(self, desc=''):
        """
        Prints progress using a terminal progress bar
        """
        print_progress(self.iw, self.nwin, prefix=desc)

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import ibllib.dsp.fourier as ft


def lp(ts, fac, pad=0.2):
    """
    Smooth the data in frequency domain (assumes a uniform sampling rate), using edge padding

    ibllib.dsp.smooth.lp(ts, [.1, .15])
    :param ts: input signal to be smoothed
    :param fac: 2 element vector of the frequency edges relative to Nyquist: [0.15, 0.2] keeps
    everything up to 15% of the full band tapering down to 20%
    :param pad: padding on the edges of the time serie, between 0 and 1 (0.2 means 20% of the size)
    :return: smoothed time series
    """
    # keep at least two periods for the padding
    lpad = int(np.ceil(ts.shape[0] * pad))
    ts_ = np.pad(ts, lpad, mode='edge')
    ts_ = ft.lp(ts_, 1, np.array(fac) / 2)
    return ts_[lpad:-lpad]


def rolling_window(x, window_len=11, window='blackman'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    :param x: The input signal
    :type x: list or numpy.array
    :param window_len: The dimension of the smoothing window,
                       should be an **odd** integer, defaults to 11
    :type window_len: int, optional
    :param window: The type of window from ['flat', 'hanning', 'hamming',
                   'bartlett', 'blackman']
                   flat window will produce a moving average smoothing,
                   defaults to 'blackman'
    :type window: str, optional
    :raises ValueError: Smooth only accepts 1 dimension arrays.
    :raises ValueError: Input vector needs to be bigger than window size.
    :raises ValueError: Window is not one of 'flat', 'hanning', 'hamming',
                        'bartlett', 'blackman'
    :return: Smoothed array
    :rtype: numpy.array
    """
    # **NOTE:** length(output) != length(input), to correct this:
    # return y[(window_len/2-1):-(window_len/2)] instead of just y.
    if isinstance(x, list):
        x = np.array(x)

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is not one of 'flat', 'hanning', 'hamming',\
'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[round((window_len / 2 - 1)):round(-(window_len / 2))]


def non_uniform_savgol(x, y, window, polynom):
    """Applies a Savitzky-Golay filter to y with non-uniform spacing as defined in x.
    This is based on
    https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
    The borders are interpolated like scipy.signal.savgol_filter would do
    https://dsp.stackexchange.com/a/64313
    Parameters
    ----------
    x : array_like
        List of floats representing the x values of the data
    y : array_like
        List of floats representing the y values. Must have same length as x
    window : int (odd)
        Window length of datapoints. Must be odd and smaller than x
    polynom : int
        The order of polynom used. Must be smaller than the window size
    Returns
    -------
    np.array
        The smoothed y values
    """

    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')
    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')
    if type(window) is not int:
        raise TypeError('"window" must be an integer')
    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')
    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')
    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)
        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)
        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)
        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


def smooth_interpolate_savgol(signal, window=31, order=3, interp_kind='cubic'):
    """Run savitzy-golay filter on signal, interpolate through nan points.

    Parameters
    ----------
    signal : np.ndarray
        original noisy signal of shape (t,), may contain nans
    window : int
        window of polynomial fit for savitzy-golay filter
    order : int
        order of polynomial for savitzy-golay filter
    interp_kind : str
        type of interpolation for nans, e.g. 'linear', 'quadratic', 'cubic'
    Returns
    -------
    np.array
        smoothed, interpolated signal for each time point, shape (t,)
    """

    signal_noisy_w_nans = np.copy(signal)
    timestamps = np.arange(signal_noisy_w_nans.shape[0])
    good_idxs = np.where(~np.isnan(signal_noisy_w_nans))[0]
    # perform savitzky-golay filtering on non-nan points
    signal_smooth_nonans = non_uniform_savgol(
        timestamps[good_idxs], signal_noisy_w_nans[good_idxs], window=window, polynom=order)
    signal_smooth_w_nans = np.copy(signal_noisy_w_nans)
    signal_smooth_w_nans[good_idxs] = signal_smooth_nonans
    # interpolate nan points
    interpolater = interp1d(
        timestamps[good_idxs], signal_smooth_nonans, kind=interp_kind, fill_value='extrapolate')
    signal = interpolater(timestamps)

    return signal


def smooth_demo():

    t = np.linspace(-4, 4, 100)
    x = np.sin(t)
    xn = x + np.random.randn(len(t)) * 0.1

    ws = 31

    plt.subplot(211)
    plt.plot(np.ones(ws))

    windows = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    for w in windows[1:]:
        eval('plt.plot(np.' + w + '(ws) )')

    plt.axis([0, 30, 0, 1.1])

    plt.legend(windows)
    plt.title("The smoothing windows")
    plt.subplot(212)
    plt.plot(x)
    plt.plot(xn)
    for w in windows:
        plt.plot(rolling_window(xn, 10, w))
    lst = ['original signal', 'signal with noise']
    lst.extend(windows)

    plt.legend(lst)
    plt.title("Smoothing a noisy signal")
    plt.ion()


if __name__ == '__main__':
    smooth_demo()

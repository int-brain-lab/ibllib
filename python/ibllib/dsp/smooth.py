import numpy as np
import matplotlib.pyplot as plt


def rolling_window(x, window_len=11, window='blackman'):
    """
    Smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

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

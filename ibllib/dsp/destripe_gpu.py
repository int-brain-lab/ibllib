from math import pi

from scipy.signal import butter

from .filter_gpu import sosfiltfilt_gpu
from .fourier import channel_shift
from .voltage import _get_destripe_parameters, interpolate_bad_channels, kfilt


def destripe_array(data, fs=30000, fshigh=300., taper_size=64, sample_shifts=None, channel_labels=None,
                   channel_xcoords=None, channel_ycoords=None):
    """
    Applies de-striping to a cupy array
    :param data: float32 cupy array, shape (n_channels, n_times)
    :param fs: Int, sampling rate
    :param fshigh: Float, high pass frequency
    :param taper_size: Int, length of cosine taper applied either side of data
    :param sample_shifts: Time shifts of channels, array with shape (n_channels)
    :param channel_labels: cupy array, shape (n_channels)
                            0:ok, 1:dead, 2:high noise, 3:outside of the brain
    :param channel_xcoords: cupy array, shape (n_channels)
    :param channel_ycoords: cupy array, shape (n_channels)
    :return: destriped data, cupy array, shape (n_channels, n_times)
    """

    import cupy as cp

    # parameters for filter
    sos = get_sos(fs, fshigh)

    # apply tapers
    if taper_size is not None and taper_size > 0:
        taper = cp.sin(cp.linspace(0, pi, taper_size*2))
        data[:, :taper_size] *= taper[:taper_size]
        data[:, -taper_size:] *= taper[taper_size:]

    # butterworth filter along time axis
    data = sosfiltfilt_gpu(sos, data)

    # align channels if the time shifts are provided
    if sample_shifts is not None:
        sample_shifts = cp.array(sample_shifts, dtype='float32')
        data = channel_shift(data, sample_shifts)

    # apply spatial filter
    #TODO: Currently using default settings, should allow user to change this in function arguments
    kfilt_kwargs = _get_destripe_parameters(fs, None, None, True)[1]

    if channel_labels is not None:
        data = interpolate_bad_channels(data, channel_labels, channel_xcoords, channel_ycoords, gpu=True)
        inside_brain = cp.where(channel_labels != 3)[0]
        data[inside_brain, :] = kfilt(data[inside_brain, :], gpu=True, **kfilt_kwargs)  # apply the k-filter / CAR
    else:
        print(data.dtype)
        data = kfilt(data, gpu=True, **kfilt_kwargs)  # apply the k-filter / CAR

    return data




def get_sos(fs, fshigh, fslow=None):
    """
    Get second-order sections for the butterworth filter
    :param fs: Int, sampling frequency
    :param fshigh: Int, high pass filter frequency
    :param fslow: Int, low pass filter frequency
    :return: sos, second-order sections
    """
    if fslow and fslow < fs / 2:
        return butter(3, (2 * fshigh / fs, 2 * fslow / fs), 'bandpass', output='sos')
    else:
        return butter(3, 2 * fshigh / fs, 'high', output='sos')


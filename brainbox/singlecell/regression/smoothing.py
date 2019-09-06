import numpy as np
from scipy.stats import norm


def times2frames(times, frame_length, initial_offset=0):
    """
    Function to map continuous times of spikes onto discrete frames
    :param times: time stamps of spikes
    :param frame_length: time length of a single frame
    :param initial_offset: start time for the first frame
    :return: integer frames where signals occur
    """
    if initial_offset > times[0]:
        raise ValueError("Initial_offset is too high")
    if type(times) != np.ndarray:
        times = np.array(times)
    return ((times - initial_offset) / frame_length).astype(int)


def gaussian_window(window_length, sigma=1):
    """
    Function to generate a Gaussian window for smoothing
    :param window_length: length of the window, must be odd for symmetric window
    :param sigma: standard deviation of the gaussian curve
    :return: Gaussian window as a numpy array for convolution
    """
    return norm.pdf(np.arange(-1 * (window_length - 1) // 2, (window_length + 1) // 2),
                   scale=sigma)


def square_window(window_length):
    """
    Square window, just sum up all the spikes in a bin
    :param window_length: length of the window, must be odd for symmetric window
    :return: Sqare window as a numpy array for convolution
    """
    return np.ones(window_length)


def frame_smoothing(frames, trial_length, window=None, window_length=None,
                    window_type=None, sigma=1):
    """
    Takes discrete frames and outputs the smoothed signal
    :param frames: Frames where signals occurred
    :param trial_length: Length of trial (in frames)
    :param window: Window for smoothing spikes
    :param window_length: Length of smoothing window, only required if using an inbuilt window type
    :param window_type: Options are Gaussian
    :param sigma: Standard deviation of Gaussian window
    :return:
    """

    if window is None:
        if window_type is None:
            raise ValueError('Must input either a custom window or a specified type')
        elif window_type.lower() == 'gaussian':
            window = gaussian_window(window_length=window_length, sigma=sigma)
        elif window_type.lower() == 'square':
            window = square_window(window_length=window_length)
        else:
            raise ValueError('Window type not recognised')
    else:
        window_length = len(window)
    smoothed_signal = np.zeros(trial_length + (window_length + 1) // 2)
    for frame in frames:
        smoothed_signal[frame - window_length // 2: frame + (window_length + 1) // 2] += window
    return smoothed_signal


def smoothing_times(times, frame_length, trial_duration, window=None, window_type=None,
                    window_length=None, sigma=1, initial_offset=0):
    """

    :param times: Times of spike events
    :param frame_length: Time length of a single frame
    :param trial_duration:
    :param window: Window for smoothing
    :param window_type: Options are Gaussian
    :param window_length: Length of smoothing window (only required if using an inbiult type)
    :param sigma: Standard deviation of Gaussian window
    :param initial_offset: Start time of the first frame
    :return:
    """
    frames = times2frames(times, frame_length, initial_offset)
    trial_length = ((trial_duration - initial_offset) / frame_length).astype(int)
    smoothed_signal = frame_smoothing(frames, trial_length,window, window_length,
                                      window_type, sigma)
    return smoothed_signal

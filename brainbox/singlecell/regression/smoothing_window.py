import numpy as np
from scipy.stats import norm

def times2frames(times, frame_length, initial_offset=0):
    # takes a set of times in a numpy and matches them to discrete frames of a fixed length
    if initial_offset > times[0]:
        raise ValueError("initial_offset is too high")
    if type(times) != np.ndarray:
        times = np.array(times)
    return ((times - initial_offset) / frame_length).astype(int)

def gaussian_window(window_length, sigma=1):
    # generates gaussian window for smoothing
    return norm.pdf(np.arange(-1 * window_length // 2, (window_length + 1) // 2), scale = sigma)

def frame_smoothing(frames, window, window_length, window_type, sigma=1):
    # takes frames of signal events and returns the smoothed signal
    if window_type.lower() == 'gaussian':
        window = gaussian_window(window_length = window_length, sigma = sigma)
    smoothed_signal = np.zeros(frames[-1] + (window_length + 1) // 2)
    for frame in frames:
        smoothed_signal[frame - window_length // 2: frame + (window_length+1) // 2] += window
    return smoothed_signal

def smoothing_times(times, frame_length, window, window_type, window_length, sigma=1, initial_offset=0):
    # wrapper function that handles everything
    frames = times2frames(times, frame_length, initial_offset)
    smoothed_signal = frame_smoothing(frames, window, window_length, window_type, sigma)
    return smoothed_signal

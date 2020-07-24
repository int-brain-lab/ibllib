"""
Set of functions to handle wheel data
"""
import numpy as np
from numpy import pi
import scipy.interpolate as interpolate
from scipy.signal import convolve, gaussian
from scipy.linalg import hankel
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from typing import TypeVar, Type, Sequence, Optional, Union
# from ibllib.io.extractors.ephys_fpga import WHEEL_TICKS  # FIXME Circular dependencies

__all__ = ['cm_to_deg',
           'cm_to_rad',
           'interpolate_position',
           'last_movement_onset',
           'movements',
           'samples_to_cm',
           'traces_by_trial',
           'velocity_smoothed',
           'within_ranges']

D = TypeVar('D', bound=np.generic)
Array = Union[np.ndarray, Sequence]

# Define some constants
ENC_RES = 1024 * 4  # Rotary encoder resolution, assumes X4 encoding
WHEEL_DIAMETER = 3.1 * 2  # Wheel diameter in cm


def interpolate_position(re_ts, re_pos, freq=1000, kind='linear', fill_gaps=None):
    """
    Return linearly interpolated wheel position.

    Parameters
    ----------
    re_ts : array_like
        Array of timestamps
    re_pos: array_like
        Array of unwrapped wheel positions
    freq : float
        frequency in Hz of the interpolation
    kind : {'linear', 'cubic'}
        Type of interpolation. Defaults to linear interpolation.
    fill_gaps : float
        Minimum gap length to fill. For gaps over this time (seconds),
        forward fill values before interpolation
    Returns
    -------
    yinterp : array
        Interpolated position
    t : array
        Timestamps of interpolated positions
    """
    t = np.arange(re_ts[0], re_ts[-1], 1 / freq)  # Evenly resample at frequency
    yinterp = interpolate.interp1d(re_ts, re_pos, kind=kind)(t)

    if fill_gaps:
        #  Find large gaps and forward fill @fixme This is inefficient
        gaps, = np.where(np.diff(re_ts) >= fill_gaps)

        for i in gaps:
            yinterp[(t >= re_ts[i]) & (t < re_ts[i + 1])] = re_pos[i]

    return yinterp, t


def velocity(re_ts, re_pos):
    """
    Compute wheel velocity from non-uniformly sampled wheel data. Returns the velocity
    at the same samples locations as the position through interpolation.

    Parameters
    ----------
    re_ts : array_like
        Array of timestamps
    re_pos: array_like
        Array of unwrapped wheel positions

    Returns
    -------
    np.ndarray
        numpy array of velocities
    """
    dp = np.diff(re_pos)
    dt = np.diff(re_ts)
    # Compute raw velocity
    vel = dp / dt
    # Compute velocity time scale
    tv = re_ts[:-1] + dt / 2
    # interpolate over original time scale
    if tv.size > 1:
        ifcn = interpolate.interp1d(tv, vel, fill_value="extrapolate")
        return ifcn(re_ts)


def velocity_smoothed(pos, freq, smooth_size=0.03):
    """
    Compute wheel velocity from uniformly sampled wheel data

    Parameters
    ----------
    pos : array_like
        Array of wheel positions
    smooth_size : float
        Size of Gaussian smoothing window in seconds
    freq : float
        Sampling frequency of the data

    Returns
    -------
    vel : np.ndarray
        Array of velocity values
    acc : np.ndarray
        Array of acceleration values
    """
    # Define our smoothing window with an area of 1 so the units won't be changed
    std_samps = np.round(smooth_size * freq)  # Standard deviation relative to sampling frequency
    N = std_samps * 6  # Number of points in the Gaussian covering +/-3 standard deviations
    gauss_std = (N - 1) / 6
    win = gaussian(N, gauss_std)
    win = win / win.sum()  # Normalize amplitude

    # Convolve and multiply by sampling frequency to restore original units
    vel = np.insert(convolve(np.diff(pos), win, mode='same'), 0, 0) * freq
    acc = np.insert(convolve(np.diff(vel), win, mode='same'), 0, 0) * freq

    return vel, acc


def last_movement_onset(t, vel, event_time):
    """
    Find the time at which movement started, given an event timestamp that occurred during the
    movement.  Movement start is defined as the first sample after the velocity has been zero
    for at least 50ms.  Wheel inputs should be evenly sampled.

    :param t: numpy array of wheel timestamps in seconds
    :param vel: numpy array of wheel velocities
    :param event_time: timestamp anywhere during movement of interest, e.g. peak velocity
    :return: timestamp of movement onset
    """
    # Look back from timestamp
    threshold = 50e-3
    mask = t < event_time
    times = t[mask]
    vel = vel[mask]
    t = None  # Initialize
    for i, t in enumerate(times[::-1]):
        i = times.size - i
        idx = np.min(np.where((t - times) < threshold))
        if np.max(np.abs(vel[idx:i])) < 0.5:
            break

    # Return timestamp
    return t


def movements(t, pos, freq=1000, pos_thresh=8, t_thresh=.2, min_gap=.1, pos_thresh_onset=1.5,
              min_dur=.05, make_plots=False):
    """
    Detect wheel movements.

    Parameters
    ----------
    t : array_like
        An array of evenly sampled wheel timestamps in absolute seconds
    pos : array_like
        An array of evenly sampled wheel positions
    freq : int
        The sampling rate of the wheel data
    pos_thresh : float
        The minimum required movement during the t_thresh window to be considered part of a
        movement
    t_thresh : float
        The time window over which to check whether the pos_thresh has been crossed
    min_gap : float
        The minimum time between one movement's offset and another movement's onset in order to be
        considered separate.  Movements with a gap smaller than this are 'stictched together'
    pos_thresh_onset : float
        A lower threshold for finding precise onset times.  The first position of each movement
        transition that is this much bigger than the starting position is considered the onset
    min_dur : float
        The minimum duration of a valid movement.  Detected movements shorter than this are ignored
    make_plots : boolean
        Plot trace of position and velocity, showing detected onsets and offsets

    Returns
    -------
    onsets : np.ndarray
        Timestamps of detected movement onsets
    offsets : np.ndarray
        Timestamps of detected movement offsets
    peak_amps : np.ndarray
        The absolute maximum amplitude of each detected movement, relative to onset position
    peak_vel_times : np.ndarray
        Timestamps of peak velocity for each detected movement
    """
    # Wheel position must be evenly sampled.
    dt = np.diff(t)
    assert np.all(np.abs(dt - dt.mean()) < 1e-10), 'Values not evenly sampled'

    # Convert the time threshold into number of samples given the sampling frequency
    t_thresh_samps = int(np.round(t_thresh * freq))
    max_disp = np.empty(t.size, dtype=float)  # initialize array of total wheel displacement

    # Calculate a Hankel matrix of size t_thresh_samps in batches.  This is effectively a
    # sliding window within which we look for changes in position greater than pos_thresh
    BATCH_SIZE = 10000  # do this in batches in order to keep memory usage reasonable
    c = 0  # index of 'window' position
    while True:
        i2proc = np.arange(BATCH_SIZE) + c
        i2proc = i2proc[i2proc < t.size]
        w2e = hankel(pos[i2proc], np.full(t_thresh_samps, np.nan))
        # Below is the total change in position for each window
        max_disp[i2proc] = np.nanmax(w2e, axis=1) - np.nanmin(w2e, axis=1)
        c += BATCH_SIZE - t_thresh_samps
        if i2proc[-1] == t.size - 1:
            break

    moving = max_disp > pos_thresh  # for each window is the change in position greater than
    # our threshold?
    moving = np.insert(moving, 0, False)  # First sample should always be not moving to ensure
    # we have an onset
    moving[-1] = False  # Likewise, ensure we always end on an offset

    onset_samps = np.where(~moving[:-1] & moving[1:])[0]
    offset_samps = np.where(moving[:-1] & ~moving[1:])[0]
    too_short = np.where((onset_samps[1:] - offset_samps[:-1]) / freq < min_gap)[0]
    for p in too_short:
        moving[offset_samps[p]:onset_samps[p + 1] + 1] = True

    onset_samps = np.where(~moving[:-1] & moving[1:])[0]
    onsets_disp_arr = np.empty((onset_samps.size, t_thresh_samps))
    c = 0
    cwt = 0
    while onset_samps.size != 0:
        i2proc = np.arange(BATCH_SIZE) + c
        icomm = np.intersect1d(i2proc[:-t_thresh_samps - 1], onset_samps, assume_unique=True)
        itpltz = np.intersect1d(i2proc[:-t_thresh_samps - 1], onset_samps,
                                return_indices=True, assume_unique=True)[1]
        i2proc = i2proc[i2proc < t.size]
        if icomm.size > 0:
            w2e = hankel(pos[i2proc], np.full(t_thresh_samps, np.nan))
            w2e = np.abs((w2e.T - w2e[:, 0]).T)
            onsets_disp_arr[cwt + np.arange(icomm.size), :] = w2e[itpltz, :]
            cwt += icomm.size
        c += BATCH_SIZE - t_thresh_samps
        if i2proc[-1] >= onset_samps[-1]:
            break

    has_onset = onsets_disp_arr > pos_thresh_onset
    A = np.argmin(np.fliplr(has_onset).T, axis=0)
    onset_lags = t_thresh_samps - A
    onset_samps = onset_samps + onset_lags - 1
    onsets = t[onset_samps]
    offset_samps = np.where(moving[:-1] & ~moving[1:])[0]
    offsets = t[offset_samps]

    durations = offsets - onsets
    too_short = durations < min_dur
    onset_samps = onset_samps[~too_short]
    onsets = onsets[~too_short]
    offset_samps = offset_samps[~too_short]
    offsets = offsets[~too_short]

    moveGaps = onsets[1:] - offsets[:-1]
    gap_too_small = moveGaps < min_gap
    if onsets.size > 0:
        onsets = onsets[np.insert(~gap_too_small, 0, True)]  # always keep first onset
        onset_samps = onset_samps[np.insert(~gap_too_small, 0, True)]
        offsets = offsets[np.append(~gap_too_small, True)]  # always keep last offset
        offset_samps = offset_samps[np.append(~gap_too_small, True)]

    # Calculate the peak amplitudes -
    # the maximum absolute value of the difference from the onset position
    peaks = (pos[m + np.abs(pos[m:n] - pos[m]).argmax()] - pos[m]
             for m, n in zip(onset_samps, offset_samps))
    peak_amps = np.fromiter(peaks, dtype=float, count=onsets.size)
    N = 10  # Number of points in the Gaussian
    STDEV = 1.8  # Equivalent to a width factor (alpha value) of 2.5
    gauss = gaussian(N, STDEV)  # A 10-point Gaussian window of a given s.d.
    vel = convolve(np.diff(np.insert(pos, 0, 0)), gauss, mode='same')
    # For each movement period, find the timestamp where the absolute velocity was greatest
    peaks = (t[m + np.abs(vel[m:n]).argmax()] for m, n in zip(onset_samps, offset_samps))
    peak_vel_times = np.fromiter(peaks, dtype=float, count=onsets.size)

    if make_plots:
        fig, axes = plt.subplots(nrows=2, sharex='all')
        indices = np.sort(np.hstack((onset_samps, offset_samps)))  # Points to split trace
        vel, acc = velocity_smoothed(pos, freq, 0.015)

        # Plot the wheel position and velocity
        for ax, y in zip(axes, (pos, vel)):
            ax.plot(onsets, y[onset_samps], 'go')
            ax.plot(offsets, y[offset_samps], 'bo')

            t_split = np.split(np.vstack((t, y)).T, indices, axis=0)
            ax.add_collection(LineCollection(t_split[1::2], colors='r'))  # Moving
            ax.add_collection(LineCollection(t_split[0::2], colors='k'))  # Not moving

        axes[1].autoscale()  # rescale after adding line collections
        axes[0].autoscale()
        axes[0].set_ylabel('position')
        axes[1].set_ylabel('velocity')
        axes[1].set_xlabel('time')
        axes[0].legend(['onsets', 'offsets', 'in movement'])
        plt.show()

    return onsets, offsets, peak_amps, peak_vel_times


def cm_to_deg(positions, wheel_diameter=WHEEL_DIAMETER):
    """
    Convert wheel position to degrees turned.  This may be useful for e.g. calculating velocity
    in revolutions per second
    :param positions: array of wheel positions in cm
    :param wheel_diameter: the diameter of the wheel in cm
    :return: array of wheel positions in degrees turned

    # Example: Convert linear cm to degrees
    >>> cm_to_deg(3.142 * WHEEL_DIAMETER)
    360.04667846020925

    # Example: Get positions in deg from cm for 5cm diameter wheel
    >>> import numpy as np
    >>> cm_to_deg(np.array([0.0270526 , 0.04057891, 0.05410521, 0.06763151]), wheel_diameter=5)
    array([0.61999992, 0.93000011, 1.24000007, 1.55000003])
    """
    return positions / (wheel_diameter * pi) * 360


def cm_to_rad(positions, wheel_diameter=WHEEL_DIAMETER):
    """
    Convert wheel position to radians.  This may be useful for e.g. calculating angular velocity.
    :param positions: array of wheel positions in cm
    :param wheel_diameter: the diameter of the wheel in cm
    :return: array of wheel angle in radians

    # Example: Convert linear cm to radians
    >>> cm_to_rad(1)
    0.3225806451612903

    # Example: Get positions in rad from cm for 5cm diameter wheel
    >>> import numpy as np
    >>> cm_to_rad(np.array([0.0270526 , 0.04057891, 0.05410521, 0.06763151]), wheel_diameter=5)
    array([0.01082104, 0.01623156, 0.02164208, 0.0270526 ])
    """
    return positions * (2 / wheel_diameter)


def samples_to_cm(positions, wheel_diameter=WHEEL_DIAMETER, resolution=ENC_RES):
    """
    Convert wheel position samples to cm linear displacement.  This may be useful for
    inter-converting threshold units
    :param positions: array of wheel positions in sample counts
    :param wheel_diameter: the diameter of the wheel in cm
    :param resolution: resolution of the rotary encoder
    :return: array of wheel angle in radians

    # Example: Get resolution in linear cm
    >>> samples_to_cm(1)
    0.004755340442445488

    # Example: Get positions in linear cm for 4X, 360 ppr encoder
    >>> import numpy as np
    >>> samples_to_cm(np.array([2, 3, 4, 5, 6, 7, 6, 5, 4]), resolution=360*4)
    array([0.0270526 , 0.04057891, 0.05410521, 0.06763151, 0.08115781,
           0.09468411, 0.08115781, 0.06763151, 0.05410521])
    """
    return positions / resolution * pi * wheel_diameter


def direction_changes(t, vel, intervals):
    """
    Find the direction changes for the given movement intervals.

    Parameters
    ----------
    t : array_like
        An array of evenly sampled wheel timestamps in absolute seconds
    vel : array_like
        An array of evenly sampled wheel positions
    intervals : array_like
        An n-by-2 array of wheel movement intervals

    Returns
    ----------
    times : iterable
        A list of numpy arrays of direction change timestamps, one array per interval
    indices : iterable
        A list of numpy arrays containing indices of direction changes; the size of times
    """
    indices = []
    times = []
    chg = np.insert(np.diff(np.sign(vel)) != 0, 0, 0)

    for on, off in intervals.reshape(-1, 2):
        mask = np.logical_and(t > on, t < off)
        ind, = np.where(np.logical_and(mask, chg))
        times.append(t[ind])
        indices.append(ind)

    return times, indices


def within_ranges(x: np.ndarray, ranges: Array, labels: Optional[Array] = None,
                  mode: str = 'vector', dtype: Type[D] = 'int8') -> np.ndarray:
    """
    Detects which points of the input vector lie within one of the ranges specified in the ranges.
    Returns an array the size of x with a 1 if the corresponding point is within a range.

    The function uses a stable sort algorithm (timsort) to find the edges within the input array.
    Edge behaviour is inclusive.

    Ranges are [(start0, stop0), (start1, stop1), etc.] or n-by-2 numpy array.
    The ranges may be optionally assigned a row in 'matrix' mode or a numerical label in 'vector'
    mode. Labels must have a length of n.  Overlapping ranges have a value that is the sum of the
    relevant range labels (ones in 'matrix' mode).

    If mode is 'matrix' (default) it will give a matrix output where each range is assigned a
    particular row index with 1 if the point belongs to that range label.  Multiple ranges can be
    assigned to a particular row, e.g. [0, 0,1] would give a 2-by-N matrix with the first two
    ranges in the first row.  Points within more than one range are given a value > 1
    If mode is 'vector' it will give a vector, specifying the range of each point.

    Parameters
    ----------
    x : array_like
        An array whose points are tested against the ranges.  multi-dimensional arrays are
        flattened to 1D
    ranges : array_like
        A list of tuples or N-by-2 array of ranges to test, where N is the number of ranges,
        i.e. [[start0, stop0],
        [start1, stop1]]
    labels : vector, list
        If mode is 'vector'; a list of integer labels to demarcate which points lie within each
        range.  In 'matrix' mode; a list of column indices (ranges can share indices).
        The number of labels should match the number of ranges.  If None, ones are used for all
        ranges.
    mode : {'matrix', 'vector'}
        The type of output to return.  If 'matrix' (default), an N-by-M matrix is returned where N
        is the size of x and M corresponds to the max index in labels, e.g. with labels=[0,1,2],
        the output matrix would have 3 columns.  If 'vector' a vector the size of x is returned.
    dtype : str, numeric or boolean type
        The data type of the returned array.  If type is bool, the labels in vector mode will be
        ignored.  Default is int8.


    Returns
    -------
    A vector of size like x where zeros indicate that the points do not lie within ranges (
    'vector' mode) or a matrix where out.shape[0] == x.size and out.shape[1] == max(labels) + 1.

    Examples
    -------
    # Assert that points in ranges are mutually exclusive
    np.all(within_ranges(x, ranges) <= 1)

    Tests
    -------
    >>> import numpy as np
    >>> within_ranges(np.arange(11), [(1, 2), (5, 8)])
    array([0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0], dtype=int8)
    >>> ranges = np.array([[1, 2], [5, 8]])
    >>> within_ranges(np.arange(10) + 1, ranges, labels=np.array([0,1]), mode='matrix')
    array([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 1, 1, 1, 0, 0]], dtype=int8)
    >>> within_ranges(np.arange(11), [(1,2), (5,8), (4,6)], labels=[0,1,1], mode='matrix')
    array([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 2, 2, 1, 1, 0, 0]], dtype=int8)
    >>> within_ranges(np.arange(10) + 1, ranges, np.array([3,1]), mode='vector')
    array([3, 3, 0, 0, 1, 1, 1, 1, 0, 0], dtype=int8)
    >>> within_ranges(np.arange(11), [(1,2), (5,8), (4,6)], dtype=bool)
    array([False,  True,  True, False,  True,  True,  True,  True,  True,
           False, False])
    """
    # Flatten
    x = x.ravel()

    # Ensure ranges are numpy
    ranges = np.array(ranges)

    # Get size info
    n_points = x.size
    n_ranges = ranges.shape[0]

    if labels is None:
        # In 'matrix' mode default row index is 0
        labels = np.zeros((n_ranges,), dtype='uint32')
        if mode == 'vector':  # Otherwise default numerical label is 1
            labels += 1
    assert len(labels) >= n_ranges, 'range labels do not match number of ranges'
    n_labels = np.unique(labels).size

    # If no ranges given, short circuit function and return zeros
    if n_ranges == 0:
        return np.zeros_like(x, dtype=dtype)

    # Check end comes after start in each case
    assert np.all(np.diff(ranges, axis=1) > 0), 'ranges ends must all be greater than starts'

    # Make array containing points, starts and finishes

    # This order means it will be inclusive
    to_sort = np.concatenate((ranges[:, 0], x, ranges[:, 1]))
    # worst case O(n*log(n)) but will be better than  this as most of the array is ordered;
    # memory overhead ~n/2
    idx = np.argsort(to_sort, kind='stable')

    # Make delta array containing 1 for every start and -1 for every stop
    # with one row for each range label
    if mode == 'matrix':
        delta_shape = (n_labels, n_points + 2 * n_ranges)
        delta = np.zeros(delta_shape, dtype='int8')

        delta[labels, np.arange(n_ranges)] = 1
        delta[labels, n_points + n_ranges + np.arange(n_ranges)] = -1

        # Arrange in order
        delta_sorted = delta[:, idx]

        # Take cumulative sums
        summed = np.cumsum(delta_sorted, axis=1)

        # Reorder back to original order
        reordered = np.zeros(delta_shape, dtype=dtype)
        reordered[:, idx] = summed.reshape(delta_shape[0], -1)
        return reordered[:, np.arange(n_ranges, n_points + n_ranges)]

    elif mode == 'vector':
        delta_shape = (n_points + 2 * n_ranges,)
        r_delta = np.zeros(delta_shape, dtype='int32')
        r_delta[np.arange(n_ranges)] = labels
        r_delta[n_points + n_ranges + np.arange(n_ranges)] = -labels

        # Arrange in order
        r_delta_sorted = r_delta[idx]

        # Take cumulative sum
        r_summed = np.cumsum(r_delta_sorted)

        # Reorder back to original
        r_reordered = np.zeros_like(r_summed, dtype=dtype)
        r_reordered[idx] = r_summed

        return r_reordered[np.arange(n_ranges, n_points + n_ranges)]
    else:
        raise ValueError('unknown mode type, options are "matrix" and "vector"')


def traces_by_trial(t, *args, start=None, end=None, separate=True):
    """
    Returns list of tuples of positions and velocity for samples between stimulus onset and
    feedback.
    :param t: numpy array of timestamps
    :param args: optional numpy arrays of the same length as timestamps, such as positions,
    velocities or accelerations
    :param start: start timestamp or array thereof
    :param end: end timestamp or array thereof
    :param separate: when True, the output is returned as tuples list of the form [(t, args[0],
    args[1]), ...], when False, the output is a list of n-by-m ndarrays where n = number of
    positional args and m = len(t)
    :return: list of sliced arrays where length == len(start)
    """
    if start is None:
        start = t[0]
    if end is None:
        end = t[-1]
    traces = np.stack((t, *args))
    assert len(start) == len(end), 'number of start timestamps must equal end timestamps'

    def to_mask(a, b):
        return np.logical_and(t > a, t < b)

    cuts = [traces[:, to_mask(s, e)] for s, e in zip(start, end)]
    return [(cuts[n][0, :], cuts[n][1, :]) for n in range(len(cuts))] if separate else cuts


if __name__ == "__main__":
    import doctest
    doctest.testmod()

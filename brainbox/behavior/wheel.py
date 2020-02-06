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

__all__ = ['cm_to_deg',
           'cm_to_rad',
           'convolve',
           'hankel',
           'interpolate_position',
           'last_movement_onset',
           'movements',
           'samples_to_cm',
           'traces_by_trial',
           'velocity',
           'velocity_smoothed', ]

# Define some constants
ENC_RES = 1024 * 4  # Rotary encoder resolution, assumes X4 encoding
WHEEL_DIAMETER = 3.1 * 2  # Wheel diameter in cm


def interpolate_position(re_ts, re_pos, freq=1000, kind='linear', fill_gaps=None):
    """
    Return linearly interpolated wheel position.
    :param re_ts: numpy array of timestamps
    :param re_pos: numpy array of unwrapped wheel positions
    :param freq: frequency in Hz of the interpolation
    :param kind: type of interpolation; 'linear' (default) or 'cubic'
    :param fill_gaps: for gaps over this time (seconds), forward fill values before interpolation
    :return: interpolated position and timestamps
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

    :param re_ts: numpy array of timestamps
    :param re_pos: numpy array of unwrapped wheel positions
    :return: numpy array of velocities
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
    :param pos: numpy array of wheel positions
    :param smooth_size: size of Gaussian smoothing window in seconds
    :param freq: sampling frequency of the data
    :return: tuple of velocity and acceleration values as numpy arrays
    """
    # Define our smoothing window with an area of 1 so the units won't be changed
    stdSamps = np.round(smooth_size * freq)  # Standard deviation relative to sampling frequency
    N = stdSamps * 6  # Number of points in the Gaussian
    gauss_std = (N - 1) / 6  # @fixme magic number everywhere!
    win = gaussian(N, gauss_std)
    win = win / win.sum()  # Normalize amplitude

    # Convolve and multiply by sampling frequency to restore original units
    vel = np.insert(convolve(np.diff(pos), win, mode='same'), 0, 0) * freq
    acc = np.insert(convolve(np.diff(vel), win, mode='same'), 0, 0) * freq

    return vel, acc


def last_movement_onset(t, vel, event_time):
    """
    Find the time at which movement started, given an event timestamp that accured during the
    movement.  Movement start is defined as the first sample after the velocity has been zero
    for at least 50ms
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
    Return linearly interpolated wheel position.
    :param t: An array of wheel timestamps in absolute seconds
    :param pos: An array of evenly sampled wheel positions
    :param freq: The sampling rate for linear interpolation
    :param pos_thresh: The minimum required movement during the t_thresh window to be considered
    part of a movement
    :param t_thresh: The time window over which to check whether the pos_thresh has been crossed
    :param min_gap: The minimum time between one movement's offset and another movement's onset
    in order to be considered separate.  Movements with a gap smaller than this are 'stictched
    together'
    :param pos_thresh_onset: A lower threshold for finding precise onset times.  The first
    position of each movement transition that is this much bigger than the starting position is
    considered the onset
    :param min_dur: The minimum duration of a valid movement.  Detected movements shorter than
    this are ignored
    :param make_plots: plot trace of position and velocity, showing detected onsets and offsets
    :return: Tuple of onset and offset times, movement amplitudes and peak velocity times
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
    b, a = np.where(~np.fliplr(has_onset).T)
    # A = np.asarray([np.min(b[a == i]) for i in np.unique(a)])
    # A = np.vectorize(lambda i: np.min(b[a == i]))(np.unique(a))
    first = (np.min(b[a == i]) for i in np.unique(a))
    A = np.fromiter(first, dtype=int, count=np.unique(a).size)
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

    move_amps = pos[offset_samps] - pos[onset_samps]
    N = 10  # Number of points in the Gaussian
    STDEV = 1.8  # Equivalent to a width factor (alpha value) of 2.5
    gauss = gaussian(N, STDEV)  # A 10-point Gaussian window of a given s.d.
    vel = convolve(np.diff(np.insert(pos, 0, 0)), gauss, mode='same')
    # For each movement period, find the timestamp where the absolute velocity was greatest
    peaks = (t[m + np.abs(vel[m:n]).argmax()] for m, n in zip(onset_samps, offset_samps))
    peak_vel_times = np.fromiter(peaks, dtype=float, count=onsets.size)

    if make_plots:
        fig, axes = plt.subplots(nrows=2, sharex='all')
        indicies = np.sort(np.hstack((onset_samps, offset_samps)))  # Points to split trace
        vel, acc = velocity_smoothed(pos, freq, 0.015)

        # Plot the wheel position and velocity
        for ax, y in zip(axes, (pos, vel)):
            ax.plot(onsets, y[onset_samps], 'go')
            ax.plot(offsets, y[offset_samps], 'bo')

            t_split = np.split(np.vstack((t, y)).T, indicies, axis=0)
            ax.add_collection(LineCollection(t_split[1::2], colors='r'))  # Moving
            ax.add_collection(LineCollection(t_split[0::2], colors='k'))  # Not moving

        axes[0].set_ylabel('position')
        axes[1].set_ylabel('velocity')
        axes[1].set_xlabel('time')
        axes[0].legend(['onsets', 'offsets', 'in movement'])
        plt.show()

    return onsets, offsets, move_amps, peak_vel_times


def cm_to_deg(positions, wheel_diameter=WHEEL_DIAMETER):
    """
    Convert wheel position to degrees turned.  This may be useful for e.g. calculating velocity
    in revolutions per second
    :param positions: array of wheel positions in cm
    :param wheel_diameter: the diameter of the wheel in cm
    :return: array of wheel positions in degrees turned
    """
    return positions / (wheel_diameter * pi) * 360


def cm_to_rad(positions, wheel_diameter=WHEEL_DIAMETER):
    """
    Convert wheel position to radians.  This may be useful for e.g. calculating angular velocity.
    :param positions: array of wheel positions in cm
    :param wheel_diameter: the diameter of the wheel in cm
    :return: array of wheel angle in radians
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
    """
    return positions / resolution * pi * wheel_diameter


def traces_by_trial(t, pos, trials, start='stimOn_times', end='feedback_times'):
    """
    Returns list of tuples of positions and velocity for samples between stimulus onset and
    feedback.
    :param t: numpy array of timestamps
    :param pos: numpy array of wheel positions (could also be velocities or accelerations)
    :param trials: dict of trials ALFs
    :param start: trails key to use as the start index for splitting
    :param end: trails key to use as the end index for splitting
    :return: list of traces between each start and end event
    """
    if start == 'intervals' and (end is None or end == 'interval'):
        start = trials['intervals'][:, 0]
        end = trials['intervals'][:, 1]
    traces = np.vstack((pos, t))

    def to_mask(a, b):
        (t > a) & (t < b)
    #  to_dict = lambda a, b, c: position: a
    cuts = [traces[:, to_mask(s, e)] for s, e in zip(trials[start], trials[end])]
    # ans = map(to_mask, zip(trials[start], trials[end]))
    # for s, e in zip(trials[start], trials[end]):
    #     mask = (wheel['times'] > s) & (wheel['times'] < e)
    #     cuts = traces[:, mask]

    return [(cuts[n][0, :], cuts[n][1, :], cuts[n][2, :]) for n in range(len(cuts))]

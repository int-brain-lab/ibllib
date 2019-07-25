"""
Set of functions to handle wheel data
"""
import numpy as np
import scipy.interpolate as interpolate


def velocity(re_ts, re_pos):
    """
    Compute wheel velocity from a non-uniformly sampled wheel data. Returns the velocity
    at the same samples locations than the position through interpolation.

    :param re_ts: numpy array of time stamps
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
    ifcn = interpolate.interp1d(tv, vel, fill_value="extrapolate")
    return ifcn(re_ts)

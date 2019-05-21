#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: nico
# @Date:   2018-02-13 18:16:02
"""
Created on Wed Jan 14 11:30:29 2015

@author: nico
"""
import numpy as np
from math import factorial


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Smoothes input signal with a Savitzky-Golay function.

    Uses a kernel of size "window_size" and a polinomial fit of order "order".

    :param y: [description]
    :type y: [type]
    :param window_size: [description]
    :type window_size: [type]
    :param order: [description]
    :type order: [type]
    :param deriv: [description], defaults to 0
    :param deriv: int, optional
    :param rate: [description], defaults to 1
    :param rate: int, optional
    :raises ValueError: [description]
    :raises TypeError: [description]
    :raises TypeError: [description]
    :return: [description]
    :rtype: [type]
    """
    y = np.array(y)

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(
        -half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

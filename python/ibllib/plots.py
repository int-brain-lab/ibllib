#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from ibllib.dsp import rms


def wiggle(w, fs=1, gain=0.71, color='k', ax=None, fill=True, linewidth=0.5, **kwargs):
    """
    Matplotlib display of wiggle traces

    :param w: 2D array (numpy array dimension nsamples, ntraces)
    :param fs: sampling frequency
    :param gain: display gain
    :param color: color of traces
    :param ax: matplotlib axes object
    :return: None
    """
    nech, ntr = w.shape
    tscale = np.arange(nech) / fs
    sf = gain / np.sqrt(rms(w.flatten()))

    def insert_zeros(trace):
        # Insert zero locations in data trace and tt vector based on linear fit
        # Find zeros
        zc_idx = np.where(np.diff(np.signbit(trace)))[0]
        x1 = tscale[zc_idx]
        x2 = tscale[zc_idx + 1]
        y1 = trace[zc_idx]
        y2 = trace[zc_idx + 1]
        a = (y2 - y1) / (x2 - x1)
        tt_zero = x1 - y1 / a
        # split tt and trace
        tt_split = np.split(tscale, zc_idx + 1)
        trace_split = np.split(trace, zc_idx + 1)
        tt_zi = tt_split[0]
        trace_zi = trace_split[0]
        # insert zeros in tt and trace
        for i in range(len(tt_zero)):
            tt_zi = np.hstack(
                (tt_zi, np.array([tt_zero[i]]), tt_split[i + 1]))
            trace_zi = np.hstack(
                (trace_zi, np.zeros(1), trace_split[i + 1]))
        return trace_zi, tt_zi

    if not ax:
        ax = plt.gca()
    for ntr in range(ntr):
        if fill:
            trace, t_trace = insert_zeros(w[:, ntr] * sf)
            ax.fill_betweenx(t_trace, ntr, trace + ntr,
                             where=trace >= 0,
                             facecolor=color,
                             linewidth=linewidth)
        ax.plot(w[:, ntr] * sf + ntr, tscale, color, linewidth=linewidth, **kwargs)

    ax.set_xlim(-1, ntr + 1)
    ax.set_ylim(tscale[0], tscale[-1])
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Trace')
    ax.invert_yaxis()


def traces(w, **kwargs):
    """
    Matplotlib display of traces

    :param w: 2D array (numpy array dimension nsamples, ntraces)
    :param fs: sampling frequency
    :param gain: display gain
    :param ax: matplotlib axes object
    :return: None
    """
    wiggle(w, **kwargs, fill=False)


if __name__ == "__main__":
    w = np.random.rand(500, 40) - 0.5
    wiggle(w, fs=30000)
    traces(w, fs=30000, color='r')

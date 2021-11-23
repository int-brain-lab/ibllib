#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scipy

import ibllib.dsp as dsp


def wiggle(w, fs=1, gain=0.71, color='k', ax=None, fill=True, linewidth=0.5, t0=0, clip=2,
           **kwargs):
    """
    Matplotlib display of wiggle traces

    :param w: 2D array (numpy array dimension nsamples, ntraces)
    :param fs: sampling frequency
    :param gain: display gain
    :param color: ('k') color of traces
    :param ax: (None) matplotlib axes object
    :param fill: (True) fill variable area above 0
    :param t0: (0) timestamp of the first sample
    :return: None
    """
    nech, ntr = w.shape
    tscale = np.arange(nech) / fs
    sf = gain / np.sqrt(dsp.rms(w.flatten()))

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
            if clip:
                trace = np.maximum(np.minimum(trace, clip), -clip)
            ax.fill_betweenx(t_trace + t0, ntr, trace + ntr,
                             where=trace >= 0,
                             facecolor=color,
                             linewidth=linewidth)
        wplot = np.minimum(np.maximum(w[:, ntr] * sf, -clip), clip)
        ax.plot(wplot + ntr, tscale + t0, color, linewidth=linewidth, **kwargs)

    ax.set_xlim(-1, ntr + 1)
    ax.set_ylim(tscale[0] + t0, tscale[-1] + t0)
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Trace')
    ax.invert_yaxis()

    return ax


class Density:
    def __init__(self, w, fs=1, cmap='bone', ax=None, taxis=0, **kwargs):
        """
        Matplotlib display of traces as a density display

        :param w: 2D array (numpy array dimension nsamples, ntraces)
        :param fs: sampling frequency (Hz)
        :param ax: axis to plot in
        :return: None
        """
        w = w.reshape(w.shape[0], -1)
        if taxis == 0:
            nech, ntr = w.shape
            tscale = np.array([0, nech - 1]) / fs * 1e3
            extent = [-0.5, ntr - 0.5, tscale[1], tscale[0]]
            xlabel, ylabel, origin = ('Trace', 'Time (ms)', 'upper')
        elif taxis == 1:
            ntr, nech = w.shape
            tscale = np.array([0, nech - 1]) / fs * 1e3
            extent = [tscale[0], tscale[1], -0.5, ntr - 0.5]
            ylabel, xlabel, origin = ('Trace', 'Time (ms)', 'lower')
        if ax is None:
            self.figure, ax = plt.subplots()
        else:
            self.figure = ax.get_figure()
        self.im = ax.imshow(w, aspect='auto', cmap=cmap, extent=extent, origin=origin, **kwargs)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        self.cid_key = self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ax = ax

    def on_key_press(self, event):
        if event.key == 'ctrl+a':
            self.im.set_data(self.im.get_array() * np.sqrt(2))
        elif event.key == 'ctrl+z':
            self.im.set_data(self.im.get_array() / np.sqrt(2))
        else:
            return
        self.figure.canvas.draw()


class Traces:
    def __init__(self, w, fs=1, gain=0.71, color='k', ax=None, linewidth=0.5, t0=0, **kwargs):
        """
        Matplotlib display of traces as a density display

        :param w: 2D array (numpy array dimension nsamples, ntraces)
        :param fs: sampling frequency (Hz)
        :param ax: axis to plot in
        :return: None
        """
        w = w.reshape(w.shape[0], -1)
        nech, ntr = w.shape
        tscale = np.arange(nech) / fs * 1e3
        sf = gain / dsp.rms(w.flatten()) / 2
        if ax is None:
            self.figure, ax = plt.subplots()
        else:
            self.figure = ax.get_figure()
        self.plot = ax.plot(w * sf + np.arange(ntr), tscale + t0, color,
                            linewidth=linewidth, **kwargs)
        ax.set_xlim(-1, ntr + 1)
        ax.set_ylim(tscale[0] + t0, tscale[-1] + t0)
        ax.set_ylabel('Time (ms)')
        ax.set_xlabel('Trace')
        ax.invert_yaxis()
        self.cid_key = self.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ax = ax

    def on_key_press(self, event):
        if event.key == 'ctrl+a':
            for i, l in enumerate(self.plot):
                l.set_xdata((l.get_xdata() - i) * np.sqrt(2) + i)
        elif event.key == 'ctrl+z':
            for i, l in enumerate(self.plot):
                l.set_xdata((l.get_xdata() - i) / np.sqrt(2) + i)
        else:
            return
        self.figure.canvas.draw()


def squares(tscale, polarity, ax=None, yrange=[-1, 1], **kwargs):
    """
    Matplotlib display of rising and falling fronts in a square-wave pattern

    :param tscale: time of indices of fronts
    :param polarity: polarity of front (1: rising, -1:falling)
    :param ax: matplotlib axes object
    :return: None
    """
    if not ax:
        ax = plt.gca()
    isort = np.argsort(tscale)
    tscale = tscale[isort]
    polarity = polarity[isort]
    f = np.tile(polarity, (2, 1))
    t = np.concatenate((tscale, np.r_[tscale[1:], tscale[-1]])).reshape(2, f.shape[1])
    ydata = f.transpose().ravel()
    ydata = (ydata + 1) / 2 * (yrange[1] - yrange[0]) + yrange[0]
    ax.plot(t.transpose().ravel(), ydata, **kwargs)


def vertical_lines(x, ymin=0, ymax=1, ax=None, **kwargs):
    """
    From a x vector, draw separate vertical lines at each x location ranging from ymin to ymax

    :param x: numpy array vector of x values where to display lnes
    :param ymin: lower end of the lines (scalar)
    :param ymax: higher end of the lines (scalar)
    :param ax: (optional) matplotlib axis instance
    :return: None
    """
    x = np.tile(x, (3, 1))
    x[2, :] = np.nan
    y = np.zeros_like(x)
    y[0, :] = ymin
    y[1, :] = ymax
    y[2, :] = np.nan
    if not ax:
        ax = plt.gca()
    ax.plot(x.T.flatten(), y.T.flatten(), **kwargs)


def spectrum(w, fs, smooth=None, unwrap=True, axis=0, **kwargs):
    """
    Display spectral density of a signal along a given dimension
    spectrum(w, fs)
    :param w: signal
    :param fs: sampling frequency (Hz)
    :param smooth: (None) frequency samples to smooth over
    :param unwrap: (True) unwraps the phase specrum
    :param axis: axis on which to compute the FFT
    :param kwargs: plot arguments to be passed to matplotlib
    :return: matplotlib axes
    """
    axis = 0
    smooth = None
    unwrap = True

    ns = w.shape[axis]
    fscale = dsp.fscale(ns, 1 / fs, one_sided=True)
    W = scipy.fft.rfft(w, axis=axis)
    amp = 20 * np.log10(np.abs(W))
    phi = np.angle(W)

    if unwrap:
        phi = np.unwrap(phi)

    if smooth:
        nf = np.round(smooth / fscale[1] / 2) * 2 + 1
        amp = dsp.smooth.mwa(amp, nf)
        phi = dsp.smooth.mwa(phi, nf)

    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(fscale, amp, **kwargs)
    ax[1].plot(fscale, phi, **kwargs)

    ax[0].set_title('Spectral Density (dB rel to amplitude.Hz^-0.5)')
    ax[0].set_ylabel('Amp (dB)')
    ax[1].set_ylabel('Phase (rad)')
    ax[1].set_xlabel('Frequency (Hz)')
    return ax


def color_cycle(ind=None):
    """
    Gets the matplotlib color-cycle as RGB numpy array of floats between 0 and 1
    :return:
    """
    # import matplotlib as mpl
    # c = np.uint32(np.array([int(c['color'][1:], 16) for c in mpl.rcParams['axes.prop_cycle']]))
    # c = np.double(np.flip(np.reshape(c.view(np.uint8), (c.size, 4))[:, :3], 1)) / 255
    c = np.array([[0.12156863, 0.46666667, 0.70588235],
                  [1., 0.49803922, 0.05490196],
                  [0.17254902, 0.62745098, 0.17254902],
                  [0.83921569, 0.15294118, 0.15686275],
                  [0.58039216, 0.40392157, 0.74117647],
                  [0.54901961, 0.3372549, 0.29411765],
                  [0.89019608, 0.46666667, 0.76078431],
                  [0.49803922, 0.49803922, 0.49803922],
                  [0.7372549, 0.74117647, 0.13333333],
                  [0.09019608, 0.74509804, 0.81176471]])
    if ind is None:
        return c
    else:
        return tuple(c[ind % c.shape[0], :])


if __name__ == "__main__":
    w = np.random.rand(500, 40) - 0.5
    wiggle(w, fs=30000)
    Traces(w, fs=30000, color='r')

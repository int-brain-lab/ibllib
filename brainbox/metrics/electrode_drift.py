import numpy as np

from ibllib.dsp import smooth
from brainbox.processing import bincount2D


def estimate_drift(spike_times, spike_amps, spike_depths):
    # binning parameters
    DT_SECS = 1  # output sampling rate of the depth estimation (seconds)
    DEPTH_BIN_UM = 2  # binning parameter for depth
    AMP_RES_V = 100 * 1e-6  # binning parameter for amplitudes
    NXCORR = 50  # positive and negative lag in depth samples to look for depth
    NT_SMOOTH = 9  # length of the Gaussian smoothing window in samples (DT_SECS rate)
    DISPLAY = False

    # experimental: try the amp with a log scale
    na = int(np.ceil(np.nanmax(spike_amps) / AMP_RES_V))
    nd = int(np.ceil(np.nanmax(spike_depths) / DEPTH_BIN_UM))
    nt = int(np.ceil(np.max(spike_times) / DT_SECS))

    # 3d histogram of spikes along amplitude, depths and time
    atd_hist = np.zeros((na, nt, nd))
    abins = np.ceil(spike_amps / AMP_RES_V)
    for i, abin in enumerate(np.unique(abins)):
        inds = np.where(np.logical_and(abins == abin, ~np.isnan(spike_depths)))[0]
        a, _, _ = bincount2D(spike_depths[inds], spike_times[inds], DEPTH_BIN_UM, DT_SECS,
                             [0, nd * DEPTH_BIN_UM], [0, nt * DT_SECS])
        atd_hist[i] = a[:-1, :-1]

    # compute the depth lag by xcorr
    # experimental: LP the fft for a better tracking ?
    atd_ = np.fft.fft(atd_hist, axis=-1)
    xcorr = np.real(np.fft.ifft(atd_ * np.conj(atd_[:, 0, :])[:, np.newaxis, :]))
    xcorr = np.sum(xcorr, axis=0)
    xcorr = np.c_[xcorr[:, -NXCORR:], xcorr[:, :NXCORR + 1]]

    # experimental: parabolic fit to get max values
    raw_drift = (np.argmax(xcorr, axis=-1) - NXCORR) * DEPTH_BIN_UM
    drift = smooth.rolling_window(raw_drift, window_len=NT_SMOOTH, window='hanning')

    if DISPLAY:
        import matplotlib.pyplot as plt
        from ibllib.plots import Density
        Density(atd_hist[5, :, :])
        plt.figure()
        plt.plot(xcorr.transpose())
        plt.figure()
        plt.plot(raw_drift - 50)
        plt.plot(drift - 50)

    return drift

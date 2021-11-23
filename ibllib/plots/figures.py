"""
Module that produces figures, usually for the extraction pipeline
"""
from pathlib import Path

import numpy as np
import scipy.signal


def ephys_bad_channels(raw, fs, channel_labels, channel_features, title="ephys_bad_channels", save_dir=None):
    nc = raw.shape[0]
    inoisy = np.where(channel_labels == 2)[0]
    idead = np.where(channel_labels == 1)[0]
    ioutside = np.where(channel_labels == 3)[0]
    from easyqc.gui import viewseis
    import matplotlib.pyplot as plt

    # display voltage traces
    eqcs = []
    butter_kwargs = {'N': 3, 'Wn': 300 / fs * 2, 'btype': 'highpass'}
    # butterworth, for display only
    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    butt = scipy.signal.sosfiltfilt(sos, raw)
    eqcs.append(viewseis(butt.T, si=1 / fs * 1e3, title='butt', taxis=0))
    for eqc in eqcs:
        y, x = np.meshgrid(ioutside, np.linspace(0, 1 * 1e3, 500))
        eqc.ctrl.add_scatter(x.flatten(), y.flatten(), rgb=(164, 142, 35), label='outside')
        y, x = np.meshgrid(inoisy, np.linspace(0, 1 * 1e3, 500))
        eqc.ctrl.add_scatter(x.flatten(), y.flatten(), rgb=(255, 0, 0), label='noisy')
        y, x = np.meshgrid(idead, np.linspace(0, 1 * 1e3, 500))
        eqc.ctrl.add_scatter(x.flatten(), y.flatten(), rgb=(0, 0, 255), label='dead')
    # display features
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=[16, 9], tight_layout=True)

    # fig.suptitle(f"pid:{pid}, \n eid:{eid}, \n {one.eid2path(eid).parts[-3:]}, {pname}")
    fig.suptitle(title)
    axs[0, 0].plot(channel_features['rms_raw'] * 1e6)
    axs[0, 0].set(title='rms', xlabel='channel number', ylabel='rms (uV)', ylim=[0, 100])

    axs[1, 0].plot(channel_features['psd_hf'])
    axs[1, 0].plot(inoisy, np.minimum(channel_features['psd_hf'][inoisy], 0.0999), 'xr')
    axs[1, 0].set(title='PSD above 12kHz', xlabel='channel number', ylabel='PSD (uV ** 2 / Hz)', ylim=[0, 0.1])
    axs[1, 0].legend = ['psd', 'noisy']

    axs[0, 1].plot(channel_features['xcor_hf'])
    axs[0, 1].plot(channel_features['xcor_lf'])

    axs[0, 1].plot(idead, channel_features['xcor_hf'][idead], 'xb')
    axs[0, 1].plot(ioutside, channel_features['xcor_lf'][ioutside], 'xy')
    axs[0, 1].set(title='Similarity', xlabel='channel number', ylabel='', ylim=[-1.5, 0.5])
    axs[0, 1].legend(['detrend', 'trend', 'dead', 'outside'])

    fscale, psd = scipy.signal.welch(raw * 1e6, fs=fs)  # units; uV ** 2 / Hz
    axs[1, 1].imshow(20 * np.log10(psd).T, extent=[0, nc - 1, fscale[0], fscale[-1]], origin='lower', aspect='auto',
                     vmin=-50, vmax=-20)
    axs[1, 1].set(title='PSD', xlabel='channel number', ylabel="Frequency (Hz)")
    axs[1, 1].plot(idead, idead * 0 + fs / 2 - 500, 'xb')
    axs[1, 1].plot(inoisy, inoisy * 0 + fs / 2 - 500, 'xr')
    axs[1, 1].plot(ioutside, ioutside * 0 + fs / 2 - 500, 'xy')

    eqcs[0].ctrl.set_gain(-90)
    eqcs[0].resize(1960, 1200)
    eqcs[0].viewBox_seismic.setXRange(450, 500)
    eqcs[0].viewBox_seismic.setYRange(0, nc)
    eqcs[0].ctrl.propagate()

    if save_dir is not None:
        fig.savefig(Path(save_dir).joinpath(f"{title}.png"))
        for eqc in eqcs:
            eqc.grab().save(str(Path(save_dir).joinpath(f"{title}_data_{eqc.windowTitle()}.png")))

    return fig, eqcs[0]

"""
Module that produces figures, usually for the extraction pipeline
"""
import logging
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import pandas as pd
import scipy.signal

from one.api import ONE
from one.alf.exceptions import ALFObjectNotFound
from ibllib.io.video import get_video_frame, url_from_eid

logger = logging.getLogger('ibllib')


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


def raw_destripe(raw, fs, t0, i_plt, n_plt,
                 fig=None, axs=None, savedir=None, detect_badch=True,
                 SAMPLE_SKIP=200, DISPLAY_TIME=0.05, N_CHAN=384,
                 MIN_X=-0.00011, MAX_X=0.00011):
    '''
    :param raw: raw ephys data, Ns x Nc, x-axis: time (s), y-axis: channel
    :param fs: sampling freq (Hz) of the raw ephys data
    :param t0: time (s) of ephys sample beginning from session start
    :param i_plt: increment of plot to display image one (start from 0, has to be < n_plt)
    :param n_plt: total number of subplot on figure
    :param fig: figure handle
    :param axs: axis handle
    :param savedir: filename, including directory, to save figure to
    :param detect_badch: boolean, to detect or not bad channels
    :param SAMPLE_SKIP: number of samples to skip at origin of ephsy sample for display
    :param DISPLAY_TIME: time (s) to display
    :param N_CHAN: number of expected channels on the probe
    :param MIN_X: max voltage for color range
    :param MAX_X: min voltage for color range
    :return: fig, axs
    '''

    # Import
    import matplotlib.pyplot as plt
    from ibllib.dsp import voltage
    from ibllib.plots import Density

    # Init fig
    if fig is None or axs is None:
        fig, axs = plt.subplots(nrows=1, ncols=n_plt, figsize=(14, 5), gridspec_kw={'width_ratios': 4 * n_plt})

    if i_plt > len(axs) - 1:  # Error
        raise ValueError(f'The given increment of subplot ({i_plt+1}) '
                         f'is larger than the total number of subplots ({len(axs)})')

    [nc, ns] = raw.shape
    if nc == N_CHAN:
        destripe = voltage.destripe(raw, fs=fs)
        X = destripe[:, :int(DISPLAY_TIME * fs)].T
        Xs = X[SAMPLE_SKIP:].T  # Remove artifact at beginning
        Tplot = Xs.shape[1] / fs

        # PLOT RAW DATA
        d = Density(-Xs, fs=fs, taxis=1, ax=axs[i_plt], vmin=MIN_X, vmax=MAX_X, cmap='Greys') # noqa
        axs[i_plt].set_ylabel('')
        axs[i_plt].set_xlim((0, Tplot * 1e3))
        axs[i_plt].set_ylim((0, nc))

        # Init title
        title_plt = f't0 = {int(t0 / 60)} min'

        if detect_badch:
            # Detect and remove bad channels prior to spike detection
            labels, xfeats = voltage.detect_bad_channels(raw, fs)
            idx_badchan = np.where(labels != 0)[0]
            # Plot bad channels on raw data
            x, y = np.meshgrid(idx_badchan, np.linspace(0, Tplot * 1e3, 20))
            axs[i_plt].plot(y.flatten(), x.flatten(), '.k', markersize=1)
            # Append title
            title_plt += f', n={len(idx_badchan)} bad ch'

        # Set title
        axs[i_plt].title.set_text(title_plt)

    else:
        axs[i_plt].title.set_text(f'CANNOT DESTRIPE, N CHAN = {nc}')

    # Amend some axis style
    if i_plt > 0:
        axs[i_plt].set_yticklabels('')

    # Fig layout
    fig.tight_layout()
    if savedir is not None:
        fig.savefig(fname=savedir)

    return fig, axs


def dlc_qc(eid, one=None):
    
    import matplotlib.pyplot as plt
    import brainbox.behavior.wheel as bbox_wheel
    from brainbox.behavior.dlc import SAMPLING, T_BIN, plot_trace_on_frame, plot_wheel_position, plot_lick_psth, \
        plot_lick_raster, plot_motion_energy_psth, plot_speed_psth, plot_pupil_diameter_psth

    one = one or ONE()
    # Load data
    data = {}
    for cam in ['left', 'right', 'body']:
        # Load camera data (at least dlc should exist)
        data[f'vid_{cam}'] = one.load_object(eid, f'{cam}Camera', collection='alf')
        # Load a single frame for each video, first check if data is local, otherwise stream
        video_path = one.eid2path(eid).joinpath('raw_video_data', f'_iblrig_{cam}Camera.raw.mp4')
        if not video_path.exists():
            video_path = url_from_eid(eid, one=one)[cam]
        try:
            data[f'frame_{cam}'] = get_video_frame(video_path, frame_number=5 * 60 * SAMPLING[cam])[:, :, 0]
        except TypeError:
            logger.warning(f"Could not load video frame for {cam} camera, some DLC QC plots have to be skipped.")
            data[f'frame_{cam}'] = None
    # Try to load alf objects
    for alf_object in ['trials', 'wheel', 'licks']:
        try:
            data[f'{alf_object}'] = one.load_object(eid, alf_object)
        except ALFObjectNotFound:
            logger.warning(f"Could not load {alf_object} object for session {eid}, some plots have to be skipped.")
            data[f'{alf_object}'] = None
            
    # Load / clean up data from alf objects
    data['licks'] = data['licks'].times if data['licks'] else None
    data['wheel'] = bbox_wheel.interpolate_position(data['wheel'].timestamps, data['wheel'].position,
                                                    freq=1 / T_BIN) if data['wheel'] else (None, None)
    if data['trials']:
        data['trials'] = pd.DataFrame({k: data['trials'][k] for k in ['stimOn_times', 'feedback_times', 'choice', 'feedbackType']})
        # Discard nan events and too long trials
        data['trials'] = data['trials'].dropna()
        data['trials'] = data['trials'].drop(data['trials'][(data['trials']['feedback_times']
                                                             - data['trials']['stimOn_times']) > 10].index)
    
    # List panels: axis functions and inputs
    panels = [(plot_trace_on_frame, {'frame': data['frame_left'], 'dlc_df': data['vid_left'].dlc, 'cam': 'left'}),
              (plot_trace_on_frame, {'frame': data['frame_right'], 'dlc_df': data['vid_right'].dlc, 'cam': 'right'}),
              (plot_trace_on_frame, {'frame': data['frame_body'], 'dlc_df': data['vid_body'].dlc, 'cam': 'body'}),
              (plot_wheel_position,
               {'wheel_position': data['wheel'][0], 'wheel_time': data['wheel'][1], 'trials_df': data['trials']}),
              (plot_motion_energy_psth,
               {'vid_data_dict': {'left': data['vid_left'], 'right': data['vid_right'], 'body': data['vid_body']},
                'trials_df': data['trials']}),
              (plot_speed_psth,
               {'dlc_df': data['vid_left'].dlc, 'cam_times': data['vid_left'].times, 'trials_df': data['trials'],
                'feature': 'paw_r', 'cam': 'left'}),
              (plot_speed_psth,
               {'dlc_df': data['vid_left'].dlc, 'cam_times': data['vid_left'].times, 'trials_df': data['trials'],
                'feature': 'nose_tip', 'cam': 'left', 'legend': False}),
              (plot_lick_psth, {'lick_times': data['licks'], 'trials_df': data['trials']}),
              (plot_lick_raster, {'lick_times': data['licks'], 'trials_df': data['trials']}),
              (plot_pupil_diameter_psth, {'vid_data': data['vid_left'], 'trials_df': data['trials'], 'cam': 'left'})
              ]

    # Plotting
    plt.rcParams.update({'font.size': 10})
    fig = plt.figure(figsize=(17, 10))
    for i, panel in enumerate(panels):
        ax = plt.subplot(2, 5, i + 1)
        ax.text(-0.1, 1.15, ascii_uppercase[i], transform=ax.transAxes, fontsize=16, fontweight='bold')
        # Check if any of the inputs is None
        if any([v is None for v in panel[1].values()]):
            ax.text(.5, .5, f"Data incomplete\n{panel[0].__name__}", color='r', fontweight='bold',
                    fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            plt.axis('off')
        else:
            try:
                panel[0](**panel[1])
            except BaseException:
                ax.text(.5, .5, f'Error in \n{panel[0].__name__}', color='r', fontweight='bold',
                        fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig

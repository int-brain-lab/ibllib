"""
Module that produces figures, usually for the extraction pipeline
"""
import logging
from pathlib import Path
from string import ascii_uppercase

import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from ibllib.dsp import voltage
from ibllib.plots.snapshot import ReportSnapshotProbe
from one.api import ONE
import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound
from ibllib.io.video import get_video_frame, url_from_eid
from brainbox.plot import driftmap
from brainbox.behavior.dlc import SAMPLING, plot_trace_on_frame, plot_wheel_position, plot_lick_hist, \
    plot_lick_raster, plot_motion_energy_hist, plot_speed_hist, plot_pupil_diameter_hist
from brainbox.io.one import load_spike_sorting_fast
from brainbox.ephys_plots import plot_brain_regions


logger = logging.getLogger('ibllib')


class SpikeSorting(ReportSnapshotProbe):
    """
    Plots raw electrophysiology AP band
    :param session_path: session path
    :param probe_id: str, UUID of the probe insertion for which to create the plot
    :param **kwargs: keyword arguments passed to tasks.Task
    """

    def _run(self, collection=None):
        """runs for initiated PID, streams data, destripe and check bad channels"""
        all_here, output_files = self.assert_expected(self.output_files, silent=True)
        spike_sorting_runs = self.one.list_datasets(self.eid, filename='spikes.times.npy', collection=f'alf/{self.pname}*')
        if all_here and len(output_files) == len(spike_sorting_runs):
            return output_files
        logger.info(self.output_directory)
        output_files = []
        for run in spike_sorting_runs:
            collection = str(Path(run).parent)
            spikes, clusters, channels = load_spike_sorting_fast(
                eid=self.eid, probe=self.pname, one=self.one, nested=False, collection=collection,
                dataset_types=['spikes.depths'], brain_regions=self.brain_regions)

            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [.95, .05]}, sharey=True, figsize=(16, 9))
            driftmap(spikes.times, spikes.depths, t_bin=0.007, d_bin=10, vmax=0.5, ax=axs[0])
            if 'atlas_id' in channels.keys():
                plot_brain_regions(channels['atlas_id'], channel_depths=channels['axial_um'],
                                   brain_regions=None, display=True, ax=axs[1])
            title_str = f"{self.pid_label}, {collection}, {self.pid} \n " \
                        f"{spikes.clusters.size:_} spikes, {clusters.depths.size:_} clusters"
            logger.info(title_str.replace("\n", ""))
            axs[0].set(ylim=[0, 3800], title=title_str)
            run_label = str(Path(collection).relative_to(f'alf/{self.pname}'))
            run_label = "" if run_label == '.' else run_label
            output_files.append(self.output_directory.joinpath(f"spike_sorting_raster_{run_label}.png"))
            fig.savefig(output_files[-1])
            plt.close(fig)
        return output_files

    def get_probe_signature(self):
        input_signature = [('spikes.times.npy', f'alf/{self.pname}', True),
                           ('spikes.amps.npy', f'alf/{self.pname}', True),
                           ('spikes.depths.npy', f'alf/{self.pname}', True)]
        output_signature = [('spike_sorting_raster*.png', f'snapshot/{self.pname}', True)]
        self.signature = {'input_files': input_signature, 'output_files': output_signature}


class BadChannelsAp(ReportSnapshotProbe):
    """
    Plots raw electrophysiology AP band
    task = BadChannelsAp(pid, one=one=one)
    :param session_path: session path
    :param probe_id: str, UUID of the probe insertion for which to create the plot
    :param **kwargs: keyword arguments passed to tasks.Task
    """

    def get_probe_signature(self):
        pname = self.pname
        input_signature = [('*ap.meta', f'raw_ephys_data/{pname}', True),
                           ('*ap.ch', f'raw_ephys_data/{pname}', False)]
        # ('*ap.cbin', f'raw_ephys_data/{pname}', False)]
        output_signature = [('raw_ephys_bad_channels.png', f'snapshot/{pname}', True),
                            ('raw_ephys_bad_channels_highpass.png', f'snapshot/{pname}', True),
                            ('raw_ephys_bad_channels_highpass.png', f'snapshot/{pname}', True),
                            ('raw_ephys_bad_channels_destripe.png', f'snapshot/{pname}', True),
                            ('raw_ephys_bad_channels_difference.png', f'snapshot/{pname}', True),
                            ]
        self.signature = {'input_files': input_signature, 'output_files': output_signature}

    def _run(self):
        """runs for initiated PID, streams data, destripe and check bad channels"""
        assert self.pid
        SNAPSHOT_LABEL = "raw_ephys_bad_channels"
        eid, pname = self.one.pid2eid(self.pid)
        output_files = list(self.output_directory.glob(f'{SNAPSHOT_LABEL}*'))
        if len(output_files) == 4:
            return output_files
        self.output_directory.mkdir(exist_ok=True, parents=True)
        from brainbox.io.spikeglx import stream
        T0 = 60 * 30
        sr, t0 = stream(self.pid, T0, nsecs=1, one=self.one)
        raw = sr[:, :-sr.nsync].T
        channel_labels, channel_features = voltage.detect_bad_channels(raw, sr.fs)
        _, _, output_files = ephys_bad_channels(
            raw=raw, fs=sr.fs, channel_labels=channel_labels, channel_features=channel_features,
            title=SNAPSHOT_LABEL, destripe=True, save_dir=self.output_directory)
        return output_files


def ephys_bad_channels(raw, fs, channel_labels, channel_features, title="ephys_bad_channels", save_dir=None,
                       destripe=False, eqcs=None):
    nc, ns = raw.shape
    rl = ns / fs
    if fs >= 2600:  # AP band
        ylim_rms = [0, 100]
        ylim_psd_hf = [0, 0.1]
        eqc_xrange = [450, 500]
        butter_kwargs = {'N': 3, 'Wn': 300 / fs * 2, 'btype': 'highpass'}
        eqc_gain = - 90
    else:
        # we are working with the LFP
        ylim_rms = [0, 1000]
        ylim_psd_hf = [0, 1]
        eqc_xrange = [450, 950]
        butter_kwargs = {'N': 3, 'Wn': np.array([2, 125]) / fs * 2, 'btype': 'bandpass'}
        eqc_gain = - 78

    inoisy = np.where(channel_labels == 2)[0]
    idead = np.where(channel_labels == 1)[0]
    ioutside = np.where(channel_labels == 3)[0]
    from easyqc.gui import viewseis

    # display voltage traces
    eqcs = [] if eqcs is None else eqcs
    # butterworth, for display only
    sos = scipy.signal.butter(**butter_kwargs, output='sos')
    butt = scipy.signal.sosfiltfilt(sos, raw)
    eqcs.append(viewseis(butt.T, si=1 / fs * 1e3, title='highpass', taxis=0))
    if destripe:
        dest = voltage.destripe(raw, fs=fs, channel_labels=channel_labels)
        eqcs.append(viewseis(dest.T, si=1 / fs * 1e3, title='destripe', taxis=0))
        eqcs.append(viewseis((butt - dest).T, si=1 / fs * 1e3, title='difference', taxis=0))
    for eqc in eqcs:
        y, x = np.meshgrid(ioutside, np.linspace(0, rl * 1e3, 500))
        eqc.ctrl.add_scatter(x.flatten(), y.flatten(), rgb=(164, 142, 35), label='outside')
        y, x = np.meshgrid(inoisy, np.linspace(0, rl * 1e3, 500))
        eqc.ctrl.add_scatter(x.flatten(), y.flatten(), rgb=(255, 0, 0), label='noisy')
        y, x = np.meshgrid(idead, np.linspace(0, rl * 1e3, 500))
        eqc.ctrl.add_scatter(x.flatten(), y.flatten(), rgb=(0, 0, 255), label='dead')
    # display features
    fig, axs = plt.subplots(2, 2, sharex=True, figsize=[16, 9], tight_layout=True)

    # fig.suptitle(f"pid:{pid}, \n eid:{eid}, \n {one.eid2path(eid).parts[-3:]}, {pname}")
    fig.suptitle(title)
    axs[0, 0].plot(channel_features['rms_raw'] * 1e6)
    axs[0, 0].set(title='rms', xlabel='channel number', ylabel='rms (uV)', ylim=ylim_rms)

    axs[1, 0].plot(channel_features['psd_hf'])
    axs[1, 0].plot(inoisy, np.minimum(channel_features['psd_hf'][inoisy], 0.0999), 'xr')
    axs[1, 0].set(title='PSD above 80% Nyquist', xlabel='channel number', ylabel='PSD (uV ** 2 / Hz)', ylim=ylim_psd_hf)
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
    axs[1, 1].plot(idead, idead * 0 + fs / 4, 'xb')
    axs[1, 1].plot(inoisy, inoisy * 0 + fs / 4, 'xr')
    axs[1, 1].plot(ioutside, ioutside * 0 + fs / 4, 'xy')

    eqcs[0].ctrl.set_gain(eqc_gain)
    eqcs[0].resize(1960, 1200)
    eqcs[0].viewBox_seismic.setXRange(*eqc_xrange)
    eqcs[0].viewBox_seismic.setYRange(0, nc)
    eqcs[0].ctrl.propagate()

    if save_dir is not None:
        output_files = [Path(save_dir).joinpath(f"{title}.png")]
        fig.savefig(output_files[0])
        for eqc in eqcs:
            output_files.append(Path(save_dir).joinpath(f"{title}_{eqc.windowTitle()}.png"))
            eqc.grab().save(str(output_files[-1]))
        return fig, eqcs, output_files
    else:
        return fig, eqcs


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


def dlc_qc_plot(eid, one=None):
    """
    Creates DLC QC plot.
    Data is searched first locally, then on Alyx. Panels that lack required data are skipped.

    Required data to create all panels
     'raw_video_data/_iblrig_bodyCamera.raw.mp4',
     'raw_video_data/_iblrig_leftCamera.raw.mp4',
     'raw_video_data/_iblrig_rightCamera.raw.mp4',
     'alf/_ibl_bodyCamera.dlc.pqt',
     'alf/_ibl_leftCamera.dlc.pqt',
     'alf/_ibl_rightCamera.dlc.pqt',
     'alf/_ibl_bodyCamera.times.npy',
     'alf/_ibl_leftCamera.times.npy',
     'alf/_ibl_rightCamera.times.npy',
     'alf/_ibl_leftCamera.features.pqt',
     'alf/rightROIMotionEnergy.position.npy',
     'alf/leftROIMotionEnergy.position.npy',
     'alf/bodyROIMotionEnergy.position.npy',
     'alf/_ibl_trials.choice.npy',
     'alf/_ibl_trials.feedbackType.npy',
     'alf/_ibl_trials.feedback_times.npy',
     'alf/_ibl_trials.stimOn_times.npy',
     'alf/_ibl_wheel.position.npy',
     'alf/_ibl_wheel.timestamps.npy',
     'alf/licks.times.npy',

    :params eid: Session ID
    :params one: ONE instance, if None is given, default ONE is instantiated
    :returns: Matplotlib figure
    """

    one = one or ONE()
    data = {}
    # Camera data
    for cam in ['left', 'right', 'body']:
        # Load a single frame for each video, first check if data is local, otherwise stream
        video_path = one.eid2path(eid).joinpath('raw_video_data', f'_iblrig_{cam}Camera.raw.mp4')
        if not video_path.exists():
            try:
                video_path = url_from_eid(eid, one=one)[cam]
            except KeyError:
                logger.warning(f"No raw video data found for {cam} camera, some DLC QC plots have to be skipped.")
                data[f'{cam}_frame'] = None
        try:
            data[f'{cam}_frame'] = get_video_frame(video_path, frame_number=5 * 60 * SAMPLING[cam])[:, :, 0]
        except TypeError:
            logger.warning(f"Could not load video frame for {cam} camera, some DLC QC plots have to be skipped.")
            data[f'{cam}_frame'] = None
        # Load other video associated data
        for feat in ['dlc', 'times', 'features', 'ROIMotionEnergy']:
            # Check locally first, then try to load from alyx, if nothing works, set to None
            local_file = list(one.eid2path(eid).joinpath('alf').glob(f'*{cam}Camera.{feat}*'))
            alyx_file = [ds for ds in one.list_datasets(eid) if f'{cam}Camera.{feat}' in ds]
            if feat == 'features' and cam in ['body', 'right']:
                continue
            elif len(local_file) > 0:
                data[f'{cam}_{feat}'] = alfio.load_file_content(local_file[0])
            elif len(alyx_file) > 0:
                data[f'{cam}_{feat}'] = one.load_dataset(eid, alyx_file[0])
            else:
                logger.warning(f"Could not load _ibl_{cam}Camera.{feat} some DLC QC plots have to be skipped.")
                data[f'{cam}_{feat}'] = None
    # Session data
    for alf_object in ['trials', 'wheel', 'licks']:
        try:
            data[f'{alf_object}'] = alfio.load_object(one.eid2path(eid).joinpath('alf'), alf_object)
            continue
        except ALFObjectNotFound:
            pass
        try:
            data[f'{alf_object}'] = one.load_object(eid, alf_object)
        except ALFObjectNotFound:
            logger.warning(f"Could not load {alf_object} object for session {eid}, some plots have to be skipped.")
            data[f'{alf_object}'] = None
    # Simplify to what we actually need
    data['licks'] = data['licks'].times if data['licks'] else None
    data['left_pupil'] = data['left_features'].pupilDiameter_smooth if data['left_features'] is not None else None
    data['wheel_time'] = data['wheel'].timestamps if data['wheel'] is not None else None
    data['wheel_position'] = data['wheel'].position if data['wheel'] is not None else None
    if data['trials']:
        data['trials'] = pd.DataFrame(
            {k: data['trials'][k] for k in ['stimOn_times', 'feedback_times', 'choice', 'feedbackType']})
        # Discard nan events and too long trials
        data['trials'] = data['trials'].dropna()
        data['trials'] = data['trials'].drop(
            data['trials'][(data['trials']['feedback_times'] - data['trials']['stimOn_times']) > 10].index)
    # List panels: axis functions and inputs
    panels = [(plot_trace_on_frame, {'frame': data['left_frame'], 'dlc_df': data['left_dlc'], 'cam': 'left'}),
              (plot_trace_on_frame, {'frame': data['right_frame'], 'dlc_df': data['right_dlc'], 'cam': 'right'}),
              (plot_trace_on_frame, {'frame': data['body_frame'], 'dlc_df': data['body_dlc'], 'cam': 'body'}),
              (plot_wheel_position,
               {'wheel_position': data['wheel_position'], 'wheel_time': data['wheel_time'], 'trials_df': data['trials']}),
              (plot_motion_energy_hist,
               {'camera_dict': {'left': {'motion_energy': data['left_ROIMotionEnergy'], 'times': data['left_times']},
                                'right': {'motion_energy': data['right_ROIMotionEnergy'], 'times': data['right_times']},
                                'body': {'motion_energy': data['body_ROIMotionEnergy'], 'times': data['body_times']}},
                'trials_df': data['trials']}),
              (plot_speed_hist,
               {'dlc_df': data['left_dlc'], 'cam_times': data['left_times'], 'trials_df': data['trials']}),
              (plot_speed_hist,
               {'dlc_df': data['left_dlc'], 'cam_times': data['left_times'], 'trials_df': data['trials'],
                'feature': 'nose_tip', 'legend': False}),
              (plot_lick_hist, {'lick_times': data['licks'], 'trials_df': data['trials']}),
              (plot_lick_raster, {'lick_times': data['licks'], 'trials_df': data['trials']}),
              (plot_pupil_diameter_hist,
               {'pupil_diameter': data['left_pupil'], 'cam_times': data['left_times'], 'trials_df': data['trials']})
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

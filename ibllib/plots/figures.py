"""
Module that produces figures, usually for the extraction pipeline
"""
import logging
from pathlib import Path
import traceback
from string import ascii_uppercase

import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from ibllib.dsp import voltage
from ibllib.plots.snapshot import ReportSnapshotProbe, ReportSnapshot
from one.api import ONE
import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound
from ibllib.io.video import get_video_frame, url_from_eid
from brainbox.plot import driftmap
from brainbox.behavior.dlc import SAMPLING, plot_trace_on_frame, plot_wheel_position, plot_lick_hist, \
    plot_lick_raster, plot_motion_energy_hist, plot_speed_hist, plot_pupil_diameter_hist
from brainbox.ephys_plots import image_lfp_spectrum_plot, image_rms_plot, plot_brain_regions
from brainbox.io.one import load_spike_sorting_fast
from brainbox.behavior import training


logger = logging.getLogger('ibllib')


def set_axis_label_size(ax, labels=14, ticklabels=12, title=14, cmap=False):
    """
    Function to normalise size of all axis labels
    :param ax:
    :param labels:
    :param ticklabels:
    :param title:
    :param cmap:
    :return:
    """

    ax.xaxis.get_label().set_fontsize(labels)
    ax.yaxis.get_label().set_fontsize(labels)
    ax.tick_params(labelsize=ticklabels)
    ax.title.set_fontsize(title)

    if cmap:
        cbar = ax.images[-1].colorbar
        cbar.ax.tick_params(labelsize=ticklabels)
        cbar.ax.yaxis.get_label().set_fontsize(labels)


def remove_axis_outline(ax):
    """
    Function to remove outline of empty axis
    :param ax:
    :return:
    """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)


class BehaviourPlots(ReportSnapshot):
    """
    Behavioural plots
    """

    signature = {'input_files': [('*trials*', 'alf', True)],
                 'output_files': [('psychometric_curve.png', 'snapshot/behaviour', True),
                                  ('chronometric_curve.png', 'snapshot/behaviour', True),
                                  ('reaction_time_with_trials.png', 'snapshot/behaviour', True)]
                 }

    def __init__(self, eid, session_path=None, one=None, **kwargs):
        self.one = one or ONE()
        self.eid = eid
        self.session_path = session_path or self.one.eid2path(self.eid)
        super(BehaviourPlots, self).__init__(self.session_path, self.eid,
                                             **kwargs)
        self.output_directory = self.session_path.joinpath('snapshot', 'behaviour')
        self.output_directory.mkdir(exist_ok=True, parents=True)

    def _run(self):

        output_files = []
        trials = alfio.load_object(self.session_path.joinpath('alf'), 'trials')
        title = '_'.join(list(self.session_path.parts[-3:]))

        fig, ax = training.plot_psychometric(trials, title=title, figsize=(8, 6))
        set_axis_label_size(ax)
        save_path = Path(self.output_directory).joinpath("psychometric_curve.png")
        output_files.append(save_path)
        fig.savefig(save_path)
        plt.close(fig)

        fig, ax = training.plot_reaction_time(trials, title=title, figsize=(8, 6))
        set_axis_label_size(ax)
        save_path = Path(self.output_directory).joinpath("chronometric_curve.png")
        output_files.append(save_path)
        fig.savefig(save_path)
        plt.close(fig)

        fig, ax = training.plot_reaction_time_over_trials(trials, title=title, figsize=(8, 6))
        set_axis_label_size(ax)
        save_path = Path(self.output_directory).joinpath("reaction_time_with_trials.png")
        output_files.append(save_path)
        fig.savefig(save_path)
        plt.close(fig)

        return output_files


# TODO put into histology and alignment pipeline
class HistologySlices(ReportSnapshotProbe):
    """
    Plots coronal and sagittal slice showing electrode locations
    """

    def _run(self):

        assert self.pid
        assert self.brain_atlas

        output_files = []

        electrodes = self.get_channels('electrodeSites', f'alf/{self.pname}')

        if self.hist_lookup[self.histology_status] > 0:
            fig = plt.figure(figsize=(12, 9))
            gs = fig.add_gridspec(2, 2, width_ratios=[.95, .05])
            ax1 = fig.add_subplot(gs[0, 0])
            self.brain_atlas.plot_tilted_slice(electrodes['mlapdv'], 1, ax=ax1)
            ax1.scatter(electrodes['mlapdv'][:, 0] * 1e6, electrodes['mlapdv'][:, 2] * 1e6, s=8, c='r')
            ax1.set_title(f"{self.pid_label}")

            ax2 = fig.add_subplot(gs[1, 0])
            self.brain_atlas.plot_tilted_slice(electrodes['mlapdv'], 0, ax=ax2)
            ax2.scatter(electrodes['mlapdv'][:, 1] * 1e6, electrodes['mlapdv'][:, 2] * 1e6, s=8, c='r')

            ax3 = fig.add_subplot(gs[:, 1])
            plot_brain_regions(electrodes['atlas_id'], brain_regions=self.brain_regions, display=True, ax=ax3,
                               title=self.histology_status)

            save_path = Path(self.output_directory).joinpath("histology_slices.png")
            output_files.append(save_path)
            fig.savefig(save_path)
            plt.close(fig)

        return output_files

    def get_probe_signature(self):
        input_signature = [('electrodeSites.localCoordinates.npy', f'alf/{self.pname}', False),
                           ('electrodeSites.brainLocationIds_ccf_2017.npy', f'alf/{self.pname}', False),
                           ('electrodeSites.mlapdv.npy', f'alf/{self.pname}', False)]
        output_signature = [('histology_slices.png', f'snapshot/{self.pname}', True)]
        self.signature = {'input_files': input_signature, 'output_files': output_signature}


class LfpPlots(ReportSnapshotProbe):
    """
    Plots LFP spectrum and LFP RMS plots
    """

    def _run(self):

        assert self.pid

        output_files = []

        if self.location != 'server':
            self.histology_status = self.get_histology_status()
            electrodes = self.get_channels('electrodeSites', f'alf/{self.pname}')

        # lfp spectrum
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [.95, .05]}, figsize=(16, 9))
        lfp = alfio.load_object(self.session_path.joinpath(f'raw_ephys_data/{self.pname}'), 'ephysSpectralDensityLF',
                                namespace='iblqc')
        _, _, _ = image_lfp_spectrum_plot(lfp.power, lfp.freqs, clim=[-65, -95], fig_kwargs={'figsize': (8, 6)}, ax=axs[0],
                                          display=True, title=f"{self.pid_label}")
        set_axis_label_size(axs[0], cmap=True)
        if self.histology_status:
            plot_brain_regions(electrodes['atlas_id'], brain_regions=self.brain_regions, display=True, ax=axs[1],
                               title=self.histology_status)
            set_axis_label_size(axs[1])
        else:
            remove_axis_outline(axs[1])

        save_path = Path(self.output_directory).joinpath("lfp_spectrum.png")
        output_files.append(save_path)
        fig.savefig(save_path)
        plt.close(fig)

        # lfp rms
        # TODO need to figure out the clim range
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [.95, .05]}, figsize=(16, 9))
        lfp = alfio.load_object(self.session_path.joinpath(f'raw_ephys_data/{self.pname}'), 'ephysTimeRmsLF', namespace='iblqc')
        _, _, _ = image_rms_plot(lfp.rms, lfp.timestamps, median_subtract=False, band='LFP', clim=[-35, -45], ax=axs[0],
                                 cmap='inferno', fig_kwargs={'figsize': (8, 6)}, display=True, title=f"{self.pid_label}")
        set_axis_label_size(axs[0], cmap=True)
        if self.histology_status:
            plot_brain_regions(electrodes['atlas_id'], brain_regions=self.brain_regions, display=True, ax=axs[1],
                               title=self.histology_status)
            set_axis_label_size(axs[1])
        else:
            remove_axis_outline(axs[1])

        save_path = Path(self.output_directory).joinpath("lfp_rms.png")
        output_files.append(save_path)
        fig.savefig(save_path)
        plt.close(fig)

        return output_files

    def get_probe_signature(self):
        input_signature = [('_iblqc_ephysTimeRmsLF.rms.npy', f'raw_ephys_data/{self.pname}', True),
                           ('_iblqc_ephysTimeRmsLF.timestamps.npy', f'raw_ephys_data/{self.pname}', True),
                           ('_iblqc_ephysSpectralDensityLF.freqs.npy', f'raw_ephys_data/{self.pname}', True),
                           ('_iblqc_ephysSpectralDensityLF.power.npy', f'raw_ephys_data/{self.pname}', True),
                           ('electrodeSites.localCoordinates.npy', f'alf/{self.pname}', False),
                           ('electrodeSites.brainLocationIds_ccf_2017.npy', f'alf/{self.pname}', False),
                           ('electrodeSites.mlapdv.npy', f'alf/{self.pname}', False)]
        output_signature = [('lfp_spectrum.png', f'snapshot/{self.pname}', True),
                            ('lfp_rms.png', f'snapshot/{self.pname}', True)]
        self.signature = {'input_files': input_signature, 'output_files': output_signature}


class ApPlots(ReportSnapshotProbe):
    """
    Plots AP RMS plots
    """

    def _run(self):

        assert self.pid

        output_files = []

        if self.location != 'server':
            self.histology_status = self.get_histology_status()
            electrodes = self.get_channels('electrodeSites', f'alf/{self.pname}')

        # TODO need to figure out the clim range
        fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [.95, .05]}, figsize=(16, 9))
        ap = alfio.load_object(self.session_path.joinpath(f'raw_ephys_data/{self.pname}'), 'ephysTimeRmsAP', namespace='iblqc')
        _, _, _ = image_rms_plot(ap.rms, ap.timestamps, median_subtract=False, band='AP', ax=axs[0],
                                 fig_kwargs={'figsize': (8, 6)}, display=True, title=f"{self.pid_label}")
        set_axis_label_size(axs[0], cmap=True)
        if self.histology_status:
            plot_brain_regions(electrodes['atlas_id'], brain_regions=self.brain_regions, display=True, ax=axs[1],
                               title=self.histology_status)
            set_axis_label_size(axs[1])
        else:
            remove_axis_outline(axs[1])

        save_path = Path(self.output_directory).joinpath("ap_rms.png")
        output_files.append(save_path)
        fig.savefig(save_path)
        plt.close(fig)

        return output_files

    def get_probe_signature(self):
        input_signature = [('_iblqc_ephysTimeRmsAP.rms.npy', f'raw_ephys_data/{self.pname}', True),
                           ('_iblqc_ephysTimeRmsAP.timestamps.npy', f'raw_ephys_data/{self.pname}', True),
                           ('electrodeSites.localCoordinates.npy', f'alf/{self.pname}', False),
                           ('electrodeSites.brainLocationIds_ccf_2017.npy', f'alf/{self.pname}', False),
                           ('electrodeSites.mlapdv.npy', f'alf/{self.pname}', False)]
        output_signature = [('ap_rms.png', f'snapshot/{self.pname}', True)]
        self.signature = {'input_files': input_signature, 'output_files': output_signature}


class SpikeSorting(ReportSnapshotProbe):
    """
    Plots raw electrophysiology AP band
    :param session_path: session path
    :param probe_id: str, UUID of the probe insertion for which to create the plot
    :param **kwargs: keyword arguments passed to tasks.Task
    """

    def _run(self, collection=None):
        """runs for initiated PID, streams data, destripe and check bad channels"""

        def plot_driftmap(self, spikes, clusters, channels, collection, ylim=(0, 3840)):
            fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios': [.95, .05]}, figsize=(16, 9))
            driftmap(spikes.times, spikes.depths, t_bin=0.007, d_bin=10, vmax=0.5, ax=axs[0])
            title_str = f"{self.pid_label}, {collection}, {self.pid} \n " \
                        f"{spikes.clusters.size:_} spikes, {clusters.depths.size:_} clusters"
            axs[0].set(ylim=ylim, title=title_str)
            run_label = str(Path(collection).relative_to(f'alf/{self.pname}'))
            run_label = "ks2matlab" if run_label == '.' else run_label
            outfile = self.output_directory.joinpath(f"spike_sorting_raster_{run_label}.png")
            set_axis_label_size(axs[0])

            if self.histology_status:
                plot_brain_regions(channels['atlas_id'], channel_depths=channels['axial_um'],
                                   brain_regions=self.brain_regions, display=True, ax=axs[1], title=self.histology_status)
                axs[1].set(ylim=ylim)
                set_axis_label_size(axs[1])
            else:
                remove_axis_outline(axs[1])

            fig.savefig(outfile)
            plt.close(fig)

            return outfile, fig, axs

        output_files = []
        if self.location == 'server':
            assert collection
            spikes = alfio.load_object(self.session_path.joinpath(collection), 'spikes')
            clusters = alfio.load_object(self.session_path.joinpath(collection), 'clusters')
            channels = alfio.load_object(self.session_path.joinpath(collection), 'channels')

            out, fig, axs = plot_driftmap(self, spikes, clusters, channels, collection)
            output_files.append(out)

        else:
            self.histology_status = self.get_histology_status()
            all_here, output_files = self.assert_expected(self.output_files, silent=True)
            spike_sorting_runs = self.one.list_datasets(self.eid, filename='spikes.times.npy', collection=f'alf/{self.pname}*')
            if all_here and len(output_files) == len(spike_sorting_runs):
                return output_files
            logger.info(self.output_directory)
            for run in spike_sorting_runs:
                collection = str(Path(run).parent.as_posix())
                spikes, clusters, channels = load_spike_sorting_fast(
                    eid=self.eid, probe=self.pname, one=self.one, nested=False, collection=collection,
                    dataset_types=['spikes.depths'], brain_regions=self.brain_regions)

                if 'atlas_id' not in channels.keys():
                    channels = self.get_channels('channels', collection)

                out, fig, axs = plot_driftmap(self, spikes, clusters, channels, collection)
                output_files.append(out)

        return output_files

    def get_probe_signature(self):
        input_signature = [('spikes.times.npy', f'alf/{self.pname}*', True),
                           ('spikes.amps.npy', f'alf/{self.pname}*', True),
                           ('spikes.depths.npy', f'alf/{self.pname}*', True),
                           ('clusters.depths.npy', f'alf/{self.pname}*', True),
                           ('channels.localCoordinates.npy', f'alf/{self.pname}*', False),
                           ('channels.mlapdv.npy', f'alf/{self.pname}*', False),
                           ('channels.brainLocationIds_ccf_2017.npy', f'alf/{self.pname}*', False)]
        output_signature = [('spike_sorting_raster*.png', f'snapshot/{self.pname}', True)]
        self.signature = {'input_files': input_signature, 'output_files': output_signature}

    def get_signatures(self, **kwargs):
        files_spikes = Path(self.session_path).joinpath('alf').rglob('spikes.times.npy')
        folder_probes = [f.parent for f in files_spikes]

        full_input_files = []
        for sig in self.signature['input_files']:
            for folder in folder_probes:
                full_input_files.append((sig[0], str(folder.relative_to(self.session_path)), sig[2]))
        if len(full_input_files) != 0:
            self.input_files = full_input_files
        else:
            self.input_files = self.signature['input_files']

        self.output_files = self.signature['output_files']


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
            # Sometimes there is a file but the object is empty
            if data[f'{cam}_{feat}'] is not None and len(data[f'{cam}_{feat}']) == 0:
                logger.warning(f"Object loaded from _ibl_{cam}Camera.{feat} is empty, some plots have to be skipped.")
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
    data['left_pupil'] = data['left_features'].pupilDiameter_smooth if (
        data['left_features'] is not None and not np.all(np.isnan(data['left_features'].pupilDiameter_smooth))
    ) else None
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
        if any([v is None for v in panel[1].values()]) or any([v.values() is None for v in panel[1].values()
                                                               if isinstance(v, dict)]):
            ax.text(.5, .5, f"Data incomplete\n{panel[0].__name__}", color='r', fontweight='bold',
                    fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            plt.axis('off')
        else:
            try:
                panel[0](**panel[1])
            except BaseException:
                logger.error(f'Error in {panel[0].__name__}\n' + traceback.format_exc())
                ax.text(.5, .5, f'Error in \n{panel[0].__name__}', color='r', fontweight='bold',
                        fontsize=12, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig

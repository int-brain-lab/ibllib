"""Mesoscope (timeline) data extraction."""
import numpy as np
import one.alf.io as alfio
import matplotlib.pyplot as plt
from neurodsp.utils import falls

from ibllib.plots.misc import squares, vertical_lines
from ibllib.io.raw_daq_loaders import (load_sync_timeline, timeline_meta2chmap, timeline_get_channel,
                                       correct_counter_discontinuities)
import ibllib.io.extractors.base as extractors_base
from ibllib.io.extractors.default_channel_maps import DEFAULT_MAPS
from ibllib.io.extractors.ephys_fpga import FpgaTrials, WHEEL_TICKS, WHEEL_RADIUS_CM, get_sync_fronts
from ibllib.io.extractors.training_wheel import extract_wheel_moves
from ibllib.io.extractors.camera import attribute_times
from ibllib.io.extractors.ephys_fpga import _assign_events_bpod


def timeline2sync(timeline, chmap=None):
    """
    Extract the sync from a Timeline object.

    Parameters
    ----------
    timeline : dict, one.alf.AlfBunch
        A timeline object with the keys {'timestamps', 'raw', 'meta'}.
    chmap : dict
        An optional map of channel names and their corresponding array column index (NB: index
        must start from 1).

    Returns
    -------
    one.alf.io.AlfBunch
        A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses and
        the corresponding channel numbers.
    dict
        A map of channel names and their corresponding indices.
    """
    if not chmap:  # attempt to extract from the meta file using expected channel names, or use expected channel numbers
        default = DEFAULT_MAPS['mesoscope']['timeline']
        chmap = timeline_meta2chmap(timeline['meta'], include_channels=default.keys()) or default
    sync = load_sync_timeline(timeline, chmap=chmap)
    return sync, chmap


def plot_timeline(timeline, channels=None, raw=True):
    """
    Plot the timeline data.

    Parameters
    ----------
    timeline : one.alf.io.AlfBunch
        The timeline data object.
    channels : list of str
        An iterable of channel names to plot.
    raw : bool
        If true, plot the raw DAQ samples; if false, apply TTL thresholds and plot changes.

    Returns
    -------
    matplotlib.pyplot.Figure
        The figure containing timeline subplots.
    list of matplotlib.pyplot.Axes
        The axes for each timeline channel plotted.
    """
    meta = {x.copy().pop('name'): x for x in timeline['meta']['inputs']}
    channels = channels or meta.keys()
    fig, axes = plt.subplots(len(channels), 1)
    if not raw:
        chmap = {ch: meta[ch]['arrayColumn'] for ch in channels}
        sync, chmap = timeline2sync(timeline, chmap)
    for i, (ax, ch) in enumerate(zip(axes, channels)):
        if raw:
            # axesScale controls vertical scaling of each trace (multiplicative)
            values = timeline['raw'][:, meta[ch]['arrayColumn'] - 1] * meta[ch]['axesScale']
            ax.plot(timeline['timestamps'], values)
        elif np.any(idx := sync['channels'] == chmap[ch]):
            squares(sync['times'][idx], sync['polarities'][idx], ax=ax)
        ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        ax.spines['bottom'].set_visible(False), ax.spines['left'].set_visible(True)
        ax.set_ylabel(ch, rotation=45, fontsize=8)
    # Add back x-axis ticks to the last plot
    axes[-1].tick_params(axis='x', which='both', bottom=True, labelbottom=True)
    axes[-1].spines['bottom'].set_visible(True)
    plt.get_current_fig_manager().window.showMaximized()  # full screen
    fig.tight_layout(h_pad=0)
    return fig, axes


class TimelineTrials(FpgaTrials):
    """Similar extraction to the FPGA, however counter and position channels are treated differently."""

    """one.alf.io.AlfBunch: The timeline data object"""
    timeline = None

    def __init__(self, *args, sync_collection='raw_sync_data', **kwargs):
        """An extractor for all ephys trial data, in Timeline time"""
        super().__init__(*args, **kwargs)
        self.timeline = alfio.load_object(self.session_path / sync_collection, 'DAQData', namespace='timeline')

    def _extract(self, sync=None, chmap=None, sync_collection='raw_sync_data', **kwargs):
        if not (sync or chmap):
            sync, chmap = timeline2sync(self.timeline)
        if kwargs.get('display', False):
            plot_timeline(self.timeline, channels=chmap.keys(), raw=True)
        trials = super()._extract(sync, chmap, sync_collection, extractor_type='ephys', **kwargs)

        # Replace valve open times with those extracted from the DAQ
        # TODO Let's look at the expected open length based on calibration and reward volume
        mask = sync['channels'] == chmap['bpod']
        _, driver_out, _, = _assign_events_bpod(sync['times'][mask], sync['polarities'][mask], False)
        # Use the driver TTLs to find the valve open times that correspond to the valve opening
        valve_open_times = self.get_valve_open_times(driver_ttls=driver_out)
        trials_table = trials[self.var_names.index('table')]
        assert len(valve_open_times) == sum(trials_table.feedbackType == 1)  # TODO Relax assertion
        correct = trials_table.feedbackType == 1
        trials[self.var_names.index('valveOpen_times')][correct] = valve_open_times
        trials_table.feedback_times[correct] = valve_open_times

        # Replace audio events
        go_cue, error_cue = self._assign_events_audio(sync, chmap)
        assert go_cue.size == len(trials_table)
        assert error_cue.size == np.sum(~correct)
        trials_table.feedback_times[~correct] = error_cue
        trials_table.goCue_times = go_cue
        return trials

    def get_wheel_positions(self, ticks=WHEEL_TICKS, radius=WHEEL_RADIUS_CM, coding='x4', display=False, **kwargs):
        """
        Gets the wheel position from Timeline counter channel.

        Parameters
        ----------
        ticks : int
            Number of ticks corresponding to a full revolution (1024 for IBL rotary encoder)
        radius : float
            Radius of the wheel. Defaults to 1 for an output in radians
        coding : str {'x1', 'x2', 'x4'}
            Rotary encoder encoding (IBL default is x4)

        Returns
        -------
        dict
            wheel object with keys ('timestamps', 'position')
        dict
            wheelMoves object with keys ('intervals' 'peakAmplitude')

        FIXME Support spacers
        """
        if coding not in ('x1', 'x2', 'x4'):
            raise ValueError('Unsupported coding; must be one of x1, x2 or x4')
        info = next(x for x in self.timeline['meta']['inputs'] if x['name'].lower() == 'rotary_encoder')
        raw = self.timeline['raw'][:, info['arrayColumn'] - 1]  # -1 because MATLAB indexes from 1
        raw = correct_counter_discontinuities(raw)

        # Timeline evenly samples counter so we extract only change points
        d = np.diff(raw)
        ind, = np.where(d.astype(int))
        pos = raw[ind + 1]
        pos -= pos[0]  # Start from zero
        pos = pos / ticks * np.pi * 2 * radius / int(coding[1])  # Convert to radians

        wheel = {'timestamps': self.timeline['timestamps'][ind + 1], 'position': pos}
        moves = extract_wheel_moves(wheel['timestamps'], wheel['position'])

        if display:
            fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
            bpod_ts = self.bpod_trials['wheel_timestamps']
            bpod_pos = self.bpod_trials['wheel_position']
            ax0.plot(self.bpod2fpga(bpod_ts), bpod_pos)
            ax0.set_ylabel('Bpod wheel position / rad')
            ax1.plot(wheel['timestamps'], wheel['position'])
            ax1.set_ylabel('DAQ wheel position / rad'), ax1.set_xlabel('Time / s')
        return wheel, moves

    def get_valve_open_times(self, display=False, threshold=-2.5, floor_percentile=10, driver_ttls=None):
        """
        Get the valve open times from the raw timeline voltage trace.

        Parameters
        ----------
        display : bool
            Plot detected times on the raw voltage trace.
        threshold : float
            The threshold for applying to analogue channels.
        floor_percentile : float
            10% removes the percentile value of the analog trace before thresholding. This is to
            avoid DC offset drift.
        driver_ttls : numpy.array
            An optional array of driver TTLs to use for assigning with the valve times.

        Returns
        -------
        numpy.array
            The detected valve open times.

        FIXME Support spacers
        TODO extract close times too
        """
        tl = self.timeline
        info = next(x for x in tl['meta']['inputs'] if x['name'] == 'reward_valve')
        values = tl['raw'][:, info['arrayColumn'] - 1]  # Timeline indices start from 1
        offset = np.percentile(values, floor_percentile, axis=0)
        idx = falls(values - offset, step=threshold)  # Voltage falls when valve opens
        open_times = tl['timestamps'][idx]
        # The closing of the valve is noisy. Keep only the falls that occur immediately after a Bpod TTL
        if driver_ttls is not None:
            # Returns an array of open_times indices, one for each driver TTL
            ind = attribute_times(open_times, driver_ttls, tol=.1, take='after')
            open_times = open_times[ind[ind >= 0]]
            # TODO Log any > 40ms? Difficult to report missing valve times because of calibration

        if display:
            fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True)
            ax0.plot(tl['timestamps'], timeline_get_channel(tl, 'bpod'), 'k-o')
            if driver_ttls is not None:
                vertical_lines(driver_ttls, ymax=5, ax=ax0, linestyle='--', color='b')
            ax1.plot(tl['timestamps'], values - offset, 'k-o')
            ax1.set_ylabel('Voltage / V'), ax1.set_xlabel('Time / s')
            ax1.plot(tl['timestamps'][idx], np.zeros_like(idx), 'r*')
            if driver_ttls is not None:
                ax1.plot(open_times, np.zeros_like(open_times), 'g*')
        return open_times

    def _assign_events_audio(self, sync, chmap, display=False):
        """
        This is identical to ephys_fpga._assign_events_audio, except for the ready tone threshold.
        TODO Make DRY!
        Parameters
        ----------
        display

        Returns
        -------

        """
        audio = get_sync_fronts(sync, chmap['audio'])  # FIXME Need to support spacers

        # make sure that there are no 2 consecutive fall or consecutive rise events
        assert np.all(np.abs(np.diff(audio['polarities'])) == 2)
        # take only even time differences: ie. from rising to falling fronts
        dt = np.diff(audio['times'])

        # error tones are events lasting from 400ms to 1200ms
        i_error_tone_in = np.where(np.logical_and(np.logical_and(0.4 < dt, dt < 1.2), audio['polarities'][:-1] == 1))[0]
        t_error_tone_in = audio['times'][i_error_tone_in]

        # detect ready tone by length below 300 ms
        i_ready_tone_in = np.where(np.logical_and(dt <= 0.3, audio['polarities'][:-1] == 1))[0]
        t_ready_tone_in = audio['times'][i_ready_tone_in]
        if display:  # pragma: no cover
            fig, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].plot(self.timeline.timestamps, timeline_get_channel(self.timeline, 'audio'), 'k-o')
            squares(audio['times'], audio['polarities'], yrange=[-1, 1], ax=ax[1])
            vertical_lines(t_ready_tone_in, ymin=-.8, ymax=.8, ax=ax[1])
            vertical_lines(t_error_tone_in, ymin=-.8, ymax=.8, ax=ax[1])

        return t_ready_tone_in, t_error_tone_in


class MesoscopeSyncTimeline(extractors_base.BaseExtractor):
    """Extraction of mesoscope imaging times."""

    var_names = ('mpci_times', 'mpciStack_timeshift')
    save_names = ('mpci.times.npy', 'mpciStack.timeshift.npy')

    """one.alf.io.AlfBunch: The timeline data object"""
    rawImagingData = None  # TODO Document

    def __init__(self, session_path, n_ROIs, **kwargs):
        super().__init__(session_path, **kwargs)  # TODO Document
        rois = list(map(lambda n: f'ROI{n:02}', range(n_ROIs)))
        self.var_names = [f'{x}_{y.lower()}' for x in self.var_names for y in rois]
        self.save_names = [f'{y}/{x}' for x in self.save_names for y in rois]

    def _extract(self, sync=None, chmap=None, device_collection='raw_mesoscope_data', bout=None):
        """
        Extract the frame timestamps for each individual field of view (FOV) and the time offsets
        for each line scan.

        Parameters
        ----------
        sync : one.alf.io.AlfBunch
            A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses
            and the corresponding channel numbers.
        chmap : dict
            A map of channel names and their corresponding indices. Only the 'neural_frames'
            channel is required.
        device_collection : str
            The location of the raw imaging data.
        bout : int
            The imagining bout number, from 0-n, corresponding to raw_imaging_data folders.

        Returns
        -------
        list of numpy.array
            A list of timestamps for each FOV and the time offsets for each line scan.
        """
        self.rawImagingData = alfio.load_object(self.session_path / device_collection, 'rawImagingData')

        frame_times = sync['times'][sync['channels'] == chmap['neural_frames']]
        assert frame_times.size == self.rawImagingData.times_scanImage.size

        # imaging_start_time = datetime.datetime(*map(round, self.rawImagingData.meta['acquisitionStartTime']))
        # TODO Extract UDP messages for imaging bouts

        # Calculate line shifts
        _, fov_time_shifts, line_time_shifts = self.get_timeshifts()
        fov_times = [frame_times + offset for offset in fov_time_shifts]

        return fov_times + line_time_shifts

    def get_timeshifts(self):
        # TODO Document
        FOVs = self.rawImagingData.meta['FOV']

        # Double-check meta extracted properly
        raw_meta = self.rawImagingData.meta['rawScanImageMeta']
        artist = raw_meta['Artist']
        # TODO This assertion might need to be removed if some ROIs are deactivated
        assert len(artist['RoiGroups']['imagingRoiGroup']['rois']) == len(FOVs)

        # Number of scan lines per FOV, i.e. number of Y pixels / image height
        n_lines = np.array([x['nXnYnZ'][1] for x in FOVs])
        n_valid_lines = np.sum(n_lines)  # Number of lines imaged excluding flybacks
        # Number of lines during flyback
        n_lines_per_gap = int((raw_meta['Height'] - n_valid_lines) / (len(FOVs) - 1));
        # The start and end indices of each FOV in the raw images
        fov_start_idx = np.insert(np.cumsum(n_lines[:-1] + n_lines_per_gap), 0, 0)
        fov_end_idx = fov_start_idx + n_lines
        line_period = self.rawImagingData.meta['scanImageParams']['hRoiManager']['linePeriod']

        line_indices = []
        fov_time_shifts = fov_start_idx * line_period
        line_time_shifts = []

        for ln, s, e in zip(n_lines, fov_start_idx, fov_end_idx):
            line_indices.append(np.arange(s, e))
            line_time_shifts.append(np.arange(0, ln) * line_period)

        return line_indices, fov_time_shifts, line_time_shifts


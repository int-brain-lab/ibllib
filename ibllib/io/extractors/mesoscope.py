"""Mesoscope (timeline) data extraction."""
import logging

import numpy as np
import one.alf.io as alfio
from one.util import ensure_list
from one.alf.files import session_path_parts
import matplotlib.pyplot as plt
from neurodsp.utils import falls
from pkg_resources import parse_version

from ibllib.plots.misc import squares, vertical_lines
from ibllib.io.raw_daq_loaders import (extract_sync_timeline, timeline_get_channel,
                                       correct_counter_discontinuities, load_timeline_sync_and_chmap)
import ibllib.io.extractors.base as extractors_base
from ibllib.io.extractors.ephys_fpga import FpgaTrials, WHEEL_TICKS, WHEEL_RADIUS_CM, get_sync_fronts, get_protocol_period
from ibllib.io.extractors.training_wheel import extract_wheel_moves
from ibllib.io.extractors.camera import attribute_times
from ibllib.io.extractors.ephys_fpga import _assign_events_bpod

_logger = logging.getLogger(__name__)


def patch_imaging_meta(meta: dict) -> dict:
    """
    Patch imaging meta data for compatibility across versions.

    A copy of the dict is NOT returned.

    Parameters
    ----------
    dict : dict
        A folder path that contains a rawImagingData.meta file.

    Returns
    -------
    dict
        The loaded meta data file, updated to the most recent version.
    """
    # 2023-05-17 (unversioned) adds nFrames and channelSaved keys
    if parse_version(meta.get('version') or '0.0.0') <= parse_version('0.0.0'):
        if 'channelSaved' not in meta:
            meta['channelSaved'] = next((x['channelIdx'] for x in meta['FOV'] if 'channelIdx' in x), [])
    return meta


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
    fig, axes = plt.subplots(len(channels), 1, sharex=True)
    axes = ensure_list(axes)
    if not raw:
        chmap = {ch: meta[ch]['arrayColumn'] for ch in channels}
        sync = extract_sync_timeline(timeline, chmap=chmap)
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
        self.timeline = alfio.load_object(self.session_path / sync_collection, 'DAQdata', namespace='timeline')

    def _extract(self, sync=None, chmap=None, sync_collection='raw_sync_data', **kwargs):
        if not (sync or chmap):
            sync, chmap = load_timeline_sync_and_chmap(
                self.session_path / sync_collection, timeline=self.timeline, chmap=chmap)

        if kwargs.get('display', False):
            plot_timeline(self.timeline, channels=chmap.keys(), raw=True)
        trials = super()._extract(sync, chmap, sync_collection, extractor_type='ephys', **kwargs)

        # If no protocol number is defined, trim timestamps based on Bpod trials intervals
        trials_table = trials[self.var_names.index('table')]
        bpod = get_sync_fronts(sync, chmap['bpod'])
        if kwargs.get('protocol_number') is None:
            tmin = trials_table.intervals_0.iloc[0] - 1
            tmax = trials_table.intervals_1.iloc[-1]
            # Ensure wheel is cut off based on trials
            wheel_ts_idx = self.var_names.index('wheel_timestamps')
            mask = np.logical_and(tmin <= trials[wheel_ts_idx], trials[wheel_ts_idx] <= tmax)
            trials[wheel_ts_idx] = trials[wheel_ts_idx][mask]
            wheel_pos_idx = self.var_names.index('wheel_position')
            trials[wheel_pos_idx] = trials[wheel_pos_idx][mask]
            move_idx = self.var_names.index('wheelMoves_intervals')
            mask = np.logical_and(trials[move_idx][:, 0] >= tmin, trials[move_idx][:, 0] <= tmax)
            trials[move_idx] = trials[move_idx][mask, :]
        else:
            tmin, tmax = get_protocol_period(self.session_path, kwargs['protocol_number'], bpod)
        bpod = get_sync_fronts(sync, chmap['bpod'], tmin, tmax)

        self.frame2ttl = get_sync_fronts(sync, chmap['frame2ttl'], tmin, tmax)  # save for later access by QC

        # Replace valve open times with those extracted from the DAQ
        # TODO Let's look at the expected open length based on calibration and reward volume
        assert len(bpod['times']) > 0, 'No Bpod TTLs detected on DAQ'
        _, driver_out, _, = _assign_events_bpod(bpod['times'], bpod['polarities'], False)
        # Use the driver TTLs to find the valve open times that correspond to the valve opening
        valve_open_times = self.get_valve_open_times(driver_ttls=driver_out)
        assert len(valve_open_times) == sum(trials_table.feedbackType == 1)  # TODO Relax assertion
        correct = trials_table.feedbackType == 1
        trials[self.var_names.index('valveOpen_times')][correct] = valve_open_times
        trials_table.feedback_times[correct] = valve_open_times

        # Replace audio events
        self.audio = get_sync_fronts(sync, chmap['audio'], tmin, tmax)
        # Attempt to assign the go cue and error tone onsets based on TTL length
        go_cue, error_cue = self._assign_events_audio(self.audio['times'], self.audio['polarities'])

        assert error_cue.size == np.sum(~correct), 'N detected error tones does not match number of incorrect trials'
        assert go_cue.size <= len(trials_table), 'More go cue tones detected than trials!'

        if go_cue.size < len(trials_table):
            _logger.warning('%i go cue tones missed', len(trials_table) - go_cue.size)
            """
            If the error cues are all assigned and some go cues are missed it may be that some
            responses were so fast that the go cue and error tone merged.
            """
            err_trig = self.bpod2fpga(self.bpod_trials['errorCueTrigger_times'])
            go_trig = self.bpod2fpga(self.bpod_trials['goCueTrigger_times'])
            assert not np.any(np.isnan(go_trig))
            assert err_trig.size == go_trig.size

            def first_true(arr):
                """Return the index of the first True value in an array."""
                indices = np.where(arr)[0]
                return None if len(indices) == 0 else indices[0]

            # Find which trials are missing a go cue
            _go_cue = np.full(len(trials_table), np.nan)
            for i, intervals in enumerate(trials_table[['intervals_0', 'intervals_1']].values):
                idx = first_true(np.logical_and(go_cue > intervals[0], go_cue < intervals[1]))
                if idx is not None:
                    _go_cue[i] = go_cue[idx]

            # Get all the DAQ timestamps where audio channel was HIGH
            raw = timeline_get_channel(self.timeline, 'audio')
            raw = (raw - raw.min()) / (raw.max() - raw.min())  # min-max normalize
            ups = self.timeline.timestamps[raw > .5]  # timestamps where input HIGH
            for i in np.where(np.isnan(_go_cue))[0]:
                # Get the timestamp of the first HIGH after the trigger times
                _go_cue[i] = ups[first_true(ups > go_trig[i])]
                idx = first_true(np.logical_and(
                    error_cue > trials_table['intervals_0'][i],
                    error_cue < trials_table['intervals_1'][i]))
                if np.isnan(err_trig[i]):
                    if idx is not None:
                        error_cue = np.delete(error_cue, idx)  # Remove mis-assigned error tone time
                else:
                    error_cue[idx] = ups[first_true(ups > err_trig[i])]
            go_cue = _go_cue

        trials_table.feedback_times[~correct] = error_cue
        trials_table.goCue_times = go_cue
        return trials

    def extract_wheel_sync(self, ticks=WHEEL_TICKS, radius=WHEEL_RADIUS_CM, coding='x4', tmin=None, tmax=None):
        """
        Gets the wheel position from Timeline counter channel.

        Parameters
        ----------
        ticks : int
            Number of ticks corresponding to a full revolution (1024 for IBL rotary encoder).
        radius : float
            Radius of the wheel. Defaults to 1 for an output in radians.
        coding : str {'x1', 'x2', 'x4'}
            Rotary encoder encoding (IBL default is x4).
        tmin : float
            The minimum time from which to extract the sync pulses.
        tmax : float
            The maximum time up to which we extract the sync pulses.

        Returns
        -------
        np.array
            Wheel timestamps in seconds.
        np.array
            Wheel positions in radians.

        See Also
        --------
        ibllib.io.extractors.ephys_fpga.extract_wheel_sync
        """
        if coding not in ('x1', 'x2', 'x4'):
            raise ValueError('Unsupported coding; must be one of x1, x2 or x4')
        raw = correct_counter_discontinuities(timeline_get_channel(self.timeline, 'rotary_encoder'))

        # Timeline evenly samples counter so we extract only change points
        d = np.diff(raw)
        ind, = np.where(d.astype(int))
        pos = raw[ind + 1]
        pos -= pos[0]  # Start from zero
        pos = pos / ticks * np.pi * 2 * radius / int(coding[1])  # Convert to radians

        # Get timestamps of changes and trim based on protocol spacers
        ts = self.timeline['timestamps'][ind + 1]
        tmin = ts.min() if tmin is None else tmin
        tmax = ts.max() if tmax is None else tmax
        mask = np.logical_and(ts >= tmin, ts <= tmax)
        return ts[mask], pos[mask]

    def get_wheel_positions(self, ticks=WHEEL_TICKS, radius=WHEEL_RADIUS_CM, coding='x4',
                            tmin=None, tmax=None, display=False, **kwargs):
        """
        Gets the wheel position and detected movements from Timeline counter channel.

        Called by the super class extractor (FPGATrials._extract).

        Parameters
        ----------
        ticks : int
            Number of ticks corresponding to a full revolution (1024 for IBL rotary encoder).
        radius : float
            Radius of the wheel. Defaults to 1 for an output in radians.
        coding : str {'x1', 'x2', 'x4'}
            Rotary encoder encoding (IBL default is x4).
        tmin : float
            The minimum time from which to extract the sync pulses.
        tmax : float
            The maximum time up to which we extract the sync pulses.
        display : bool
            If true, plot the wheel positions from bpod and the DAQ.

        Returns
        -------
        dict
            wheel object with keys ('timestamps', 'position').
        dict
            wheelMoves object with keys ('intervals' 'peakAmplitude').
        """
        wheel = self.extract_wheel_sync(ticks=ticks, radius=radius, coding=coding, tmin=tmin, tmax=tmax)
        wheel = dict(zip(('timestamps', 'position'), wheel))
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

    def _assign_events_audio(self, audio_times, audio_polarities, display=False):
        """
        This is identical to ephys_fpga._assign_events_audio, except for the ready tone threshold.

        Parameters
        ----------
        audio_times : numpy.array
            An array of audio TTL front times.
        audio_polarities : numpy.array
            An array of audio TTL front polarities (1 for rises, -1 for falls).
        display : bool
            If true, display audio pulses and the assigned onsets.

        Returns
        -------
        numpy.array
            The times of the go cue onsets.
        numpy.array
            The times of the error tone onsets.
        """
        # make sure that there are no 2 consecutive fall or consecutive rise events
        assert np.all(np.abs(np.diff(audio_polarities)) == 2)
        # take only even time differences: ie. from rising to falling fronts
        dt = np.diff(audio_times)
        onsets = audio_polarities[:-1] == 1

        # error tones are events lasting from 400ms to 1200ms
        i_error_tone_in = np.where(np.logical_and(0.4 < dt, dt < 1.2) & onsets)[0]
        t_error_tone_in = audio_times[i_error_tone_in]

        # detect ready tone by length below 300 ms
        i_ready_tone_in = np.where(np.logical_and(dt <= 0.3, onsets))[0]
        t_ready_tone_in = audio_times[i_ready_tone_in]
        if display:  # pragma: no cover
            fig, ax = plt.subplots(nrows=2, sharex=True)
            ax[0].plot(self.timeline.timestamps, timeline_get_channel(self.timeline, 'audio'), 'k-o')
            ax[0].set_ylabel('Voltage / V')
            squares(audio_times, audio_polarities, yrange=[-1, 1], ax=ax[1])
            vertical_lines(t_ready_tone_in, ymin=-.8, ymax=.8, ax=ax[1], label='go cue')
            vertical_lines(t_error_tone_in, ymin=-.8, ymax=.8, ax=ax[1], label='error tone')
            ax[1].set_xlabel('Time / s')
            ax[1].legend()

        return t_ready_tone_in, t_error_tone_in


class MesoscopeSyncTimeline(extractors_base.BaseExtractor):
    """Extraction of mesoscope imaging times."""

    var_names = ('mpci_times', 'mpciStack_timeshift')
    save_names = ('mpci.times.npy', 'mpciStack.timeshift.npy')

    """one.alf.io.AlfBunch: The raw imaging meta data and frame times"""
    rawImagingData = None

    def __init__(self, session_path, n_FOVs):
        """
        Extract the mesoscope frame times from DAQ data acquired through Timeline.

        Parameters
        ----------
        session_path : str, pathlib.Path
            The session path to extract times from.
        n_FOVs : int
            The number of fields of view acquired.
        """
        super().__init__(session_path)
        self.n_FOVs = n_FOVs
        fov = list(map(lambda n: f'FOV_{n:02}', range(self.n_FOVs)))
        self.var_names = [f'{x}_{y.lower()}' for x in self.var_names for y in fov]
        self.save_names = [f'{y}/{x}' for x in self.save_names for y in fov]

    def _extract(self, sync=None, chmap=None, device_collection='raw_imaging_data', events=None):
        """
        Extract the frame timestamps for each individual field of view (FOV) and the time offsets
        for each line scan.

        The detected frame times from the 'neural_frames' channel of the DAQ are split into bouts
        corresponding to the number of raw_imaging_data folders. These timestamps should match the
        number of frame timestamps extracted from the image file headers (found in the
        rawImagingData.times file).  The field of view (FOV) shifts are then applied to these
        timestamps for each field of view and provided together with the line shifts.

        Parameters
        ----------
        sync : one.alf.io.AlfBunch
            A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses
            and the corresponding channel numbers.
        chmap : dict
            A map of channel names and their corresponding indices. Only the 'neural_frames'
            channel is required.
        device_collection : str, iterable of str
            The location of the raw imaging data.
        events : pandas.DataFrame
            A table of software events, with columns {'time_timeline' 'name_timeline',
            'event_timeline'}.

        Returns
        -------
        list of numpy.array
            A list of timestamps for each FOV and the time offsets for each line scan.
        """
        frame_times = sync['times'][sync['channels'] == chmap['neural_frames']]

        # imaging_start_time = datetime.datetime(*map(round, self.rawImagingData.meta['acquisitionStartTime']))
        if isinstance(device_collection, str):
            device_collection = [device_collection]
        if events is not None:
            events = events[events.name == 'mpepUDP']
        edges = self.get_bout_edges(frame_times, device_collection, events)
        fov_times = []
        line_shifts = []
        for (tmin, tmax), collection in zip(edges, sorted(device_collection)):
            imaging_data = alfio.load_object(self.session_path / collection, 'rawImagingData')
            imaging_data['meta'] = patch_imaging_meta(imaging_data['meta'])
            # Calculate line shifts
            _, fov_time_shifts, line_time_shifts = self.get_timeshifts(imaging_data['meta'])
            assert len(fov_time_shifts) == self.n_FOVs, f'unexpected number of FOVs for {collection}'
            ts = frame_times[np.logical_and(frame_times >= tmin, frame_times <= tmax)]
            assert ts.size >= imaging_data['times_scanImage'].size, f'fewer DAQ timestamps for {collection} than expected'
            if ts.size > imaging_data['times_scanImage'].size:
                _logger.warning(
                    'More DAQ frame times detected for %s than were found in the raw image data.\n'
                    'N DAQ frame times:\t%i\nN raw image data times:\t%i.\n'
                    'This may occur if the bout detection fails (e.g. UDPs recorded late), '
                    'when image data is corrupt, or when frames are not written to file.',
                    collection, ts.size, imaging_data['times_scanImage'].size)
                _logger.info('Dropping last %i frame times for %s', ts.size - imaging_data['times_scanImage'].size, collection)
                ts = ts[:imaging_data['times_scanImage'].size]
            fov_times.append([ts + offset for offset in fov_time_shifts])
            if not line_shifts:
                line_shifts = line_time_shifts
            else:  # The line shifts should be the same across all imaging bouts
                [np.testing.assert_array_equal(x, y) for x, y in zip(line_time_shifts, line_shifts)]

        # Concatenate imaging timestamps across all bouts for each field of view
        fov_times = list(map(np.concatenate, zip(*fov_times)))
        n_fov_times, = set(map(len, fov_times))
        if n_fov_times != frame_times.size:
            # This may happen if an experimenter deletes a raw_imaging_data folder
            _logger.debug('FOV timestamps length does not match neural frame count; imaging bout(s) likely missing')
        return fov_times + line_shifts

    def get_bout_edges(self, frame_times, collections=None, events=None, min_gap=1., display=False):
        """
        Return an array of edge times for each imaging bout corresponding to a raw_imaging_data
        collection.

        Parameters
        ----------
        frame_times : numpy.array
            An array of all neural frame count times.
        collections : iterable of str
            A set of raw_imaging_data collections, used to extract selected imaging periods.
        events : pandas.DataFrame
            A table of UDP event times, corresponding to times when recordings start and end.
        min_gap : float
            If start or end events not present, split bouts by finding gaps larger than this value.
        display : bool
            If true, plot the detected bout edges and raw frame times.

        Returns
        -------
        numpy.array
            An array of imaging bout intervals.
        """
        if events is None or events.empty:
            # No UDP events to mark blocks so separate based on gaps in frame rate
            idx = np.where(np.diff(frame_times) > min_gap)[0]
            starts = np.r_[frame_times[0], frame_times[idx + 1]]
            ends = np.r_[frame_times[idx], frame_times[-1]]
        else:
            # Split using Exp/BlockStart and Exp/BlockEnd times
            _, subject, date, _ = session_path_parts(self.session_path)
            pattern = rf'(Exp|Block)%s\s{subject}\s{date.replace("-", "")}\s\d+'

            # Get start times
            UDP_start = events[events['info'].str.match(pattern % 'Start')]
            if len(UDP_start) > 1 and UDP_start.loc[0, 'info'].startswith('Exp'):
                # Use ExpStart instead of first bout start
                UDP_start = UDP_start.copy().drop(1)
            # Use ExpStart/End instead of first/last BlockStart/End
            starts = frame_times[[np.where(frame_times >= t)[0][0] for t in UDP_start.time]]

            # Get end times
            UDP_end = events[events['info'].str.match(pattern % 'End')]
            if len(UDP_end) > 1 and UDP_end['info'].values[-1].startswith('Exp'):
                # Use last BlockEnd instead of ExpEnd
                UDP_end = UDP_end.copy().drop(UDP_end.index[-1])
            if not UDP_end.empty:
                ends = frame_times[[np.where(frame_times <= t)[0][-1] for t in UDP_end.time]]
            else:
                # Get index of last frame to occur within a second of the previous frame
                consec = np.r_[np.diff(frame_times) > min_gap, True]
                idx = [np.where(np.logical_and(frame_times > t, consec))[0][0] for t in starts]
                ends = frame_times[idx]

        # Remove any missing imaging bout collections
        edges = np.c_[starts, ends]
        if collections:
            if edges.shape[0] > len(collections):
                # Remove any bouts that correspond to a skipped collection
                # e.g. if {raw_imaging_data_00, raw_imaging_data_02}, remove middle bout
                include = sorted(int(c.rsplit('_', 1)[-1]) for c in collections)
                edges = edges[include, :]
            elif edges.shape[0] < len(collections):
                raise ValueError('More raw imaging folders than detected bouts')

        if display:
            _, ax = plt.subplots(1)
            ax.step(frame_times, np.arange(frame_times.size), label='frame times', color='k', )
            vertical_lines(edges[:, 0], ax=ax, ymin=0, ymax=frame_times.size, label='bout start', color='b')
            vertical_lines(edges[:, 1], ax=ax, ymin=0, ymax=frame_times.size, label='bout end', color='orange')
            if edges.shape[0] != len(starts):
                vertical_lines(np.setdiff1d(starts, edges[:, 0]), ax=ax, ymin=0, ymax=frame_times.size,
                               label='missing bout start', linestyle=':', color='b')
                vertical_lines(np.setdiff1d(ends, edges[:, 1]), ax=ax, ymin=0, ymax=frame_times.size,
                               label='missing bout end', linestyle=':', color='orange')
            ax.set_xlabel('Time / s'), ax.set_ylabel('Frame #'), ax.legend(loc='lower right')
        return edges

    @staticmethod
    def get_timeshifts(raw_imaging_meta):
        """
        Calculate the time shifts for each field of view (FOV) and the relative offsets for each
        scan line.

        Parameters
        ----------
        raw_imaging_meta : dict
            Extracted ScanImage meta data (_ibl_rawImagingData.meta.json).

        Returns
        -------
        list of numpy.array
            A list of arrays, one per FOV, containing indices of each image scan line.
        numpy.array
            An array of FOV time offsets (one value per FOV) relative to each frame acquisition
            time.
        list of numpy.array
            A list of arrays, one per FOV, containing the time offsets for each scan line, relative
            to each FOV offset.
        """
        FOVs = raw_imaging_meta['FOV']

        # Double-check meta extracted properly
        raw_meta = raw_imaging_meta['rawScanImageMeta']
        artist = raw_meta['Artist']
        assert sum(x['enable'] for x in artist['RoiGroups']['imagingRoiGroup']['rois']) == len(FOVs)

        # Number of scan lines per FOV, i.e. number of Y pixels / image height
        n_lines = np.array([x['nXnYnZ'][1] for x in FOVs])
        n_valid_lines = np.sum(n_lines)  # Number of lines imaged excluding flybacks
        # Number of lines during flyback
        n_lines_per_gap = int((raw_meta['Height'] - n_valid_lines) / (len(FOVs) - 1))
        # The start and end indices of each FOV in the raw images
        fov_start_idx = np.insert(np.cumsum(n_lines[:-1] + n_lines_per_gap), 0, 0)
        fov_end_idx = fov_start_idx + n_lines
        line_period = raw_imaging_meta['scanImageParams']['hRoiManager']['linePeriod']

        line_indices = []
        fov_time_shifts = fov_start_idx * line_period
        line_time_shifts = []

        for ln, s, e in zip(n_lines, fov_start_idx, fov_end_idx):
            line_indices.append(np.arange(s, e))
            line_time_shifts.append(np.arange(0, ln) * line_period)

        return line_indices, fov_time_shifts, line_time_shifts

"""Mesoscope (timeline) trial extraction."""
import numpy as np
import one.alf.io as alfio
import matplotlib.pyplot as plt
from neurodsp.utils import falls

from ibllib.plots.misc import squares
from ibllib.io.raw_daq_loaders import load_sync_timeline, timeline_meta2chmap
from ibllib.io.extractors.default_channel_maps import DEFAULT_MAPS
from ibllib.io.extractors.ephys_fpga import FpgaTrials, WHEEL_TICKS, WHEEL_RADIUS_CM
from ibllib.io.extractors.training_wheel import extract_wheel_moves


def _timeline2sync(timeline, chmap=None):
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
        sync, chmap = _timeline2sync(timeline, chmap)
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
    """Similar extraction to the FPGA, however counter and position channels are treated differently"""

    """one.alf.io.AlfBunch: The timeline data object"""
    timeline = None

    def __init__(self, *args, sync_collection='raw_sync_data', **kwargs):
        """An extractor for all ephys trial data, in Timeline time"""
        super().__init__(*args, **kwargs)
        self.timeline = alfio.load_object(self.session_path / sync_collection, 'DAQdata', namespace='timeline')

    def _extract(self, sync=None, chmap=None, sync_collection='raw_sync_data', **kwargs):
        if not (sync or chmap):
            sync, chmap = _timeline2sync(self.timeline)
        if kwargs.get('display', False):
            plot_timeline(self.timeline, channels=chmap.keys(), raw=True)
        trials = super()._extract(sync, chmap, sync_collection, extractor_type='ephys', **kwargs)
        # Replace valve open times with those extracted from the DAQ
        trials[self.var_names.index('valveOpen_times')] = self.get_valve_open_times()
        return trials

    def get_wheel_positions(self, ticks=WHEEL_TICKS, radius=WHEEL_RADIUS_CM, coding='x4'):
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
        """
        if coding not in ('x1', 'x2', 'x4'):
            raise ValueError('Unsupported coding; must be one of x1, x2 or x4')
        info = next(x for x in self.timeline['meta']['inputs'] if x['name'].lower() == 'rotary_encoder')
        raw = self.timeline['raw'][:, info['arrayColumn'] - 1]  # -1 because MATLAB indexes from 1
        # Timeline evenly samples counter so we extract only change points
        d = np.diff(raw)
        ind, = np.where(d.astype(int))
        pos = raw[ind + 1]
        pos -= pos[0]  # Start from zero
        pos = pos / ticks * np.pi * 2 * radius / int(coding[1])  # Convert to radians

        wheel = {'timestamps': self.timeline['timestamps'][ind + 1], 'position': pos}
        moves = extract_wheel_moves(wheel['timestamps'], wheel['position'])
        return wheel, moves

    def get_valve_open_times(self, display=False, threshold=-2.5, floor_percentile=10):
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

        Returns
        -------
        numpy.array
            The detected valve open times.
        """
        info = next(x for x in self.timeline['meta']['inputs'] if x['name'] == 'reward_valve')
        values = self.timeline['raw'][:, info['arrayColumn'] - 1]  # Timeline indices start from 1
        offset = np.percentile(values, floor_percentile, axis=0)
        idx = falls(values - offset, step=threshold)
        open_times = self.timeline['timestamps'][idx]
        if display:
            fig, ax = plt.subplots()
            ax.plot(self.timeline['timestamps'], values - offset)
            ax.plot(open_times, np.zeros_like(idx), 'r*')
            ax.set_ylabel('Voltage / V'), ax.set_xlabel('Time / s')
        return open_times

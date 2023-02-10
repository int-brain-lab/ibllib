import numpy as np
import one.alf.io as alfio

from ibllib.io.raw_daq_loaders import load_sync_timeline, timeline_meta2chmap

from ibllib.io.extractors.default_channel_maps import DEFAULT_MAPS
from ibllib.io.extractors.ephys_fpga import FpgaTrials, WHEEL_TICKS, WHEEL_RADIUS_CM
from ibllib.io.extractors.training_wheel import extract_wheel_moves


def _timeline2sync(session_path, sync_collection='raw_sync_data', chmap=None):
    """

    Parameters
    ----------
    session_path
    sync_collection
    chmap

    Returns
    -------

    """
    path = session_path / sync_collection
    if not chmap:  # attempt to extract from the meta file using expected channel names, or use expected channel numbers
        default = DEFAULT_MAPS['mesoscope']['timeline']
        chmap = timeline_meta2chmap(path, include_channels=default.keys()) or default
    sync = load_sync_timeline(path, chmap=chmap)
    return sync, chmap


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
            sync, chmap = _timeline2sync(self.session_path, sync_collection)
        return super()._extract(sync, chmap, sync_collection, **kwargs)

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

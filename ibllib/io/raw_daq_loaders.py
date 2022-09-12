"""Loader functions for various DAQ data formats"""
from pathlib import Path
import logging
from collections import OrderedDict

import nptdms
import numpy as np
import neurodsp.utils

logger = logging.getLogger(__name__)


def load_raw_daq_tdms(path) -> 'nptdms.tdms.TdmsFile':
    """
    Returns a dict of channel names and values from chmap

    Parameters
    ----------
    path
    chmap

    Returns
    -------

    """
    from nptdms import TdmsFile
    # If path is a directory, glob for a tdms file
    if (path := Path(path)).is_dir():  # cast to pathlib.Path
        file_path = next(path.glob('*.tdms'), None)
    else:
        file_path = path
    if not file_path or not file_path.exists():
        raise FileNotFoundError

    return TdmsFile.read(file_path)


def load_channels_tdms(path, chmap=None, return_fs=False):
    """

    Note: This currently cannot deal with arbitrary groups.

    Parameters
    ----------
    path
    chmap: dictionary mapping devices names to channel codes: example {"photometry": 'AI0', 'bpod': 'AI1'}
     if None, will read all of available channel from the DAQ

    Returns
    -------

    """
    data_file = load_raw_daq_tdms(path)
    data = {}
    if chmap:
        for name, ch in chmap.items():
            if ch.lower()[0] == 'a':
                data[name] = data_file['Analog'][ch.upper()].data
                fs = data_file['Analog'].properties['ScanRate']
            else:
                raise NotImplementedError(f'Extraction of channel "{ch}" not implemented')
    else:
        for group in (x.name for x in data_file.groups()):
            for ch in (x.name for x in data_file[group].channels()):
                data[ch] = data_file[group][ch.upper()].data
            fs = data_file[group].properties['ScanRate']  # from daqami it's unclear that fs could be set per channel
    if return_fs:
        return data, fs
    else:
        return data


def load_sync_tdms(path, sync_map, fs=None, threshold=2.5, floor_percentile=10):
    """

    Parameters
    ----------
    path : str, pathlib.Path
        The file or folder path of the raw TDMS data file.
    sync_map : dict
        A map of channel names and channel IDs.
    fs : float
        Sampling rate in Hz.
    threshold : float
        The threshold for applying to analogue channels
    floor_percentile : float
        10% removes the percentile value of the analog trace before thresholding. This is to avoid
        DC offset drift.

    Returns
    -------

    """
    data_file = load_raw_daq_tdms(path)
    sync = {}
    if any(x.lower()[0] != 'a' for x in sync_map.values()):
        raise NotImplementedError('Non-analogue or arbitrary group channel extraction not supported')

    raw_channels = [ch for ch in data_file['Analog'].channels() if ch.name.lower() in sync_map.values()]
    analogue = np.vstack([ch.data for ch in raw_channels])
    channel_ids = OrderedDict([(ch.name.lower(), ch.properties['ChannelKey']) for ch in raw_channels])
    offset = np.percentile(analogue, floor_percentile, axis=0)
    logger.info(f'estimated analogue channel DC Offset approx. {np.mean(offset):.2f}')
    analogue -= offset
    ttl = analogue > threshold
    ind, sign = neurodsp.utils.fronts(ttl.astype(int))
    try:  # attempt to get the times from the meta data
        times = np.vstack([ch.time_track() for ch in raw_channels])
        times = times[tuple(ind)]
    except KeyError:
        assert fs
        times = ind[1].astype(float) * 1/fs  # noqa

    # Sort by times
    ind_sorted = np.argsort(times)
    sync['times'] = times[ind_sorted]
    # Map index to channel key then reindex by sorted times
    sync['channels'] = np.fromiter(channel_ids.values(), dtype=int)[ind[0]][ind_sorted]
    sync['polarities'] = sign[ind_sorted]

    # Map sync name to channel key
    sync_map = {v.lower(): k for k, v in sync_map.items()}  # turn inside-out
    chmap = {sync_map[k]: v for k, v in channel_ids.items()}
    return sync, chmap

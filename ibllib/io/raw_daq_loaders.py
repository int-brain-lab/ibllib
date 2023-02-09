"""Loader functions for various DAQ data formats"""
from pathlib import Path
import logging
from collections import OrderedDict

import nptdms
import numpy as np
import neurodsp.utils
import one.alf.io as alfio

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

    def _load_digital_channels(data_file, group='Digital', ch='AuxPort'):
        # the digital channels are encoded on a single uint8 channel where each bit corresponds to an input channel
        ddata = data_file[group][ch].data.astype(np.uint8)
        nc = int(2 ** np.floor(np.log2(np.max(ddata))))
        ddata = np.unpackbits(ddata[:, np.newaxis], axis=1, count=nc, bitorder='little')
        data = {}
        for i in range(ddata.shape[1]):
            data[f'DI{i}'] = ddata[:, i]
        return data

    data_file = load_raw_daq_tdms(path)
    data = {}
    digital_channels = None
    if chmap:
        for name, ch in chmap.items():
            if ch.lower()[0] == 'a':
                data[name] = data_file['Analog'][ch.upper()].data
                fs = data_file['Analog'].properties['ScanRate']
            elif ch.lower()[0] == 'd':
                # do not attempt to load digital channels several times
                digital_channels = digital_channels or _load_digital_channels(data_file)
                data[name] = digital_channels[ch.upper()]
                fs = data_file['Digital'].properties['ScanRate']
            else:
                raise NotImplementedError(f'Extraction of channel "{ch}" not implemented')
    else:
        for group in (x.name for x in data_file.groups()):
            for ch in (x.name for x in data_file[group].channels()):
                if group == 'Digital' and ch == 'AuxPort':
                    data = {**data, **_load_digital_channels(data_file, group, ch)}
                else:
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


def load_sync_timeline(path, sync_map, threshold=2.5, floor_percentile=10):
    """
    Load sync channels from a timeline object.

    Note: Because the scan frequency is typically faster than the sample rate, the position and
    edge count channels may detect more than one front between samples.  Therefore for these, the
    raw data is more accurate than the extracted polarities.

    Parameters
    ----------
    path : str, pathlib.Path
        The file or folder path of the _timeline_DAQdata file.
    sync_map : dict
        A map of channel names and channel IDs.
    threshold : float
        The threshold for applying to analogue channels
    floor_percentile : float
        10% removes the percentile value of the analog trace before thresholding. This is to avoid
        DC offset drift.

    Returns
    -------
    one.alf.io.AlfBunch
        The sync bunch with keys ('times', 'polarities', 'channels')
    """
    timeline = alfio.load_object(path, 'DAQdata', namespace='timeline')
    assert timeline.keys() >= {'timestamps', 'raw', 'meta'}, 'Timeline object missing attributes'

    # Initialize sync object
    sync = alfio.AlfBunch((k, np.array([], dtype=d)) for k, d in
                          (('times', 'f'), ('channels', 'u1'), ('polarities', 'i1')))
    for label, i in sync_map.items():
        try:
            info = next(x for x in timeline['meta']['inputs'] if x['name'].lower() == label.lower())
        except StopIteration:
            logger.warning('sync channel "%s" not found', label)
            continue
        raw = timeline['raw'][:, info['arrayColumn'] - 1]  # -1 because MATLAB indexes from 1
        if info['measurement'] == 'Voltage':
            # Get TLLs by applying a threshold to the diff of voltage samples
            offset = np.percentile(raw, floor_percentile, axis=0)
            logger.debug(f'estimated analogue channel DC Offset approx. {np.mean(offset):.2f}')
            ind, val = neurodsp.utils.fronts(raw - offset, step=threshold)
            sync.polarities = np.concatenate((sync.polarities, np.sign(val).astype('i1')))
        elif info['measurement'] == 'EdgeCount':
            # Monotonically increasing values; extract indices where delta == 1
            ind, = np.where(np.diff(raw))
            ind += 1
            sync.polarities = np.concatenate((sync.polarities, np.ones_like(ind, dtype='i1')))
        elif info['measurement'] == 'Position':
            # Bidirectional; extract indices where delta != 0
            d = np.diff(raw)
            ind, = np.where(d.astype(int))
            sync.polarities = np.concatenate((sync.polarities, np.sign(d[ind]).astype('i1')))
            ind += 1
        else:
            raise NotImplementedError(f'{info["measurement"]} sync extraction')
        # Append timestamps of indices and channel index to sync arrays
        sync.times = np.concatenate((sync.times, timeline['timestamps'][ind]))
        sync.channels = np.concatenate((sync.channels, np.full(ind.shape, i, dtype='u1')))

    # Sort arrays by time
    assert sync.check_dimensions == 0
    t_ind = np.argsort(sync.times)
    for k in sync:
        sync[k] = sync[k][t_ind]

    return sync

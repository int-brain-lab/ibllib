from ibllib.io.raw_daq_loaders import load_sync_timeline

from ibllib.io.extractors.default_channel_maps import DEFAULT_MAPS


def _timeline2sync(session_path, chmap=None):
    chmap = chmap or DEFAULT_MAPS['mesoscope']['timeline']
    sync = load_sync_timeline(session_path / 'raw_mesoscope_data', sync_map=chmap)
    return sync, chmap

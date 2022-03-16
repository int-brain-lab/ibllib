"""Loader functions for various DAQ data formats"""
from pathlib import Path


def load_daq_tdms(path, chmap) -> dict:
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
    if (path := Path(path)).is_dir():
        file_path = next(path.glob('*.tdms'), None)
    else:
        file_path = path
    if not file_path or not file_path.exists():
        raise FileNotFoundError

    data_file = TdmsFile.read(file_path)
    sync = {}
    for name, ch in chmap.items():
        if ch.lower()[0] == 'a':
            sync[name] = data_file['Analog'][ch.upper()].data
        else:
            raise NotImplementedError(f'Extraction of channel "{ch}" not implemented')
    return sync

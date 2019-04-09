"""
Generic ALF I/O module.
Provides support for time-series reading and interpolation as per the specifications
For a full overview of the scope of the format, see:
https://ibllib.readthedocs.io/en/develop/04_reference.html#alf
"""
from pathlib import Path
import logging
import json

import numpy as np
import pandas as pd

from ibllib.io import jsonable

logger_ = logging.getLogger('ibllib')


def read_ts(filename):
    """
    Load time-series from ALF format
    t, d = alf.read_ts(filename)
    """
    if not isinstance(filename, Path):
        filename = Path(filename)

    # alf format is object.attribute.extension, for example '_ibl_wheel.position.npy'
    obj, attr, ext = filename.name.split('.')

    # looking for matching object with attribute timestamps: '_ibl_wheel.timestamps.npy'
    time_file = filename.parent / '.'.join([obj, 'timestamps', ext])

    if not time_file.exists():
        logger_.error(time_file.name + ' not found !, no time-scale for' + str(filename))
        raise FileNotFoundError(time_file.name + ' not found !, no time-scale for' + str(filename))

    return np.load(time_file), np.load(filename)


def load_file_content(fil):
    """
    Returns content of files. Designed for very generic file formats:
    so far supported contents are `json`, `npy`, `csv`, `tsv`, `ssv`, `jsonable`
    :param fil:
    :return:array/json/pandas dataframe depending on format
    """
    if not fil:
        return
    fil = Path(fil)
    if fil.stat().st_size == 0:
        return
    if fil.suffix == '.npy':
        return np.load(file=fil)
    if fil.suffix == '.json':
        try:
            with open(fil) as _fil:
                return json.loads(_fil.read())
        except Exception as e:
            logger_.error(e)
            return None
    if fil.suffix == '.jsonable':
        return jsonable.read(fil)
    if fil.suffix == '.tsv':
        return pd.read_csv(fil, delimiter='\t')
    if fil.suffix == '.csv':
        return pd.read_csv(fil)
    if fil.suffix == '.ssv':
        return pd.read_csv(fil, delimiter=' ')


def load_object(alfpath, object=None):
    """
    Reads all files (ie. attributes) sharing the same object.
    For example, if the file provided to the function is `spikes.times`, the function will
    load `spikes.time`, `spikes.clusters`, `spikes.depths`, `spike.amps` in a dictionary
    whose keys will be `time`, `clusters`, `depths`, `amps`
    :param alfpath: any alf file pertaining to the object OR directory containing files
    :param object: if a directory is provided, need to specify the name of object to load
    :return: a dictionary of all attributes pertaining to the object
    """
    alfpath = Path(alfpath)
    if alfpath.is_dir():
        if object is None:
            raise ValueError('If a path is provided, the object name should be provided too')
    else:
        object = alfpath.name.split('.')[0]
        alfpath = alfpath.parent
    # look for the files of the object
    files_alf = list(alfpath.glob(object + '.*'))
    attributes = [f.name.split('.')[1] for f in files_alf]
    OUT = {}
    # laod content for each file
    for fil, att in zip(files_alf, attributes):
        OUT[att] = load_file_content(fil)
    return OUT

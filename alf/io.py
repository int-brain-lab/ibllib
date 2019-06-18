"""
Generic ALF I/O module.
Provides support for time-series reading and interpolation as per the specifications
For a full overview of the scope of the format, see:
https://ibllib.readthedocs.io/en/develop/04_reference.html#alf
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ibllib.io import jsonable

logger_ = logging.getLogger('ibllib')


def check_dimensions(dico):
    """
    Test for consistency of dimensions as per ALF specs in a dictionary. Raises a Value Error.

    Alf broadcasting rules: only accepts consistent dimensions for a given axis
    a dimension is consistent with another if it's empty, 1, or equal to the other arrays
    dims [a, 1],  [1, b] and [a, b] are all consistent, [c, 1] is not

    :param dico: dictionary containing data
    :return: status 0 for consistent dimensions, 1 for inconsistent dimensions
    """
    shapes = [dico[lab].shape for lab in dico]
    lmax = max([len(s) for s in shapes])
    for l in range(lmax):
        sh = np.array([s[l] if (len(s) - 1 >= l) else 1 for s in shapes])
        if not np.unique(sh[sh != 1]).size <= 1:
            return int(1)
    return int(0)


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

    example: spikes = ibllib.io.alf.load_object('/path/to/my/alffolder/', 'spikes')
    """
    alfpath = Path(alfpath)
    if alfpath.is_dir():
        if object is None:
            raise ValueError('If a path is provided, the object name should be provided too')
    else:
        object = alfpath.name.split('.')[0]
        alfpath = alfpath.parent
    # look for the files of the object, raise infomative error if none found
    files_alf = list(alfpath.glob(object + '.*'))
    if not files_alf:
        raise FileNotFoundError('No object ' + str(object) + ' found in ' + str(alfpath))
    attributes = [f.name.split('.')[1] for f in files_alf]
    OUT = {}
    # load content for each file
    for fil, att in zip(files_alf, attributes):
        OUT[att] = load_file_content(fil)
    status = check_dimensions(OUT)
    if status != 0:
        logger_.warning('Inconsistent dimensions for object:' + object +
                        str([(k, v.shape) for k, v in OUT.items()]))
    return OUT


def save_object_npy(alfpath, dico, object):
    """
    Saves a dictionary in alf format using object as object name and dictionary keys as attibute
    names. Dimensions have to be consistent.

    :param alfpath: path of the folder to save data to
    :param dico: dictionary to save to npy
    :param object: name of the object to save
    :return: None

    example: ibllib.io.alf.save_object_npy('/path/to/my/alffolder/', spikes, 'spikes')
    """
    alfpath = Path(alfpath)
    status = check_dimensions(dico)
    if status != 0:
        raise ValueError('Dimensions are not consistent to save all arrays in ALF format: ' +
                         str([(k, v.shape) for k, v in dico.items()]))

    for k, v in dico.items():
        np.save(alfpath / (object + '.' + k + '.npy'), v)

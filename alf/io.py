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

from brainbox.core import Bunch
from ibllib.io import jsonable

logger_ = logging.getLogger('ibllib')


def _find_metadata(file_alf):
    """
    Loof for an existing meta-data file for an alf_file
    :param file_alf: PurePath of existing alf file
    :return: PurePath of meta-data if exists
    """
    ns, obj = file_alf.name.split('.')[:2]
    meta_data_file = list(file_alf.parent.glob(f'{ns}.{obj}*.metadata*.json'))
    if meta_data_file:
        return meta_data_file[0]


def check_dimensions(dico):
    """
    Test for consistency of dimensions as per ALF specs in a dictionary. Raises a Value Error.

    Alf broadcasting rules: only accepts consistent dimensions for a given axis
    a dimension is consistent with another if it's empty, 1, or equal to the other arrays
    dims [a, 1],  [1, b] and [a, b] are all consistent, [c, 1] is not

    :param dico: dictionary containing data
    :return: status 0 for consistent dimensions, 1 for inconsistent dimensions
    """
    shapes = [dico[lab].shape for lab in dico if isinstance(dico[lab], np.ndarray)]
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

    :param fil: file to read
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


def _ls(alfpath, object, glob='.*'):
    """
    Given a path, an object and a filter, returns all files and associated attributes
    :param alfpath: containing folder
    :param object: ALF object string
    :param glob: File filter (optional)
    :return: lists of pathlib.Path for each file and list of corresponding attributes
    """
    alfpath = Path(alfpath)
    if alfpath.is_dir():
        if object is None:
            raise ValueError('If a path is provided, the object name should be provided too')
    else:
        object = alfpath.name.split('.')[0]
        alfpath = alfpath.parent
    # look for files corresponding to the object, raise error if none found
    files_alf = list(alfpath.glob(object + glob))
    if not files_alf:
        raise FileNotFoundError('No object ' + str(object) + ' found in ' + str(alfpath))
    # in this case get the attributes and parts for each
    attributes = ['.'.join(f.name.split('.')[1:-1]) for f in files_alf]
    return files_alf, attributes


def exists(alfpath, object, attributes=None, glob='.*'):
    """
    Test if ALF object and optionally specific attributes exist in the given path
    :param alfpath: str or pathlib.Path of the folder to look into
    :param object: str ALF object name
    :param attributes: list or list of strings for wanted attributes
    :param glob: (".*") glob pattern to look for files or list of parts as per ALF specifications
    :return: Bool. For multiple attributes, returns True only if all attributes are found
    """
    # prepare the glob input argument if it's a list
    if isinstance(glob, list):
        glob = '*.' + '.'.join(glob) + '*'
    # if the object is not found, return False
    try:
        _, attributes_found = _ls(alfpath, object, glob=glob)
    except FileNotFoundError:
        return False
    # if object found and no attribute provided, True
    if not attributes:
        return True
    # if attributes provided, test if all are found
    if isinstance(attributes, str):
        attributes = [attributes]
    return set(attributes).issubset(set(attributes_found))


def load_object(alfpath, object=None, glob='.*', short_keys=False):
    """
    Reads all files (ie. attributes) sharing the same object.
    For example, if the file provided to the function is `spikes.times`, the function will
    load `spikes.time`, `spikes.clusters`, `spikes.depths`, `spike.amps` in a dictionary
    whose keys will be `time`, `clusters`, `depths`, `amps`
    Full Reference here: https://github.com/cortex-lab/ALF
    Simplified example: _namespace_object.attribute.part1.part2.extension

    :param alfpath: any alf file pertaining to the object OR directory containing files
    :param object: if a directory is provided, need to specify the name of object to load
    :param glob: a file filter string like one used in glob: "*.amps.*" for example
    :param short_keys: by default, the output dictionary keys will be compounds of attributes and
     any eventual parts separated by a dot. Use True to shorten the keys to the bare attribute.
    :return: a dictionary of all attributes pertaining to the object

    example: spikes = ibllib.io.alf.load_object('/path/to/my/alffolder/', 'spikes')
    """
    # prepare the glob input argument if it's a list
    if isinstance(glob, list):
        glob = '*.' + '.'.join(glob) + '*'
    files_alf, attributes = _ls(alfpath, object, glob=glob)
    OUT = Bunch({})
    # load content for each file
    for fil, att in zip(files_alf, attributes):
        # if there is a corresponding metadata file, read it:
        meta_data_file = _find_metadata(fil)
        # if this is the actual meta-data file, skip and it will be read later
        if meta_data_file == fil:
            continue
        OUT[att] = load_file_content(fil)
        if meta_data_file:
            meta = load_file_content(meta_data_file)
            # the columns keyword splits array along the last dimension
            if 'columns' in meta.keys():
                OUT.update({v: OUT[att][::, k] for k, v in enumerate(meta['columns'])})
                OUT.pop(att)
                meta.pop('columns')
            # if there is other stuff in the dictionary, save it, otherwise disregard
            if meta:
                OUT[att + 'metadata'] = meta
    status = check_dimensions(OUT)
    if status != 0:
        logger_.warning('Inconsistent dimensions for object:' + object + '\n' +
                        '\n'.join([f'{v.shape},    {k}' for k, v in OUT.items()]))
    if short_keys:
        for k in OUT:
            if k != k.split('.')[0]:
                OUT[k.split('.')[0]] = OUT.pop(k)
    return OUT


def save_object_npy(alfpath, dico, object, parts=''):
    """
    Saves a dictionary in alf format using object as object name and dictionary keys as attribute
    names. Dimensions have to be consistent.
    Reference here: https://github.com/cortex-lab/ALF
    Simplified example: _namespace_object.attribute.part1.part2.extension

    :param alfpath: path of the folder to save data to
    :param dico: dictionary to save to npy
    :param object: name of the object to save
    :param parts: extra parts to the ALF name
    :return: List of written files

    example: ibllib.io.alf.save_object_npy('/path/to/my/alffolder/', spikes, 'spikes')
    """
    alfpath = Path(alfpath)
    status = check_dimensions(dico)
    if isinstance(parts, list):
        parts = '.' + '.'.join(parts)
    elif parts:
        parts = '.' + parts
    if status != 0:
        raise ValueError('Dimensions are not consistent to save all arrays in ALF format: ' +
                         str([(k, v.shape) for k, v in dico.items()]))
    out_files = []
    for k, v in dico.items():
        out_file = alfpath / (object + '.' + k + parts + '.npy')
        np.save(out_file, v)
        out_files.append(out_file)
    return out_files


def save_metadata(file_alf, dico):
    """
    Writes a meta data file matching a current alf file object.
    For example given an alf file
    `clusters.ccf_location.ssv` this will write a dictionary in json format in
    `clusters.ccf_location.metadata.json`
    Reserved keywords:
     - **columns**: column names for binary tables.
     - **row**: row names for binary tables.
     - **unit**

    :param file_alf: full path to the alf object
    :param dico: dictionary containing meta-data.
    :return: None
    """
    file_meta_data = file_alf.parent / (file_alf.stem + '.metadata.json')
    with open(file_meta_data, 'w+') as fid:
        fid.write(json.dumps(dico, indent=1))

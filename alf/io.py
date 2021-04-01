"""
Generic ALF I/O module.
Provides support for time-series reading and interpolation as per the specifications
For a full overview of the scope of the format, see:
https://ibllib.readthedocs.io/en/develop/04_reference.html#alf
"""

import json
import copy
import logging
import re
from uuid import UUID
from datetime import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from brainbox.core import Bunch
from brainbox.io import parquet
from ibllib.io import jsonable
from ibllib.exceptions import ALFObjectNotFound
from . import files

_logger = logging.getLogger('ibllib')


class AlfBunch(Bunch):

    @property
    def check_dimensions(self):
        return check_dimensions(self)

    def append(self, b, inplace=False):
        """
        Appends one bunch to another, key by key
        :param bunch:
        :return: Bunch
        """
        # default is to return a copy
        if inplace:
            a = self
        else:
            a = AlfBunch(copy.deepcopy(self))
        # handles empty bunches for convenience if looping
        if b == {}:
            return a
        if a == {}:
            return b
        # right now supports only strictly matching keys. Will implement other cases as needed
        if set(a.keys()) != set(b.keys()):
            raise NotImplementedError("Append bunches only works with strictly matching keys"
                                      "For more complex merges, convert to pandas dataframe.")
        # do the merge; only concatenate lists and np arrays right now
        for k in a:
            if isinstance(a[k], np.ndarray):
                a[k] = np.concatenate((a[k], b[k]), axis=0)
            elif isinstance(a[k], list):
                a[k].extend(b[k])
            else:
                _logger.warning(f"bunch key '{k}' is a {a[k].__class__}. I don't know how to"
                                f" handle that. Use pandas for advanced features")
        check_dimensions(a)
        return a

    def to_df(self):
        return dataframe(self)


def dataframe(adict):
    """
    Converts an Bunch conforming to size conventions into a pandas Dataframe
    For 2-D arrays, stops at 10 columns per attribute
    :return: pandas Dataframe
    """
    if check_dimensions(adict) != 0:
        raise ValueError("Can only convert to Dataframe objects with consistent size")
    # easy case where there are only vectors
    if all([len(adict[k].shape) == 1 for k in adict]):
        return pd.DataFrame(adict)
    # pandas has trouble with 2d data, chop it off with a limit of 10 columns per dataset
    df = pd.DataFrame()
    for k in adict.keys():
        if adict[k].ndim == 1:
            df[k] = adict[k]
        elif adict[k].ndim == 2 and adict[k].shape[1] == 1:
            df[k] = adict[k][:, 0]
        elif adict[k].ndim == 2:
            for i in np.arange(adict[k].shape[1]):
                df[f"{k}_{i}"] = adict[k][:, i]
                if i == 9:
                    break
        else:
            _logger.warning(f"{k} attribute is 3D or more and won't convert to dataframe")
            continue
    return df


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
    excluded_attributes = ['timestamps']
    shapes = [dico[lab].shape for lab in dico if isinstance(dico[lab], np.ndarray) and
              lab.split('.')[0] not in excluded_attributes]
    # the dictionary may contain only excluded attributes, in this case return success
    if not shapes:
        return int(0)
    first_shapes = [sh[0] for sh in shapes]
    if set(first_shapes).issubset(set([max(first_shapes), 1])):
        return int(0)
    else:
        return int(1)


def read_ts(filename):
    """
    Load time-series from ALF format
    t, d = alf.read_ts(filename)
    """
    if not isinstance(filename, Path):
        filename = Path(filename)

    # alf format is object.attribute.extension, for example '_ibl_wheel.position.npy'
    _, obj, attr, *_, ext = files.alf_parts(filename.parts[-1])

    # looking for matching object with attribute timestamps: '_ibl_wheel.timestamps.npy'
    (time_file,), _ = files.filter_by(filename.parent, object=obj,
                                      attribute='timestamps', extension=ext)

    if not time_file:
        name = files.to_alf(obj, attr, ext)
        _logger.error(name + ' not found! no time-scale for' + str(filename))
        raise FileNotFoundError(name + ' not found! no time-scale for' + str(filename))

    return np.load(filename.parent / time_file), np.load(filename)


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
    if fil.suffix == '.csv':
        return pd.read_csv(fil)
    if fil.suffix == '.json':
        try:
            with open(fil) as _fil:
                return json.loads(_fil.read())
        except Exception as e:
            _logger.error(e)
            return None
    if fil.suffix == '.jsonable':
        return jsonable.read(fil)
    if fil.suffix == '.npy':
        return np.load(file=fil, allow_pickle=True)
    if fil.suffix == '.pqt':
        return parquet.load(fil)
    if fil.suffix == '.ssv':
        return pd.read_csv(fil, delimiter=' ')
    if fil.suffix == '.tsv':
        return pd.read_csv(fil, delimiter='\t')
    return Path(fil)


def _ls(alfpath, object=None, **kwargs):
    """
    Given a path, an object and a filter, returns all files and associated attributes
    :param alfpath: containing folder
    :param object: ALF object string; wildcards permitted
    :return: lists of pathlib.Path for each file and list of corresponding attributes
    """
    alfpath = Path(alfpath)
    if not alfpath.exists():
        files_alf = None
    elif alfpath.is_dir():
        if object is None:
            # List all ALF files
            files_alf, attributes = files.filter_by(alfpath)
        else:
            files_alf, attributes = files.filter_by(alfpath, object=object, **kwargs)
    else:
        object = files.alf_parts(alfpath.name)[1]
        alfpath = alfpath.parent
        files_alf, attributes = files.filter_by(alfpath, object=object, **kwargs)

    # raise error if no files found
    if not files_alf:
        err_str = 'object "%s" ' % object if object else 'ALF files'
        raise ALFObjectNotFound(f'No {err_str} found in {alfpath}')

    return [alfpath.joinpath(f) for f in files_alf], attributes


def exists(alfpath, object, attributes=None, **kwargs):
    """
    Test if ALF object and optionally specific attributes exist in the given path
    :param alfpath: str or pathlib.Path of the folder to look into
    :param object: str ALF object name
    :param attributes: list or list of strings for wanted attributes
    :return: bool. For multiple attributes, returns True only if all attributes are found
    """

    # if the object is not found, return False
    try:
        _, attributes_found = _ls(alfpath, object, **kwargs)
    except (FileNotFoundError, ALFObjectNotFound):
        return False

    # if object found and no attribute provided, True
    if not attributes:
        return True

    # if attributes provided, test if all are found
    if isinstance(attributes, str):
        attributes = [attributes]
    attributes_found = set(part[2] for part in attributes_found)
    return set(attributes).issubset(attributes_found)


def load_object(alfpath, object=None, short_keys=False, **kwargs):
    """
    Reads all files (ie. attributes) sharing the same object.
    For example, if the file provided to the function is `spikes.times`, the function will
    load `spikes.time`, `spikes.clusters`, `spikes.depths`, `spike.amps` in a dictionary
    whose keys will be `time`, `clusters`, `depths`, `amps`
    Full Reference here: https://docs.internationalbrainlab.org/en/latest/04_reference.html#alf
    Simplified example: _namespace_object.attribute_timescale.part1.part2.extension

    :param alfpath: any alf file pertaining to the object OR directory containing files
    :param object: if a directory is provided and object is None, all valid ALF files returned
    :param short_keys: by default, the output dictionary keys will be compounds of attributes,
     timescale and any eventual parts separated by a dot. Use True to shorten the keys to the
     attribute and timescale.
    :return: a dictionary of all attributes pertaining to the object

    Examples:
        # Load `spikes` object
        spikes = ibllib.io.alf.load_object('/path/to/my/alffolder/', 'spikes')

        # Load `trials` object under the `ibl` namespace
        trials = ibllib.io.alf.load_object(session_path, 'trials', namespace='ibl')

    """
    if Path(alfpath).is_dir() and object is None:
        raise ValueError('If a directory is provided, the object name should be provided too')
    files_alf, parts = _ls(alfpath, object, **kwargs)
    # Take attribute and timescale from parts list
    attributes = [p[2] if not p[3] else '_'.join(p[2:4]) for p in parts]
    if not short_keys:  # Include extra parts in the keys
        attributes = [attr + ('.' + p[4] if p[4] else '') for attr, p in zip(attributes, parts)]
    assert len(set(attributes)) == len(attributes), (
        f'multiple object {object} with the same attribute in {alfpath}, restrict parts/namespace')
    out = AlfBunch({})
    # load content for each file
    for fil, att in zip(files_alf, attributes):
        # if there is a corresponding metadata file, read it:
        meta_data_file = _find_metadata(fil)
        # if this is the actual meta-data file, skip and it will be read later
        if meta_data_file == fil:
            continue
        out[att] = load_file_content(fil)
        if meta_data_file:
            meta = load_file_content(meta_data_file)
            # the columns keyword splits array along the last dimension
            if 'columns' in meta.keys():
                out.update({v: out[att][::, k] for k, v in enumerate(meta['columns'])})
                out.pop(att)
                meta.pop('columns')
            # if there is other stuff in the dictionary, save it, otherwise disregard
            if meta:
                out[att + 'metadata'] = meta
    status = check_dimensions(out)
    if status != 0:
        _logger.warning('Inconsistent dimensions for object:' + object + '\n' +
                        '\n'.join([f'{v.shape},    {k}' for k, v in out.items()]))
    return out


def save_object_npy(alfpath, dico, object, parts=None, namespace=None, timescale=None):
    """
    Saves a dictionary in alf format using object as object name and dictionary keys as attribute
    names. Dimensions have to be consistent.
    Reference here: https://github.com/cortex-lab/ALF
    Simplified example: _namespace_object.attribute.part1.part2.extension

    :param alfpath: path of the folder to save data to
    :param dico: dictionary to save to npy; keys correspond to ALF attributes
    :param object: name of the object to save
    :param parts: extra parts to the ALF name
    :param namespace: the optional namespace of the object
    :param timescale: the optional timescale of the object
    :return: List of written files

    example: ibllib.io.alf.save_object_npy('/path/to/my/alffolder/', spikes, 'spikes')
    """
    alfpath = Path(alfpath)
    status = check_dimensions(dico)
    if status != 0:
        raise ValueError('Dimensions are not consistent to save all arrays in ALF format: ' +
                         str([(k, v.shape) for k, v in dico.items()]))
    out_files = []
    for k, v in dico.items():
        out_file = alfpath / files.to_alf(object, k, 'npy',
                                          extra=parts, namespace=namespace, timescale=timescale)
        np.save(out_file, v)
        out_files.append(out_file)
    return out_files


def save_metadata(file_alf, dico):
    """
    Writes a meta data file matching a current alf file object.
    For example given an alf file
    `clusters.ccfLocation.ssv` this will write a dictionary in json format in
    `clusters.ccfLocation.metadata.json`
    Reserved keywords:
     - **columns**: column names for binary tables.
     - **row**: row names for binary tables.
     - **unit**

    :param file_alf: full path to the alf object
    :param dico: dictionary containing meta-data.
    :return: None
    """
    assert files.is_valid(file_alf.parts[-1]), 'ALF filename not valid'
    file_meta_data = file_alf.parent / (file_alf.stem + '.metadata.json')
    with open(file_meta_data, 'w+') as fid:
        fid.write(json.dumps(dico, indent=1))


def remove_uuid_file(file_path, dry=False):
    """
     Renames a file without the UUID and returns the new pathlib.Path object
    """
    file_path = Path(file_path)
    name_parts = file_path.name.split('.')
    if not is_uuid_string(name_parts[-2]):
        return file_path
    name_parts.pop(-2)
    new_path = file_path.parent.joinpath('.'.join(name_parts))
    if not dry and file_path.exists():
        file_path.replace(new_path)
    return new_path


def remove_uuid_recursive(folder, dry=False):
    """
    Within a folder, recursive renaming of all files to remove UUID
    """
    for fn in Path(folder).rglob('*.*'):
        print(remove_uuid_file(fn, dry=False))


def add_uuid_string(file_path, uuid):
    if isinstance(uuid, str) and not is_uuid_string(uuid):
        raise ValueError('Should provide a valid UUID v4')
    uuid = str(uuid)
    # NB: Only instantiate as Path if not already a Path, otherwise we risk changing the class
    if isinstance(file_path, str):
        file_path = Path(file_path)
    name_parts = file_path.stem.split('.')
    if uuid == name_parts[-1]:
        _logger.warning(f'UUID already found in file name: {file_path.name}: IGNORE')
        return file_path
    return file_path.parent.joinpath(f"{'.'.join(name_parts)}.{uuid}{file_path.suffix}")


def is_uuid_string(string: str) -> bool:
    """
    Bool test for randomly generated hexadecimal uuid validity
    NB: uuid must be hyphen separated
    """
    if string is None:
        return False
    if len(string) != 36:
        return False
    UUID_PATTERN = re.compile(r'^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$', re.IGNORECASE)
    if UUID_PATTERN.match(string):
        return True
    else:
        return False


def is_uuid(uuid: Union[str, int, bytes, UUID]) -> bool:
    """Bool test for randomly generated hexadecimal uuid validity
    Unlike `is_uuid_string`, this function accepts UUID objects
    """
    if not isinstance(uuid, (UUID, str, bytes, int)):
        return False
    elif not isinstance(uuid, UUID):
        try:
            uuid = UUID(uuid) if isinstance(uuid, str) else UUID(**{type(uuid).__name__: uuid})
        except ValueError:
            return False
    return isinstance(uuid, UUID) and uuid.version == 4


def _isdatetime(s: str) -> bool:
    try:
        datetime.strptime(s, '%Y-%m-%d')
        return True
    except Exception:
        return False


def get_session_path(path: Union[str, Path]) -> Path:
    """Returns the session path from any filepath if the date/number
    pattern is found"""
    if path is None:
        _logger.warning('Input path is None, exiting...')
        return
    path = Path(path)
    sess = None
    for i, p in enumerate(path.parts):
        if p.isdigit() and _isdatetime(path.parts[i - 1]):
            sess = Path().joinpath(*path.parts[:i + 1])

    return sess


def is_session_path(path_object):
    """
    Checks if the syntax corresponds to a session path. Note that there is no physical check
     about existence nor contents
    :param path_object:
    :return:
    """
    return Path(path_object) == get_session_path(Path(path_object))


def _regexp_session_path(path_object, separator):
    """
    Subfunction to be able to test cross-platform
    """
    return re.search(r'/\d\d\d\d-\d\d-\d\d/\d\d\d',
                     str(path_object).replace(separator, '/'), flags=0)


def is_details_dict(dict_obj):
    if dict_obj is None:
        return False
    keys = [
        'subject',
        'start_time',
        'number',
        'lab',
        'project',
        'url',
        'task_protocol',
        'local_path'
    ]
    return set(dict_obj.keys()) == set(keys)

import collections
from pathlib import Path, PurePath
import sys
import os
import json
import warnings

warnings.warn('Support for ibllib.io.params will soon be removed, use iblutil.io.params instead',
              category=DeprecationWarning)


def as_dict(par):
    if not par or isinstance(par, dict):
        return par
    else:
        return dict(par._asdict())


def from_dict(par_dict):
    if not par_dict:
        return None
    par = collections.namedtuple('Params', par_dict.keys())

    class IBLParams(par):

        def set(self, field, value):
            d = as_dict(self)
            d[field] = value
            return from_dict(d)

        def as_dict(self):
            return as_dict(self)

    return IBLParams(**par_dict)


def getfile(str_params):
    """
    Returns full path of the param file per system convention:
     linux/mac: ~/.str_params, Windows: APPDATA folder

    :param str_params: string that identifies parm file
    :return: string of full path
    """
    # strips already existing dot if any
    parts = ['.' + p if not p.startswith('.') else p for p in Path(str_params).parts]
    if sys.platform == 'win32' or sys.platform == 'cygwin':
        pfile = str(PurePath(os.environ['APPDATA'], *parts))
    else:
        pfile = str(Path.home().joinpath(*parts))
    return pfile


def read(str_params, default=None):
    """
    Reads in and parse Json parameter file into dictionary.  If the parameter file doesn't
    exist and no defaults are provided, a FileNotFound error is raised, otherwise any extra
    default parameters will be written into the file.

    Examples:
        # Load parameters, raise error if file not found
        par = read('globus/admin')

        # Load with defaults
        par = read('globus/admin', {'local_endpoint': None, 'remote_endpoint': None})

        # Return empty dict if file not found (i.e. touch new param file)
        par = read('new_pars', {})

    :param str_params: path to text json file
    :param default: default values for missing parameters
    :return: named tuple containing parameters
    """
    pfile = getfile(str_params)
    par_dict = as_dict(default) or {}
    if Path(pfile).exists():
        with open(pfile) as fil:
            file_pars = json.loads(fil.read())
        par_dict.update(file_pars)
    elif default is None:  # No defaults provided
        raise FileNotFoundError(f'Parameter file {pfile} not found')

    if not Path(pfile).exists() or par_dict.keys() > file_pars.keys():
        # write the new parameter file with the extra param
        write(str_params, par_dict)
    return from_dict(par_dict)


def write(str_params, par):
    """
    Write a parameter file in Json format

    :param str_params: path to text json file
    :param par: dictionary containing parameters values
    :return: None
    """
    pfile = Path(getfile(str_params))
    if not pfile.parent.exists():
        pfile.parent.mkdir()
    dpar = as_dict(par)
    for k in dpar:
        if isinstance(dpar[k], Path):
            dpar[k] = str(dpar[k])
    with open(pfile, 'w') as fil:
        json.dump(as_dict(par), fil, sort_keys=False, indent=4)

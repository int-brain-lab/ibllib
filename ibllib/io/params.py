import collections
import pathlib
import sys
import os
import json


def as_dict(par):
    if not par:
        return None
    if isinstance(par, dict):
        return par
    else:
        return dict(par._asdict())


def from_dict(par_dict):
    if not par_dict:
        return None
    # par = collections.namedtuple('Params', par_dict.keys())(**par_dict)
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
    if sys.platform == 'win32' or sys.platform == 'cygwin':
        pfile = str(pathlib.PurePath(os.environ['APPDATA'], '.' + str_params))
    else:
        pfile = str(pathlib.PurePath(pathlib.Path.home(), '.' + str_params))
    return pfile


def read(str_params, default=None):
    """
    Reads in and parse Json parameter file into dictionary

    :param str_params: path to text json file
    :param default: default values for missing parameters
    :return: named tuple containing parameters
    """
    pfile = getfile(str_params)
    if pathlib.Path(pfile).exists():
        with open(pfile) as fil:
            par_dict = json.loads(fil.read())
    else:
        par_dict = as_dict(default)
    # without default parameters
    default = as_dict(default)
    # TODO : behaviour for non existing file
    # tat = params.read('rijafa', default={'toto': 'titi', 'tata': 1})
    if not default or default.keys() == par_dict.keys():
        return from_dict(par_dict)
    # if default parameters bring in a new parameter
    new_keys = set(default.keys()).difference(set(par_dict.keys()))
    for nk in new_keys:
        par_dict[nk] = default[nk]
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
    pfile = getfile(str_params)
    dpar = as_dict(par)
    for k in dpar:
        if isinstance(dpar[k], pathlib.Path):
            dpar[k] = str(dpar[k])
    with open(pfile, 'w') as fil:
        json.dump(as_dict(par), fil, sort_keys=False, indent=4)

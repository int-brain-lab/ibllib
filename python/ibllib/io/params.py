import collections
import pathlib
import sys
import os
import json


def as_dict(par):
    return dict(par._asdict())


def from_dict(par_dict):
    par = collections.namedtuple('Params', par_dict.keys())(**par_dict)
    return par


def getfile(str_params):
    if sys.platform == 'win32' or sys.platform == 'cygwin':
        pfile = str(pathlib.PurePath(os.environ['APPDATA'], '.' + str_params))
    else:
        pfile = str(pathlib.PurePath(pathlib.Path.home(), '.' + str_params))
    return pfile


def read(str_params):
    pfile = getfile(str_params)
    if os.path.isfile(pfile):
        with open(pfile) as fil:
            par_dict = json.loads(fil.read())
        return from_dict(par_dict)
    else:
        return None


def write(str_params, par):
    pfile = getfile(str_params)
    with open(pfile, 'w') as fil:
        json.dump(as_dict(par), fil, sort_keys=False, indent=4)

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Monday, January 21st 2019, 6:28:49 pm
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Union
# TODO: tests for this module!

log = logging.getLogger('ibllib')


def subjects_data_folder(folder: Path, rglob: bool = False) -> Path:
    """Given a root_data_folder will try to find a 'Subjects' data folder.
    If Subjects folder is passed will return it directly."""
    if not isinstance(folder, Path):
        folder = Path(folder)
    if rglob:
        func = folder.rglob
    else:
        func = folder.glob

    # Try to find Subjects folder one level
    if folder.name.lower() != 'subjects':
        # Try to find Subjects folder if folder.glob
        spath = [x for x in func('*') if x.name.lower() == 'subjects']
        if not spath:
            log.error('No "Subjects" folder in children folders')
            raise(ValueError)
        elif len(spath) > 1:
            log.error(f'Multiple "Subjects" folder in children folders: {spath}')
            raise(ValueError)
        else:
            folder = folder / spath[0]

    return folder


def remove_empty_folders(folder: Union[str, Path]) -> None:
    """Will iteratively remove any children empty folders"""
    all_folders = [x for x in Path(folder).rglob('*') if x.is_dir()]
    for f in all_folders:
        try:
            f.rmdir()
        except Exception:
            continue


def find_sessions(folder: Union[str, Path]) -> List[str]:
    """Returns all sessions found in all subfolders of a main data folder"""
    # Ensure folder is a Path object
    folder = Path(folder)
    out = [str(x.parent.parent) for x in folder.rglob('_iblrig_taskSettings.raw*')]
    return out


def find_subject_names(folder: Union[str, Path]) -> List[str]:
    """Returns all subject names found from a main data folder"""
    # Ensure folder is a Path object
    if not isinstance(folder, Path):
        folder = Path(folder)
    out = [x.parent.parent.parent.parent.name
           for x in folder.rglob('_iblrig_taskSettings.raw*')]
    return out


def _isdatetime(s: str) -> bool:
    try:
        datetime.strptime(s, '%Y-%m-%d')
        return True
    except Exception:
        return False


def session_path(path: Union[str, Path]) -> str:
    """Returns the session path from any filepath if the date/number
    pattern is found"""
    path = Path(path)
    sess = None
    for i, p in enumerate(path.parts):
        if p.isdigit() and _isdatetime(path.parts[i - 1]):
            sess = str(Path().joinpath(*path.parts[:i + 1]))

    return sess


def session_name(path: Union[str, Path]) -> str:
    """Returns the session name (subject/date/number) string for any filepath
    useing session_path"""
    path = Path(path)
    return '/'.join(Path(session_path(path)).parts[-3:])


def next_num_folder(session_date_folder: str) -> str:
    """Return the next number for a session given a session_date_folder"""
    session_date_folder = Path(session_date_folder)
    if not session_date_folder.exists():
        return '001'
    session_nums = [
        int(x.name) for x in session_date_folder.iterdir()
        if x.is_dir() and not x.name.startswith('.') and x.name.isdigit()
    ]
    if not session_nums:
        out = '00' + str(1)
    elif max(session_nums) < 9:
        out = '00' + str(int(max(session_nums)) + 1)
    elif 99 > max(session_nums) >= 9:
        out = '0' + str(int(max(session_nums)) + 1)
    elif max(session_nums) > 99:
        out = str(int(max(session_nums)) + 1)
    return out


def find_subject_folders(folder: Union[str, Path]) -> List[Path]:
    """Returns all subject folders found from a main data folder"""
    # Ensure folder is a Path object
    folder = Path(folder)
    out = [str(x.parent.parent.parent.parent)
           for x in folder.rglob('_iblrig_taskSettings.raw*')]
    return out


def find_mouse_sessions(folder, mouse):
    return [x for x in find_sessions(folder) if mouse in x]


def search(**kwargs):
    from oneibl.one import SEARCH_TERMS
    print(set(SEARCH_TERMS.values()))


if __name__ == "__main__":
    print(0)

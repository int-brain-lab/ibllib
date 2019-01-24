# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Monday, January 21st 2019, 6:28:49 pm
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 21-01-2019 06:28:51.5151
from pathlib import Path
from typing import List
import logging

log = logging.getLogger('ibllib')


def subjects_data_folder(folder: Path) -> Path:
    """Given a root_data_folder will try to find a 'Subjects' data folder.
    If Subjects folder is passed will return it directly."""
    # Try to find Subjects folder one level
    if folder.name.lower() != 'subjects':
        # Try to find Subjects folder if folder.glob
        spath = [x for x in folder.glob('*') if x.name.lower() == 'subjects']
        if not spath:
            raise(ValueError)
        elif len(spath) > 1:
            raise(ValueError)
        else:
            folder = folder / spath[0]

    return folder


def remove_empty_folders(folder: str or Path) -> None:
    """Will iteratively remove any children empty folders"""
    all_folders = [x for x in Path(folder).rglob('*') if x.is_dir()]
    for f in all_folders:
        try:
            f.rmdir()
        except Exception:
            continue


def find_sessions(folder: str or Path) -> List[Path]:
    # Ensure folder is a Path object
    if not isinstance(folder, Path):
        folder = Path(folder)

    folder = subjects_data_folder(folder)
    # Glob all mouse fodlers
    mouse_folders = [x for x in folder.glob('*') if x.is_dir()]
    if not mouse_folders:
        log.error(f"No subjects found in '{Path(*folder.parts[-2:])}'")
        raise(ValueError)
    flen = len(list(mouse_folders))
    log.info(f"Found '{flen}' subjects: {[x.name for x in mouse_folders]}")
    # Glob all dates
    dates = [x for mouse in mouse_folders for x in mouse.glob(
        '*') if x.is_dir()]
    log.info(f"Found '{len(dates)}' dates: {[x.name for x in dates]}")
    # Glob all sessions
    sessions = [y for x in dates for y in x.glob('*') if y.is_dir()]
    # Ensure sessions have files
    sessions = list(
        {p.parent for f in sessions for p in f.glob('*') if p.is_file()})
    snames = [str(Path(*x.parts[-3:])) for x in sessions]
    log.info(
        f"Found '{len(sessions)}' sessions: {snames}")
    sessions = [str(x) for x in sessions]

    return sessions

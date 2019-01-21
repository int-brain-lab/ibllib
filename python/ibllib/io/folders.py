# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Monday, January 21st 2019, 6:28:49 pm
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 21-01-2019 06:28:51.5151
from pathlib import Path


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
        except:
            continue

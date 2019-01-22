# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, January 22nd 2019, 12:32:14 pm
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 22-01-2019 12:32:16.1616
from pathlib import Path


def read_flag_file(fil):
    """
    Flag files are *.flag files within a session folder used to schedule some jobs
    If they are empty, should return True
    """
    # the flag file may contains specific file names for a targeted extraction
    with open(fil) as fid:
        save = list(filter(None, fid.read().splitlines()))
    # if empty, extract everything by default
    if len(save) == 0:
        save = True
    return save


def write_flag_file(fname, file_list: list = None):
    """
    Flag files are *.flag files within a session folder used to schedule some jobs
    Each line references to a file to extract or register
    """
    with open(fname, 'w+') as fid:
        if isinstance(file_list, str):
            file_list = [file_list]
        if file_list:
            fid.write('\n'.join(file_list))


def create_register_flags(root_data_folder, force=False, file_list=None):
    ses_path = Path(root_data_folder).glob('**/raw_behavior_data')
    for p in ses_path:
        flag_file = Path(p).parent.joinpath('register_me.flag')
        if p.parent.joinpath('flatiron.flag').is_file() and not force:
            continue
        if p.parent.joinpath('extract_me.error').is_file() and not force:
            continue
        if p.parent.joinpath('register_me.error').is_file() and not force:
            continue
        if force and p.parent.joinpath('flatiron.flag').is_file():
            p.parent.joinpath('flatiron.flag').unlink()
        write_flag_file(flag_file, file_list)
        print(flag_file)


def create_extract_flags(root_data_folder, force=False, file_list=None):
    # first part is to create extraction flags
    ses_path = Path(root_data_folder).glob('**/raw_behavior_data')
    for p in ses_path:
        flag_file = Path(p).parent.joinpath('extract_me.flag')
        if p.parent.joinpath('flatiron.flag').is_file() and not force:
            continue
        if p.parent.joinpath('extract_me.error').is_file() and not force:
            continue
        if p.parent.joinpath('register_me.error').is_file() and not force:
            continue
        if force and p.parent.joinpath('flatiron.flag').is_file():
            p.parent.joinpath('flatiron.flag').unlink()
        if force and p.parent.joinpath('register_me.flag').is_file():
            p.parent.joinpath('register_me.flag').unlink()
        write_flag_file(flag_file, file_list)
        print(flag_file)


def create_transfer_flags(root_data_folder, force=False, file_list=None):
    ses_path = Path(root_data_folder).glob('**/raw_behavior_data')
    for p in ses_path:
        flag_file = Path(p).parent.joinpath('extract_me.flag')
        write_flag_file(flag_file)
        print(flag_file)


def create_create_flags(root_data_folder, force=False, file_list=None):
    ses_path = Path(root_data_folder).glob('**/raw_behavior_data')
    for p in ses_path:
        flag_file = Path(p).parent.joinpath('create_me.flag')
        write_flag_file(flag_file)
        print(flag_file)


def create_flags(root_data_folder: str or Path, flags: list,
                 force: bool = False, file_list: list = None) -> None:
    ses_path = Path(root_data_folder).glob('**/raw_behavior_data')
    for p in ses_path:
        if 'create' in flags:
            create_create_flags(root_data_folder, force=force, file_list=file_list)
        if 'transfer' in flags:
            create_transfer_flags(root_data_folder, force=force, file_list=file_list)
        if 'extract' in flags:
            create_extract_flags(root_data_folder, force=force, file_list=file_list)
        if 'register' in flags:
            create_register_flags(root_data_folder, force=force, file_list=file_list)

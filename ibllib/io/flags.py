#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, January 22nd 2019, 12:32:14 pm
from pathlib import Path
import logging

logger_ = logging.getLogger('ibllib')


def read_flag_file(fname):
    """
    Flag files are ``*.flag`` files within a session folder used to schedule some jobs
    If they are empty, should return True

    :param fname: full file path of the flag file
    :type fname: str or pahlib.Path
    :return: None
    """
    # the flag file may contains specific file names for a targeted extraction
    with open(fname) as fid:
        save = list(filter(None, fid.read().splitlines()))
    # if empty, extract everything by default
    if len(save) == 0:
        save = True
    return save


def excise_flag_file(fname, removed_files=None):
    """
    Remove one or several specific files if they figure within the file
    If no file is left, deletes the flag.

    :param fname: full file path of the flag file
    :type fname: str or pahlib.Path
    :return: None
    """
    if not removed_files:
        return
    file_names = read_flag_file(fname)
    # if the file is empty, can't remove a specific file and return
    if len(file_names) == 0:
        return
    if isinstance(removed_files, str):
        removed_files = [removed_files]
    new_file_names = list(set(file_names).difference(set(removed_files)))
    # if the resulting file has no files in it, delete
    if len(new_file_names) == 0:
        Path(fname).unlink()
    else:
        write_flag_file(fname, file_list=new_file_names, clobber=True)


def write_flag_file(fname, file_list: list = None, clobber=False):
    """
    Flag files are ``*.flag`` files within a session folder used to schedule some jobs
    Each line references to a file to extract or register

    :param fname: full file path of the flag file
    :type fname: str or pathlib.Path
    :param file_list: None or list of relative paths to write in the file
    :type file_list: list
    :param clobber: (False) overwrites the flag file if any
    :type clobber: bool, optional
    :return: None
    """
    exists = Path(fname).exists()
    if exists:
        has_files = Path(fname).stat().st_size != 0
    else:
        has_files = False
    if isinstance(file_list, str) and file_list:
        file_list = [file_list]
    if isinstance(file_list, bool):
        file_list = None
    if clobber:
        mode = 'w+'
    elif exists and has_files and file_list:
        mode = 'a+'
        file_list = [''] + file_list
    else:
        mode = 'w+'
        if exists and not has_files:
            file_list = []
    with open(fname, mode) as fid:
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
        logger_.info('created flag: ' + str(flag_file))


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
        logger_.info('created flag: ' + str(flag_file))


def create_transfer_flags(root_data_folder, force=False, file_list=None):
    ses_path = Path(root_data_folder).glob('**/raw_behavior_data')
    for p in ses_path:
        flag_file = Path(p).parent.joinpath('extract_me.flag')
        write_flag_file(flag_file)
        logger_.info('created flag: ' + str(flag_file))


def create_create_flags(root_data_folder, force=False, file_list=None):
    ses_path = Path(root_data_folder).glob('**/raw_behavior_data')
    for p in ses_path:
        flag_file = Path(p).parent.joinpath('create_me.flag')
        write_flag_file(flag_file)
        logger_.info('created flag: ' + str(flag_file))


def create_compress_flags(root_data_folder, clobber=False):
    #  only create flags for raw_video_data folders:
    video_paths = Path(root_data_folder).glob('**/raw_video_data')
    for video_path in video_paths:
        ses_path = video_path.parent
        flag_file = ses_path.joinpath('compress_video.flag')
        vfiles = video_path.rglob('*.avi')
        for vfile in vfiles:
            logger_.info(str(vfile.relative_to(ses_path)) + ' added to ' + str(flag_file))
            write_flag_file(flag_file, file_list=str(vfile.relative_to(ses_path)), clobber=clobber)
    return
    # add audio flags to the list as well
    audio_paths = Path(root_data_folder).glob('**/raw_behavior__data')
    for audio_path in audio_paths:
        ses_path = audio_path.parent
        flag_file = ses_path.joinpath('compress_audio.flag')
        afiles = audio_path.rglob('*.wav')
        for afile in afiles:
            logger_.info(str(afile.relative_to(ses_path)) + ' added to ' + str(flag_file))
            write_flag_file(flag_file, file_list=str(afile.relative_to(ses_path)))


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


def delete_flags(root_data_folder):
    for f in Path(root_data_folder).rglob('*.flag'):
        f.unlink()
        logger_.info('deleted flag: ' + str(f))
    for f in Path(root_data_folder).rglob('*.error'):
        f.unlink()
        logger_.info('deleted flag: ' + str(f))

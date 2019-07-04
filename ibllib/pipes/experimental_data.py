#!/usr/bin/env python
"""
Entry point to system commands for IBL pipeline.

>>> python one_iblrig.py extract /path/to/my/session/ [--dry=True]
>>> python one_iblrig.py register /path/to/my/session/ [--dry=True]
>>> python one_iblrig.py create /path/to/my/session/ [--dry=True]
>>> python one_iblrig.py compress_video /path/to/my/session/ [--dry=True --count=4]
>>> python one_iblrig.py compress_audio /path/to/my/session/ [--dry=True --count=5]
"""

import logging
from pathlib import Path, PureWindowsPath
import subprocess

from ibllib.pipes import extract_session
from oneibl.registration import RegistrationClient
from oneibl.one import ONE
from ibllib.io import flags

logger = logging.getLogger('ibllib')
# set the logging level to paranoid
logger.setLevel('INFO')


def extract(root_data_folder, dry=False):
    extract_session.bulk(root_data_folder, dry=dry)


def register(root_data_folder, dry=False, one=None):
    # registration part
    if not one:
        one = ONE()
    rc = RegistrationClient(one=one)
    rc.register_sync(root_data_folder, dry=dry)


def create(root_data_folder, dry=False, one=None):
    # create the sessions by lookin
    if not one:
        one = ONE()
    rc = RegistrationClient(one=one)
    rc.create_sessions(root_data_folder, dry=dry)


def compress_audio(root_data_folder, dry=False, max_sessions=None):
    command = 'ffmpeg -i {file_name}.wav -c:a flac -nostats {file_name}.flac'
    _compress(root_data_folder, command, 'compress_audio.flag', dry=dry, max_sessions=max_sessions)


def compress_video(root_data_folder, dry=False, max_sessions=None):
    command = ('ffmpeg -i {file_name}.avi -codec:v libx264 -preset slow -crf 29 '
               '-nostats -loglevel 0 -codec:a copy {file_name}.mp4')
    _compress(root_data_folder, command, 'compress_video.flag', dry=dry, max_sessions=max_sessions)


def _compress(root_data_folder, command, flag_pattern, dry=False, max_sessions=None):
    #  runs a command of the form command = "ls -1 {file_name}.avi"
    c = 0
    for flag_file in Path(root_data_folder).rglob(flag_pattern):
        ses_path = flag_file.parent
        files2compress = flags.read_flag_file(flag_file)
        for f2c in files2compress:
            cfile = ses_path.joinpath(PureWindowsPath(f2c))
            c += 1
            if max_sessions and c > max_sessions:
                return
            print(cfile)
            if dry:
                continue
            if not cfile.exists():
                logger.error('NON-EXISTING RAW FILE: ' + str(cfile))
            # run the compression command redirecting output
            cfile.parent.joinpath(cfile.stem)
            # if the output file already exists, overwrite it
            outfile = cfile.parent / (cfile.stem + '.mp4')
            if outfile.exists():
                outfile.unlink()
            command2run = command.format(file_name=cfile.parent.joinpath(cfile.stem))
            process = subprocess.Popen(command2run, shell=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE)
            info, error = process.communicate()
            if process.returncode != 0:
                logger.error('COMPRESSION FAILED FOR ' + str(cfile))
                flags.excise_flag_file(flag_file, removed_files=f2c)
                with open(cfile.parent.joinpath('extract.error'), 'w+') as fid:
                    fid.write(command2run)
                    fid.write(error.decode())
                continue
            # if the command was successful delete the original file
            cfile.unlink()
            # then remove the file from the compress flag file
            flags.excise_flag_file(flag_file, removed_files=f2c)
            # and add the file to register_me.flag
            flags.write_flag_file(ses_path.joinpath('register_me.flag'), file_list=cfile.stem)

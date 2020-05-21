#!/usr/bin/env python
"""
Entry point to system commands for IBL behaviour pipeline.
Each function below corresponds to a command-line tool.
"""

import logging
from pathlib import Path, PureWindowsPath
import subprocess
import json

from ibllib.io import flags, raw_data_loaders
from ibllib.pipes import extract_session
from oneibl.registration import RegistrationClient
from oneibl.one import ONE

_logger = logging.getLogger('ibllib')
# set the logging level to paranoid
_logger.setLevel('INFO')


def _compress(root_data_folder, command, flag_pattern, dry=False, max_sessions=None):
    #  runs a command of the form command = "ls -1 {file_name}.avi"
    c = 0
    for flag_file in Path(root_data_folder).rglob(flag_pattern):
        ses_path = flag_file.parent
        files2compress = flags.read_flag_file(flag_file)
        if isinstance(files2compress, bool):
            Path(flag_file).unlink()
            continue
        for f2c in files2compress:
            cfile = ses_path.joinpath(PureWindowsPath(f2c))
            c += 1
            if max_sessions and c > max_sessions:
                return
            print(cfile)
            if dry:
                continue
            if not cfile.exists():
                _logger.error(f'NON-EXISTING RAW FILE: {cfile}. Skipping...')
                continue
            if flag_file.exists():
                flag_file.unlink()
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
                _logger.error('COMPRESSION FAILED FOR ' + str(cfile))
                with open(cfile.parent.joinpath('extract.error'), 'w+') as fid:
                    fid.write(command2run)
                    fid.write(error.decode())
            else:
                # if the command was successful delete the original file
                cfile.unlink()
                # and add the file to register_me.flag
                flags.write_flag_file(ses_path.joinpath('register_me.flag'), file_list=cfile.stem)


def create(root_data_folder, dry=False, one=None):
    # create the sessions by lookin
    if not one:
        one = ONE()
    rc = RegistrationClient(one=one)
    rc.create_sessions(root_data_folder, dry=dry)


# 01_extract_training
def extract(root_data_folder, dry=False):
    """
    Extracts behaviour only
    """
    extract_session.bulk(root_data_folder, dry=dry, glob_flag='**/extract_me.flag')


# 02_register
def register(root_data_folder, dry=False, one=None):
    # registration part
    if not one:
        one = ONE()
    rc = RegistrationClient(one=one)
    rc.register_sync(root_data_folder, dry=dry)


# 03_compress_videos
def compress_video(root_data_folder, dry=False, max_sessions=None):
    command = ('ffmpeg -i {file_name}.avi -codec:v libx264 -preset slow -crf 29 '
               '-nostats -loglevel 0 -codec:a copy {file_name}.mp4')
    _compress(root_data_folder, command, 'compress_video.flag', dry=dry, max_sessions=max_sessions)


# 04_audio_training
def audio_training(root_data_folder, dry=False, max_sessions=False):
    from ibllib.io.extractors import training_audio as audio
    audio_flags = Path(root_data_folder).rglob('audio_training.flag')
    c = 0
    for flag in audio_flags:
        c += 1
        if max_sessions and c > max_sessions:
            return
        _logger.info(flag)
        if dry:
            continue
        session_path = flag.parent
        try:
            settings = raw_data_loaders.load_settings(session_path)
            typ = extract_session.get_task_extractor_type(settings.get('PYBPOD_PROTOCOL'))
        except json.decoder.JSONDecodeError:
            typ = 'unknown'
        # this extractor is only for biased and training sessions
        if typ not in ['biased', 'training', 'habituation']:
            flag.unlink()
            continue
        audio.extract_sound(session_path, save=True, delete=True)
        flag.unlink()
        session_path.joinpath('register_me.flag').touch()

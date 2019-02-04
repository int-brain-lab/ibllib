"""
python one_iblrig.py extract /path/to/my/session/
python one_iblrig.py register /path/to/my/session/
python one_iblrig.py create /path/to/my/session/
python one_iblrig.py compress_video /path/to/my/session/
python one_iblrig.py compress_audio /path/to/my/session/
... --dry=True
... --dry True
"""

import logging
import argparse
from pathlib import Path
import subprocess

from alf import extract_session
from oneibl.registration import RegistrationClient
from oneibl.one import ONE
from ibllib.io import flags

logger = logging.getLogger('ibllib')
# set the logging level to paranoid
logger.setLevel('INFO')


def extract(root_data_folder, dry=False):
    extract_session.bulk(root_data_folder, dry=dry)


def register(root_data_folder, dry=False):
    # registration part
    one = ONE()
    rc = RegistrationClient(one=one)
    rc.register_sync(root_data_folder, dry=dry)


def create(root_data_folder, dry=False):
    # create the sessions by lookin
    one = ONE()
    rc = RegistrationClient(one=one)
    rc.create_sessions(root_data_folder, dry=dry)


def compress_audio(root_data_folder, dry=False, max_sessions=None):
    command = 'ffmpeg -i {file_name}.wav -c:a flac {file_name}.flac'
    _compress(root_data_folder, command, 'compress_audio.flag', dry=dry, max_sessions=max_sessions)


def compress_video(root_data_folder, dry=False, max_sessions=None):
    command = ('ffmpeg -i {file_name}.avi -codec:v libx264 -preset slow -crf 35'
               ' -codec:a copy {file_name}.mp4')
    _compress(root_data_folder, command, 'compress_video.flag', dry=dry, max_sessions=max_sessions)


def _compress(root_data_folder, command, flag_pattern, dry=False, max_sessions=None):
    #  runs a command of the form command = "ls -1 {file_name}.avi"
    c = 0
    for flag_file in Path(root_data_folder).rglob(flag_pattern):
        ses_path = flag_file.parent
        files2compress = flags.read_flag_file(flag_file)
        for f2c in files2compress:
            cfile = ses_path.joinpath(f2c)
            print(cfile)
            c += 1
            if max_sessions and c > max_sessions:
                return
            if dry:
                continue
            # run the compression command redirecting output
            command2run = command.format(file_name=cfile.parent.joinpath(cfile.stem))
            run = subprocess.run(command2run, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 shell=True)
            pstr = run.stdout.decode('utf-8') + run.stderr.decode('utf-8')
            if run.returncode == 0:
                logger.info(pstr)
            else:
                logger.error(pstr)
                continue
            # if the command was successful delete the original file
            cfile.unlink()
            # then remove the file from the compress flag file
            flags.excise_flag_file(flag_file, removed_files=f2c)
            # and add the file to register_me.flag
            flags.write_flag_file(ses_path.joinpath('register_me.flag'), file_list=cfile.stem)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('action', help='Action: create/extract/register ')
    parser.add_argument('folder', help='A Folder containing a session')
    parser.add_argument('--dry', help='Dry Run', required=False, default=False, type=bool)
    args = parser.parse_args()  # returns data from the options specified (echo)
    assert(Path(args.folder).exists())
    if args.action == 'extract':
        extract(args.folder, dry=args.dry)
    if args.action == 'register':
        register(args.folder, dry=args.dry)
    if args.action == 'create':
        create(args.folder, dry=args.dry)
    print('done')

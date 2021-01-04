from pathlib import Path
import subprocess
import logging

_logger = logging.getLogger('ibllib')


def compress(file_in, file_out, command, remove_original=True):
    """
    runs a command of the form 'ffmpeg -i {file_in} -c:a flac -nostats {file_out}'
    using a supbprocess
    audio compression for ephys:
    `"ffmpeg -i {file_in} -c:a flac -nostats {file_out}"`
    video compression for ephys:
    `"('ffmpeg -i {file_in} -codec:v libx264 -preset slow -crf 17 '
                   '-nostats -loglevel 0 -codec:a copy {file_out}')"`

    :param file_in: full file path of input
    :param file_out: full file path of output
    :param command: string ready to be formatted with `file_in` and `file_out`
    return: file_out
    """

    file_in = Path(file_in)
    file_out = Path(file_out)
    # if the output file already exists, overwrite it
    if file_out.exists():
        file_out.unlink()
    command2run = command.format(file_in=str(file_in), file_out=str(file_out))
    process = subprocess.Popen(command2run, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    info, error = process.communicate()
    if process.returncode != 0:
        _logger.error(f'compression failed for {file_in}: {error}')
        _logger.error(command2run)
        return process.returncode, None
    else:
        # if the command was successful delete the original file
        if remove_original:
            file_in.unlink()
        return process.returncode, file_out


def iblrig_video_compression(session_path, command):
    """
    Compresses avi files from the raw_videao_data folder into mp4. Once the compression completes
    the avi file is removed. If mp4 already exist, the funciton returns their file path
    :param session_path:
    :param command:
    for ephys:
    >>> command = ('ffmpeg -i {file_in} -y -codec:v libx264 -preset slow -crf 17 '
    >>>            '-nostats -loglevel 0 -codec:a copy {file_out}')
    for training:
    >>> command = ('ffmpeg -i {file_in} -y -codec:v libx264 -preset slow -crf 29 '
    >>>            '-nostats -loglevel 0 -codec:a copy {file_out}')
    :return: list of compressed files
    """
    output_files = list(session_path.joinpath("raw_video_data").rglob('_iblrig_*.mp4'))
    rig_avi_files = list(session_path.joinpath("raw_video_data").rglob('_iblrig_*.avi'))
    # first compress everything (the rationale is not to delete anything if there is a crash)
    for file_in in rig_avi_files:
        _logger.info(f" compressing {file_in}")
        file_out = file_in.with_suffix('.mp4')
        status, fout = compress(file_in=file_in, file_out=file_out,
                                command=command, remove_original=False)
        output_files.append(fout)
    # then remove everything
    for file_in in rig_avi_files:
        file_in.unlink()
    return output_files

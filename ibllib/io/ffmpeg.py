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

    :param file_in: full file path of input
    :param file_out: full file path of output
    :param command: string ready to be formatted with `file_in` and `file_out`
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
        return process.returncode, None
    else:
        # if the command was successful delete the original file
        if remove_original:
            file_in.unlink()
        return process.returncode, file_out

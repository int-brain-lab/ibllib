import ctypes
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import logging
from pathlib import Path
from typing import Union, List
from inspect import signature
import uuid
import socket

import spikeglx
from iblutil.io import hashfile, params
from iblutil.util import range_str
from one.alf.files import get_session_path
from one.alf.spec import is_uuid_string, is_session_path, describe
from one.api import ONE

import ibllib.io.flags as flags
import ibllib.io.raw_data_loaders as raw
from ibllib.io.misc import delete_empty_folders
import ibllib.io.session_params as sess_params

log = logging.getLogger(__name__)

DEVICE_FLAG_MAP = {'neuropixel': 'ephys',
                   'cameras': 'video',
                   'widefield': 'widefield',
                   'sync': 'sync'}


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
            raise ValueError('No "Subjects" folder in children folders')
        elif len(spath) > 1:
            raise ValueError(f'Multiple "Subjects" folder in children folders: {spath}')
        else:
            folder = folder / spath[0]

    return folder


def cli_ask_default(prompt: str, default: str):
    """
    Prompt the user for input, display the default option and return user input or default
    :param prompt: String to display to user
    :param default: The default value to return if user doesn't enter anything
    :return: User input or default
    """
    return input(f'{prompt} [default: {default}]: ') or default


def cli_ask_options(prompt: str, options: list, default_idx: int = 0) -> str:
    parsed_options = [str(x) for x in options]
    if default_idx is not None:
        parsed_options[default_idx] = f"[{parsed_options[default_idx]}]"
    options_str = " (" + " | ".join(parsed_options) + ")> "
    ans = input(prompt + options_str) or str(options[default_idx])
    if ans not in [str(x) for x in options]:
        return cli_ask_options(prompt, options, default_idx=default_idx)
    return ans


def behavior_exists(session_path: str) -> bool:
    session_path = Path(session_path)
    behavior_path = session_path / "raw_behavior_data"
    if behavior_path.exists():
        return True
    return any(session_path.glob('raw_task_data_*'))


def check_transfer(src_session_path, dst_session_path):
    """
    Check all the files in the source directory match those in the destination directory. Function
    will throw assertion errors/exceptions if number of files do not match, file names do not
    match, or if file sizes do not match.

    :param src_session_path: The source directory that was copied
    :param dst_session_path: The copy target directory
    """
    src_files = sorted([x for x in Path(src_session_path).rglob('*') if x.is_file()])
    dst_files = sorted([x for x in Path(dst_session_path).rglob('*') if x.is_file()])
    assert len(src_files) == len(dst_files), 'Not all files transferred'
    for s, d in zip(src_files, dst_files):
        assert s.name == d.name, 'file name mismatch'
        assert s.stat().st_size == d.stat().st_size, 'file size mismatch'


def rename_session(session_path: str, new_subject=None, new_date=None, new_number=None,
                   ask: bool = False) -> Path:
    """Rename a session.  Prompts the user for the new subject name, data and number and then moves
    session path to new session path.

    :param session_path: A session path to rename
    :type session_path: str
    :param new_subject: A new subject name, if none provided, the user is prompted for one
    :param new_date: A new session date, if none provided, the user is prompted for one
    :param new_number: A new session number, if none provided, the user is prompted for one
    :param ask: used to ensure prompt input from user, defaults to False
    :type ask: bool
    :return: The renamed session path
    :rtype: Path
    """
    session_path = get_session_path(session_path)
    if session_path is None:
        raise ValueError('Session path not valid ALF session folder')
    mouse = session_path.parts[-3]
    date = session_path.parts[-2]
    sess = session_path.parts[-1]
    new_mouse = new_subject or mouse
    new_date = new_date or date
    new_sess = new_number or sess
    if ask:
        new_mouse = input(f"Please insert subject NAME [current value: {mouse}]> ")
        new_date = input(f"Please insert new session DATE [current value: {date}]> ")
        new_sess = input(f"Please insert new session NUMBER [current value: {sess}]> ")

    new_session_path = Path(*session_path.parts[:-3]).joinpath(new_mouse, new_date,
                                                               new_sess.zfill(3))
    assert is_session_path(new_session_path), 'invalid subject, date or number'

    if new_session_path.exists():
        ans = input(f'Warning: session path {new_session_path} already exists.\nWould you like to '
                    f'move {new_session_path} to a backup directory? [y/N] ')
        if (ans or 'n').lower() in ['n', 'no']:
            print(f'Manual intervention required, data exists in the following directory: '
                  f'{session_path}')
            return
        if backup_session(new_session_path):
            print(f'Backup was successful, removing directory {new_session_path}...')
            shutil.rmtree(str(new_session_path), ignore_errors=True)
    shutil.move(str(session_path), str(new_session_path))
    print(session_path, "--> renamed to:")
    print(new_session_path)

    return new_session_path


def backup_session(session_path):
    """Used to move the contents of a session to a backup folder, likely before the folder is
    removed.

    :param session_path: A session path to be backed up
    :return: True if directory was backed up or exits if something went wrong
    :rtype: Bool
    """
    bk_session_path = Path()
    if Path(session_path).exists():
        try:
            bk_session_path = Path(*session_path.parts[:-4]).joinpath(
                "Subjects_backup_renamed_sessions", Path(*session_path.parts[-3:]))
            Path(bk_session_path.parent).mkdir(parents=True)
            print(f"Created path: {bk_session_path.parent}")
            # shutil.copytree(session_path, bk_session_path, dirs_exist_ok=True)
            shutil.copytree(session_path, bk_session_path)  # python 3.7 compatibility
            print(f"Copied contents from {session_path} to {bk_session_path}")
            return True
        except FileExistsError:
            log.error(f"A backup session for the given path already exists: {bk_session_path}, "
                      f"manual intervention is necessary.")
            raise
        except shutil.Error:
            log.error(f'Some kind of copy error occurred when moving files from {session_path} to '
                      f'{bk_session_path}')
            log.error(shutil.Error)
    else:
        log.error(f"The given session path does not exist: {session_path}")
        return False


def copy_with_check(src, dst, **kwargs):
    dst = Path(dst)
    if dst.exists() and Path(src).stat().st_size == dst.stat().st_size:
        return dst
    elif dst.exists():
        dst.unlink()
    return shutil.copy2(src, dst, **kwargs)


def transfer_session_folders(local_sessions: list, remote_subject_folder, subfolder_to_transfer):
    """
    Used to determine which local session folders should be transferred to which remote session folders, will prompt the user
    when necessary.

    Parameters
    ----------
    local_sessions : list
        Required list of local session folder paths to sync to local server.
    remote_subject_folder : str, pathlib.Path
        The remote location of the subject folder (typically pulled from the params).
    subfolder_to_transfer : str
        Which subfolder to sync

    Returns
    -------
    list of tuples
        For each session, a tuple of (source, destination) of attempted file transfers.
    list of bool
        A boolean True/False for success/failure of the transfer.
    """
    transfer_list = []  # list of sessions to transfer
    skip_list = ""  # "list" of sessions to skip and the reason for the skip
    # Iterate through all local sessions in the given list
    for local_session in local_sessions:
        # Set expected remote_session location and perform simple error state checks
        remote_session = remote_subject_folder.joinpath(*local_session.parts[-3:])
        # Skip session if ...
        if subfolder_to_transfer:
            if not local_session.joinpath(subfolder_to_transfer).exists():
                msg = f"{local_session} - skipping session, no '{subfolder_to_transfer}' folder found locally"
                log.warning(msg)
                skip_list += msg + "\n"
                continue
        if not remote_session.parent.exists():
            msg = f"{local_session} - no matching remote session date folder found for the given local session"
            log.info(msg)
            skip_list += msg + "\n"
            continue
        if not behavior_exists(remote_session):
            msg = f"{local_session} - skipping session, no behavior data found in remote folder {remote_session}"
            log.warning(msg)
            skip_list += msg + "\n"
            continue

        # Determine if there are multiple session numbers from the date path
        local_sessions_for_date = get_session_numbers_from_date_path(local_session.parent)
        remote_sessions_for_date = get_session_numbers_from_date_path(remote_session.parent)
        remote_session_pick = None
        if len(local_sessions_for_date) > 1 or len(remote_sessions_for_date) > 1:
            # Format folder size output for end user to review
            local_session_numbers_with_size = remote_session_numbers_with_size = ""
            for lsfd in local_sessions_for_date:
                size_in_gb = round(get_directory_size(local_session.parent / lsfd, in_gb=True), 2)
                local_session_numbers_with_size += lsfd + " (" + str(size_in_gb) + " GB)\n"
            for rsfd in remote_sessions_for_date:
                size_in_gb = round(get_directory_size(remote_session.parent / rsfd, in_gb=True), 2)
                remote_session_numbers_with_size += rsfd + " (" + str(size_in_gb) + " GB)\n"
            log.info(f"\n\nThe following local session folder(s) were found on this acquisition PC:\n\n"
                     f"{''.join(local_session_numbers_with_size)}\nThe following remote session folder(s) were found on the "
                     f"server:\n\n{''.join(remote_session_numbers_with_size)}\n")

            def _remote_session_picker(sessions_for_date):
                resp = "s"
                resp_invalid = True
                while resp_invalid:  # loop until valid user input
                    resp = input(f"\n\n--- USER INPUT NEEDED ---\nWhich REMOTE session number would you like to transfer your "
                                 f"local session to? Options {range_str(map(int, sessions_for_date))} or "
                                 f"[s]kip/[h]elp/[e]xit> ").strip().lower()
                    if resp == "h":
                        print("An example session filepath:\n")
                        describe("number")  # Explain what a session number is
                        input("Press enter to continue")
                    elif resp == "s" or resp == "e":  # exit loop
                        resp_invalid = False
                    elif len(resp) <= 3:
                        resp_invalid = False if [i for i in sessions_for_date if int(resp) == int(i)] else None
                    else:
                        print("Invalid response. Please try again.")
                return resp

            log.info(f"Evaluation for local session "
                     f"{local_session.parts[-3]}/{local_session.parts[-2]}/{local_session.parts[-1]}...")
            user_response = _remote_session_picker(remote_sessions_for_date)
            if user_response == "s":
                msg = f"{local_session} - Local session skipped due to user input"
                log.info(msg)
                skip_list += msg + "\n"
                continue
            elif user_response == "e":
                log.info("Exiting, no files transferred.")
                return
            else:
                remote_session_pick = remote_session.parent / user_response.zfill(3)

        # Append to the transfer_list
        transfer_tuple = (local_session, remote_session_pick) if remote_session_pick else (local_session, remote_session)
        transfer_list.append(transfer_tuple)
        log.info(f"{transfer_tuple[0]}, {transfer_tuple[1]} - Added to the transfer list")

    # Verify that the number of local transfer_list entries match the number of remote transfer_list entries
    if len(transfer_list) != len(set(dst for _, dst in transfer_list)):
        raise RuntimeError(
            "An invalid combination of sessions were picked; the most likely cause of this error is multiple local "
            "sessions being selected for a single remote session. Please rerun the script."
        )

    # Call rsync/rdiff function for every entry in the transfer list
    success = []
    for src, dst in transfer_list:
        if subfolder_to_transfer:
            success.append(rsync_paths(src / subfolder_to_transfer, dst / subfolder_to_transfer))
        else:
            success.append(rsync_paths(src, dst))
        if not success[-1]:
            log.error("File transfer failed, check log for reason.")

    # Notification to user for any transfers were skipped
    log.warning(f"Video transfers that were not completed:\n\n{skip_list}") if skip_list else log.info("No transfers skipped.")
    return transfer_list, success


def transfer_folder(src: Path, dst: Path, force: bool = False) -> None:
    """functionality has been replaced by transfer_session_folders function"""
    print(f"Attempting to copy:\n{src}\n--> {dst}")
    if force:
        print(f"Removing {dst}")
        shutil.rmtree(dst, ignore_errors=True)
    else:
        try:
            check_transfer(src, dst)
            print("All files already copied, use force=True to re-copy")
            return
        except AssertionError:
            pass
    print(f"Copying all files:\n{src}\n--> {dst}")
    # rsync_folder(src, dst, '**transfer_me.flag')
    if sys.version_info.minor < 8:
        # dirs_exist_ok kwarg not supported in < 3.8
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, copy_function=copy_with_check)
    else:
        shutil.copytree(src, dst, dirs_exist_ok=True, copy_function=copy_with_check)
    # If folder was created delete the src_flag_file
    if check_transfer(src, dst) is None:
        print("All files copied")
    # rdiff-backup --compare /tmp/tmpw9o1zgn0 /tmp/tmp82gg36rm
    # No changes found.  Directory matches archive data.


def load_params_dict(params_fname: str) -> dict:
    params_fpath = Path(params.getfile(params_fname))
    if not params_fpath.exists():
        return None
    with open(params_fpath, "r") as f:
        out = json.load(f)
    return out


def load_videopc_params():
    if not load_params_dict("videopc_params"):
        create_videopc_params()
    return load_params_dict("videopc_params")


def load_ephyspc_params():
    if not load_params_dict("ephyspc_params"):
        create_ephyspc_params()
    return load_params_dict("ephyspc_params")


def create_basic_transfer_params(param_str='transfer_params', local_data_path=None,
                                 remote_data_path=None, clobber=False, **kwargs):
    """Create some basic parameters common to all acquisition rigs.

    Namely prompt user for the local root data path and the remote (lab server) data path.
    NB: All params stored in uppercase by convention.

    Parameters
    ----------
    param_str : str
        The name of the parameters to load/save.
    local_data_path : str, pathlib.Path
        The local root data path, stored with the DATA_FOLDER_PATH key.  If None, user is prompted.
    remote_data_path : str, pathlib.Path
        The local root data path, stored with the REMOTE_DATA_FOLDER_PATH key.  If None, user is prompted.
    clobber : bool
        If True, any parameters in existing parameter file not found as keyword args will be removed,
        otherwise the user is prompted for these also.
    **kwargs
        Extra parameters to set. If value is None, the user is prompted.

    Returns
    -------
    dict
        The parameters written to disc.

    Examples
    --------
    Set up basic transfer parameters for modality acquisition PC

    >>> par = create_basic_transfer_params()

    Set up basic transfer paramers without prompting the user

    >>> par = create_basic_transfer_params(
    ...     local_data_path='/iblrig_data/Subjects',
    ...     remote_data_path='/mnt/iblserver.champalimaud.pt/ibldata/Subjects')

    Prompt user for extra parameter using custom prompt (will call function with current default)

    >>> from functools import partial
    >>> par = create_basic_transfer_params(
    ...     custom_arg=partial(cli_ask_default, 'Please enter custom arg value'))
    """
    parameters = params.as_dict(params.read(param_str, {})) or {}
    if local_data_path is None:
        local_data_path = parameters.get('DATA_FOLDER_PATH')
        if not local_data_path or clobber:
            local_data_path = cli_ask_default("Where's your LOCAL 'Subjects' data folder?", local_data_path)
    parameters['DATA_FOLDER_PATH'] = local_data_path

    if remote_data_path is None:
        remote_data_path = parameters.get('REMOTE_DATA_FOLDER_PATH')
        if not remote_data_path or clobber:
            remote_data_path = cli_ask_default("Where's your REMOTE 'Subjects' data folder?", remote_data_path)
    parameters['REMOTE_DATA_FOLDER_PATH'] = remote_data_path

    # Deal with extraneous parameters
    for k, v in kwargs.items():
        if callable(v):  # expect function handle with default value as input
            n_pars = len(signature(v).parameters)
            parameters[k.upper()] = v(parameters.get(k.upper())) if n_pars > 0 else v()
        elif v is None:  # generic prompt for key
            parameters[k.upper()] = cli_ask_default(
                f'Enter a value for parameter {k.upper()}', parameters.get(k.upper())
            )
        else:  # assign value to parameter
            parameters[k.upper()] = str(v)

    defined = list(map(str.upper, ('DATA_FOLDER_PATH', 'REMOTE_DATA_FOLDER_PATH', 'TRANSFER_LABEL', *kwargs.keys())))
    if clobber:
        # Delete any parameters in parameter dict that were not passed as keyword args into function
        parameters = {k: v for k, v in parameters.items() if k in defined}
    else:
        # Prompt for any other parameters that weren't passed into function
        for k in filter(lambda x: x not in defined, map(str.upper, parameters.keys())):
            parameters[k] = cli_ask_default(f'Enter a value for parameter {k}', parameters.get(k))

    if 'TRANSFER_LABEL' not in parameters:
        parameters['TRANSFER_LABEL'] = f'{socket.gethostname()}_{uuid.getnode()}'

    # Write parameters
    params.write(param_str, parameters)
    return parameters


def create_videopc_params(force=False, silent=False):
    if Path(params.getfile("videopc_params")).exists() and not force:
        print(f"{params.getfile('videopc_params')} exists already, exiting...")
        print(Path(params.getfile("videopc_params")).exists())
        return
    if silent:
        data_folder_path = r"D:\iblrig_data\Subjects"
        remote_data_folder_path = r"\\iblserver.champalimaud.pt\ibldata\Subjects"
        body_cam_idx = 0
        left_cam_idx = 1
        right_cam_idx = 2
    else:
        data_folder_path = cli_ask_default(
            r"Where's your LOCAL 'Subjects' data folder?", r"D:\iblrig_data\Subjects"
        )
        remote_data_folder_path = cli_ask_default(
            r"Where's your REMOTE 'Subjects' data folder?",
            r"\\iblserver.champalimaud.pt\ibldata\Subjects",
        )
        body_cam_idx = cli_ask_default("Please select the index of the BODY camera", "0")
        left_cam_idx = cli_ask_default("Please select the index of the LEFT camera", "1")
        right_cam_idx = cli_ask_default("Please select the index of the RIGHT camera", "2")

    param_dict = {
        "DATA_FOLDER_PATH": data_folder_path,
        "REMOTE_DATA_FOLDER_PATH": remote_data_folder_path,
        "BODY_CAM_IDX": body_cam_idx,
        "LEFT_CAM_IDX": left_cam_idx,
        "RIGHT_CAM_IDX": right_cam_idx,
    }
    params.write("videopc_params", param_dict)
    print(f"Created {params.getfile('videopc_params')}")
    print(param_dict)
    return param_dict


def create_ephyspc_params(force=False, silent=False):
    if Path(params.getfile("ephyspc_params")).exists() and not force:
        print(f"{params.getfile('ephyspc_params')} exists already, exiting...")
        print(Path(params.getfile("ephyspc_params")).exists())
        return
    if silent:
        data_folder_path = r"D:\iblrig_data\Subjects"
        remote_data_folder_path = r"\\iblserver.champalimaud.pt\ibldata\Subjects"
        probe_types = {"PROBE_TYPE_00": "3A", "PROBE_TYPE_01": "3B"}
    else:
        data_folder_path = cli_ask_default(
            r"Where's your LOCAL 'Subjects' data folder?", r"D:\iblrig_data\Subjects"
        )
        remote_data_folder_path = cli_ask_default(
            r"Where's your REMOTE 'Subjects' data folder?",
            r"\\iblserver.champalimaud.pt\ibldata\Subjects",
        )
        n_probes = int(cli_ask_default("How many probes are you using?", '2'))
        assert 100 > n_probes > 0, 'Please enter number between 1, 99 inclusive'
        probe_types = {}
        for i in range(n_probes):
            probe_types[f'PROBE_TYPE_{i:02}'] = cli_ask_options(
                f"What's the type of PROBE {i:02}?", ["3A", "3B"])
    param_dict = {
        "DATA_FOLDER_PATH": data_folder_path,
        "REMOTE_DATA_FOLDER_PATH": remote_data_folder_path,
        **probe_types
    }
    params.write("ephyspc_params", param_dict)
    print(f"Created {params.getfile('ephyspc_params')}")
    print(param_dict)
    return param_dict


def rdiff_install() -> bool:
    """
    For windows:
    * if the rdiff-backup executable does not already exist on the system
      * downloads rdiff-backup zip file
      * copies the executable to the C:\tools folder

    For linux/mac:
    * runs a pip install rdiff-backup

    Returns:
        True when install is successful, False when an error is encountered
    """
    if os.name == "nt":
        # ensure tools folder exists
        tools_folder = "C:\\tools\\"
        os.mkdir(tools_folder) if not Path(tools_folder).exists() else None

        rdiff_cmd_loc = tools_folder + "rdiff-backup.exe"
        if not Path(rdiff_cmd_loc).exists():
            import requests
            import zipfile
            from io import BytesIO

            url = "https://github.com/rdiff-backup/rdiff-backup/releases/download/v2.0.5/rdiff-backup-2.0.5.win32exe.zip"
            log.info("Downloading zip file for rdiff-backup.")
            # Download the file by sending the request to the URL, ensure success by status code
            if requests.get(url).status_code == 200:
                log.info("Download complete for rdiff-backup zip file.")
                # extracting the zip file contents
                zipfile = zipfile.ZipFile(BytesIO(requests.get(url).content))
                zipfile.extractall("C:\\Temp")
                rdiff_folder_name = zipfile.namelist()[0]  # attempting a bit of future-proofing
                # move the executable to the C:\tools folder
                shutil.copy("C:\\Temp\\" + rdiff_folder_name + "rdiff-backup.exe", rdiff_cmd_loc)
                shutil.rmtree("C:\\Temp\\" + rdiff_folder_name)  # cleanup temp folder
                try:  # attempt to call the rdiff command
                    subprocess.run([rdiff_cmd_loc, "--version"], check=True)
                except (FileNotFoundError, subprocess.CalledProcessError) as e:
                    log.error("rdiff-backup installation did not complete.\n", e)
                    return False
                return True
            else:
                log.error("Download request status code not 200, something did not go as expected.")
                return False
    else:  # anything not Windows
        try:  # package should not be installed via the requirements.txt to accommodate windows
            subprocess.run(["pip", "install", "rdiff-backup"], check=True)
        except subprocess.CalledProcessError as e:
            log.error("rdiff-backup pip install did not complete.\n", e)
            return False
        return True


def get_directory_size(dir_path: Path, in_gb=False) -> float:
    """
    Used to determine total size of all files in a given session_path, including all child directories

    Args:
        dir_path (Path): path we want to get the total size of
        in_gb (bool): set to True for returned value to be in gigabytes

    Returns:
        float: sum of all files in the given directory path (in bytes by default, in GB if specified)
    """
    total = 0
    with iter(os.scandir(dir_path)) as it:
        for entry in it:
            if entry.is_file():
                total += entry.stat().st_size
            elif entry.is_dir():
                total += get_directory_size(entry.path)
    if in_gb:
        return total / 1024 / 1024 / 1024  # in GB
    return total  # in bytes


def get_session_numbers_from_date_path(date_path: Path) -> list:
    """
    Retrieves session numbers when given a date path

    Args:
        date_path (Path): path to date, i.e. \\\\server\\some_lab\\Subjects\\Date"

    Returns:
        (list): Found sessions as a sorted list
    """
    contents = Path(date_path).glob('*')
    folders = filter(lambda x: x.is_dir() and re.match(r'^\d{3}$', x.name), contents)
    sessions_as_set = set(map(lambda x: x.name, folders))
    sessions_as_sorted_list = sorted(sessions_as_set)
    return sessions_as_sorted_list


def rsync_paths(src: Path, dst: Path) -> bool:
    """
    Used to run the rsync algorithm via a rdiff-backup command on the paths contained on the provided source and destination.
    This function relies on the rdiff-backup package and is run from the command line, i.e. subprocess.run(). Full documentation
    can be found here - https://rdiff-backup.net/docs/rdiff-backup.1.html

    Parameters
    ----------
    src : Path
        source path that contains data to be transferred
    dst : Path
        destination path that will receive the transferred data

    Returns
    -------
    bool
        True for success, False for failure

    Raises
    ------
    FileNotFoundError, subprocess.CalledProcessError
    """
    # Set rdiff_cmd_loc based on OS type (assuming C:\tools is not in Windows PATH environ)
    rdiff_cmd_loc = "C:\\tools\\rdiff-backup.exe" if os.name == "nt" else "rdiff-backup"
    try:  # Check if rdiff-backup command is available
        subprocess.run([rdiff_cmd_loc, "--version"], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        if not rdiff_install():  # Attempt to install rdiff
            log.error("rdiff-backup command is unavailable, transfers can not continue.\n", e)
            raise

    log.info("Attempting to transfer data: " + str(src) + " -> " + str(dst))
    WindowsInhibitor().inhibit() if os.name == "nt" else None  # prevent Windows from going to sleep
    try:
        rsync_command = [rdiff_cmd_loc, "--verbosity", str(0),
                         "--create-full-path", "--backup-mode", "--no-acls", "--no-eas",
                         "--no-file-statistics", "--exclude", "**transfer_me.flag",
                         str(src), str(dst)]
        subprocess.run(rsync_command, check=True)
        time.sleep(1)  # give rdiff-backup a second to complete all logging operations
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        log.error("Transfer failed with code %i.\n", e.returncode)
        if e.stderr:
            log.error(e.stderr)
        return False
    log.info("Validating transfer completed...")
    try:  # Validate the transfers succeeded
        rsync_validate = [rdiff_cmd_loc, "--verify", str(dst)]
        subprocess.run(rsync_validate, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        log.error(f"Validation for destination {dst} failed.\n", e)
        return False
    log.info("Cleaning up rdiff files...")
    shutil.rmtree(dst / "rdiff-backup-data")
    WindowsInhibitor().uninhibit() if os.name == 'nt' else None  # allow Windows to go to sleep
    return True


def confirm_ephys_remote_folder(
    local_folder=False, remote_folder=False, force=False, iblscripts_folder=False
):
    pars = load_ephyspc_params()

    if not local_folder:
        local_folder = pars["DATA_FOLDER_PATH"]
    if not remote_folder:
        remote_folder = pars["REMOTE_DATA_FOLDER_PATH"]
    local_folder = Path(local_folder)
    remote_folder = Path(remote_folder)
    # Check for Subjects folder
    local_folder = subjects_data_folder(local_folder, rglob=True)
    remote_folder = subjects_data_folder(remote_folder, rglob=True)

    print("LOCAL:", local_folder)
    print("REMOTE:", remote_folder)
    src_session_paths = [x.parent for x in local_folder.rglob("transfer_me.flag")]

    if not src_session_paths:
        print("Nothing to transfer, exiting...")
        return

    for session_path in src_session_paths:
        print(f"\nFound session: {session_path}")
        # Rename ephys files
        # FIXME: if transfer has failed and wiring file is there renaming will fail!
        rename_ephys_files(str(session_path))
        # Move ephys files
        move_ephys_files(str(session_path))
        # Copy wiring files
        copy_wiring_files(str(session_path), iblscripts_folder)
        try:
            create_alyx_probe_insertions(str(session_path))
        except BaseException as e:
            print(
                e,
                "\nCreation failed, please create the probe insertions manually.",
                "Continuing transfer...",
            )
        msg = f"Transfer to {remote_folder} with the same name?"
        resp = input(msg + "\n[y]es/[r]ename/[s]kip/[e]xit\n ^\n> ") or "y"
        resp = resp.lower()
        print(resp)
        if resp not in ["y", "r", "s", "e", "yes", "rename", "skip", "exit"]:
            return confirm_ephys_remote_folder(
                local_folder=local_folder,
                remote_folder=remote_folder,
                force=force,
                iblscripts_folder=iblscripts_folder,
            )
        elif resp == "y" or resp == "yes":
            pass
        elif resp == "r" or resp == "rename":
            session_path = rename_session(session_path)
            if not session_path:
                continue
        elif resp == "s" or resp == "skip":
            continue
        elif resp == "e" or resp == "exit":
            return

        remote_session_path = remote_folder / Path(*session_path.parts[-3:])
        if not behavior_exists(remote_session_path):
            print(f"No behavior folder found in {remote_session_path}: skipping session...")
            return
        # TODO: Check flagfiles on src.and dst + alf dir in session folder then remove
        # Try catch? wher catch condition is force transfer maybe
        transfer_folder(
            session_path / "raw_ephys_data", remote_session_path / "raw_ephys_data", force=force
        )
        # if behavior extract_me.flag exists remove it, because of ephys flag
        flag_file = session_path / "transfer_me.flag"
        flag_file.unlink()
        if (remote_session_path / "extract_me.flag").exists():
            (remote_session_path / "extract_me.flag").unlink()
        # Create remote flags
        create_ephys_transfer_done_flag(remote_session_path)
        check_create_raw_session_flag(remote_session_path)


def probe_labels_from_session_path(session_path: Union[str, Path]) -> List[str]:
    """
    Finds ephys probes according to the metadata spikeglx files. Only returns first subfolder
    name under raw_ephys_data folder, ie. raw_ephys_data/probe00/copy_of_probe00 won't be returned
    If there is a NP2.4 probe with several shanks, create several probes
    :param session_path:
    :return: list of strings
    """
    plabels = []
    raw_ephys_folder = session_path.joinpath('raw_ephys_data')
    for meta_file in raw_ephys_folder.rglob('*.ap.meta'):
        if meta_file.parents[1] != raw_ephys_folder:
            continue
        meta = spikeglx.read_meta_data(meta_file)
        nshanks = spikeglx._get_nshanks_from_meta(meta)
        if nshanks > 1:
            for i in range(nshanks):
                plabels.append(meta_file.parts[-2] + 'abcdefghij'[i])
        else:
            plabels.append(meta_file.parts[-2])
    plabels.sort()
    return plabels


def create_alyx_probe_insertions(
    session_path: str,
    force: bool = False,
    one: object = None,
    model: str = None,
    labels: list = None,
):
    if one is None:
        one = ONE(cache_rest=None)
    eid = session_path if is_uuid_string(session_path) else one.path2eid(session_path)
    if eid is None:
        log.warning("Session not found on Alyx: please create session before creating insertions")
    if model is None:
        probe_model = spikeglx.get_neuropixel_version_from_folder(session_path)
        pmodel = "3B2" if probe_model == "3B" else probe_model
    else:
        pmodel = model
    labels = labels or probe_labels_from_session_path(session_path)
    # create the qc fields in the json field
    qc_dict = {}
    qc_dict.update({"qc": "NOT_SET"})
    qc_dict.update({"extended_qc": {}})

    # create the dictionary
    insertions = []
    for plabel in labels:
        insdict = {"session": eid, "name": plabel, "model": pmodel, "json": qc_dict}
        # search for the corresponding insertion in Alyx
        alyx_insertion = one.alyx.get(f'/insertions?&session={eid}&name={plabel}', clobber=True)
        # if it doesn't exist, create it
        if len(alyx_insertion) == 0:
            alyx_insertion = one.alyx.rest("insertions", "create", data=insdict)
        else:
            iid = alyx_insertion[0]["id"]
            if force:
                alyx_insertion = one.alyx.rest("insertions", "update", id=iid, data=insdict)
            else:
                alyx_insertion = alyx_insertion[0]
        insertions.append(alyx_insertion)
    return insertions


def create_ephys_flags(session_folder: str):
    """
    Create flags for processing an ephys session.  Should be called after move_ephys_files
    :param session_folder: A path to an ephys session
    :return:
    """
    session_path = Path(session_folder)
    flags.write_flag_file(session_path.joinpath("extract_ephys.flag"))
    flags.write_flag_file(session_path.joinpath("raw_ephys_qc.flag"))
    for probe_path in session_path.joinpath('raw_ephys_data').glob('probe*'):
        flags.write_flag_file(probe_path.joinpath("spike_sorting.flag"))


def create_ephys_transfer_done_flag(session_folder: str) -> None:
    session_path = Path(session_folder)
    flags.write_flag_file(session_path.joinpath("ephys_data_transferred.flag"))


def create_video_transfer_done_flag(session_folder: str) -> None:
    session_path = Path(session_folder)
    flags.write_flag_file(session_path.joinpath("video_data_transferred.flag"))


def create_transfer_done_flag(session_folder: str, flag_name: str) -> None:
    session_path = Path(session_folder)
    flags.write_flag_file(session_path.joinpath(f"{flag_name}_data_transferred.flag"))


def check_create_raw_session_flag(session_folder: str) -> None:
    session_path = Path(session_folder)

    # if we have an experiment description file read in whether we expect video, ephys widefield etc, don't do it just based
    # on the task protocol
    experiment_description = sess_params.read_params(session_path)

    def check_status(expected, flag):
        if expected is not False and flag.exists():
            return True
        if expected is False and not flag.exists():
            return True
        else:
            return False

    if experiment_description is not None:

        if any(session_path.joinpath('_devices').glob('*')):
            return

        # Find the devices in the experiment description file
        devices = list()
        for key in DEVICE_FLAG_MAP.keys():
            if experiment_description.get('devices', {}).get(key, None) is not None:
                devices.append(key)
        # In case of widefield the sync also needs to be in it's own folder
        if 'widefield' in devices:
            devices.append('sync')

        expected_flags = [session_path.joinpath(f'{DEVICE_FLAG_MAP[dev]}_data_transferred.flag') for dev in devices]

        expected = []
        flag_files = []
        for dev, fl in zip(devices, expected_flags):
            status = check_status(dev, fl)
            if status:
                flag_files.append(fl)
            expected.append(status)

        # In this case all the copying has completed
        if all(expected):
            # make raw session flag
            flags.write_flag_file(session_path.joinpath("raw_session.flag"))
            # and unlink individual copy flags
            for fl in flag_files:
                fl.unlink()

        return

    ephys = session_path.joinpath("ephys_data_transferred.flag")
    video = session_path.joinpath("video_data_transferred.flag")

    sett = raw.load_settings(session_path)
    if sett is None:
        log.error(f"No flag created for {session_path}")
        return

    is_biased = True if "biased" in sett["PYBPOD_PROTOCOL"] else False
    is_training = True if "training" in sett["PYBPOD_PROTOCOL"] else False
    is_habituation = True if "habituation" in sett["PYBPOD_PROTOCOL"] else False
    if video.exists() and (is_biased or is_training or is_habituation):
        flags.write_flag_file(session_path.joinpath("raw_session.flag"))
        video.unlink()
    if video.exists() and ephys.exists():
        flags.write_flag_file(session_path.joinpath("raw_session.flag"))
        ephys.unlink()
        video.unlink()


def rename_ephys_files(session_folder: str) -> None:
    """rename_ephys_files is system agnostic (3A, 3B1, 3B2).
    Renames all ephys files to Alyx compatible filenames. Uses get_new_filename.

    :param session_folder: Session folder path
    :type session_folder: str
    :return: None - Changes names of files on filesystem
    :rtype: None
    """
    session_path = Path(session_folder)
    ap_files = session_path.rglob("*.ap.*")
    lf_files = session_path.rglob("*.lf.*")
    nidq_files = session_path.rglob("*.nidq.*")

    for apf in ap_files:
        new_filename = get_new_filename(apf.name)
        shutil.move(str(apf), str(apf.parent / new_filename))

    for lff in lf_files:
        new_filename = get_new_filename(lff.name)
        shutil.move(str(lff), str(lff.parent / new_filename))

    for nidqf in nidq_files:
        # Ignore wiring files: these are usually created after the file renaming however this
        # function may be called a second time upon failed transfer.
        if 'wiring' in nidqf.name:
            continue
        new_filename = get_new_filename(nidqf.name)
        shutil.move(str(nidqf), str(nidqf.parent / new_filename))


def get_new_filename(filename: str) -> str:
    """get_new_filename is system agnostic (3A, 3B1, 3B2).
    Gets an alyx compatible filename from any spikeglx ephys file.

    :param filename: Name of an ephys file
    :return: New name for ephys file
    """
    root = "_spikeglx_ephysData"
    parts = filename.split('.')
    if len(parts) < 3:
        raise ValueError(fr'unrecognized filename "{filename}"')
    pattern = r'.*(?P<gt>_g\d+_t\d+)'
    if not (match := re.match(pattern, parts[0])):
        raise ValueError(fr'unrecognized filename "{filename}"')
    return '.'.join([root + match.group(1), *parts[1:]])


def move_ephys_files(session_folder: str) -> None:
    """move_ephys_files is system agnostic (3A, 3B1, 3B2).
    Moves all properly named ephys files to appropriate locations for transfer.
    Use rename_ephys_files function before this one.

    :param session_folder: Session folder path
    :type session_folder: str
    :return: None - Moves files on filesystem
    :rtype: None
    """
    session_path = Path(session_folder)
    raw_ephys_data_path = session_path / "raw_ephys_data"

    imec_files = session_path.rglob("*.imec*")
    for imf in imec_files:
        # For 3B system probe0x == imecx
        probe_number = re.match(r'_spikeglx_ephysData_g\d_t\d.imec(\d+).*', imf.name)
        if not probe_number:
            # For 3A system imec files must be in a 'probexx' folder
            probe_label = re.search(r'probe\d+', str(imf))
            assert probe_label, f'Cannot assign probe number to file {imf}'
            probe_label = probe_label.group()
        else:
            probe_number, = probe_number.groups()
            probe_label = f'probe{probe_number.zfill(2)}'
        raw_ephys_data_path.joinpath(probe_label).mkdir(exist_ok=True)
        shutil.move(imf, raw_ephys_data_path.joinpath(probe_label, imf.name))

    # NIDAq files (3B system only)
    nidq_files = session_path.rglob("*.nidq.*")
    for nidqf in nidq_files:
        shutil.move(str(nidqf), str(raw_ephys_data_path / nidqf.name))
    # Delete all empty folders recursively
    delete_empty_folders(raw_ephys_data_path, dry=False, recursive=True)


def create_custom_ephys_wirings(iblscripts_folder: str):
    iblscripts_path = Path(iblscripts_folder)
    PARAMS = load_ephyspc_params()
    probe_set = set(v for k, v in PARAMS.items() if k.startswith('PROBE_TYPE'))

    params_path = iblscripts_path.parent / "iblscripts_params"
    params_path.mkdir(parents=True, exist_ok=True)
    wirings_path = iblscripts_path / "deploy" / "ephyspc" / "wirings"
    for k, v in PARAMS.items():
        if not k.startswith('PROBE_TYPE_'):
            continue
        probe_label = f'probe{k[-2:]}'
        if v not in ('3A', '3B'):
            raise ValueError(f'Unsupported probe type "{v}"')
        shutil.copy(
            wirings_path / f"{v}.wiring.json", params_path / f"{v}_{probe_label}.wiring.json"
        )
        print(f"Created {v}.wiring.json in {params_path} for {probe_label}")
    if "3B" in probe_set:
        shutil.copy(wirings_path / "nidq.wiring.json", params_path / "nidq.wiring.json")
        print(f"Created nidq.wiring.json in {params_path}")
    print(f"\nYou can now modify your wiring files from folder {params_path}")


def get_iblscripts_folder():
    return str(Path().cwd().parent.parent)


def copy_wiring_files(session_folder, iblscripts_folder):
    """Run after moving files to probe folders"""
    PARAMS = load_ephyspc_params()
    if PARAMS["PROBE_TYPE_00"] != PARAMS["PROBE_TYPE_01"]:
        print("Having different probe types is not supported")
        raise NotImplementedError()
    session_path = Path(session_folder)
    iblscripts_path = Path(iblscripts_folder)
    iblscripts_params_path = iblscripts_path.parent / "iblscripts_params"
    wirings_path = iblscripts_path / "deploy" / "ephyspc" / "wirings"
    termination = '.wiring.json'
    # Determine system
    ephys_system = PARAMS["PROBE_TYPE_00"]
    # Define where to get the files from (determine if custom wiring applies)
    src_wiring_path = iblscripts_params_path if iblscripts_params_path.exists() else wirings_path
    probe_wiring_file_path = src_wiring_path / f"{ephys_system}{termination}"

    if ephys_system == "3B":
        # Copy nidq file
        nidq_files = session_path.rglob("*.nidq.bin")
        for nidqf in nidq_files:
            nidq_wiring_name = ".".join(str(nidqf.name).split(".")[:-1]) + termination
            shutil.copy(
                str(src_wiring_path / f"nidq{termination}"),
                str(session_path / "raw_ephys_data" / nidq_wiring_name),
            )
    # If system is either (3A OR 3B) copy a wiring file for each ap.bin file
    for binf in session_path.rglob("*.ap.bin"):
        probe_label = re.search(r'probe\d+', str(binf))
        if probe_label:
            wiring_name = ".".join(str(binf.name).split(".")[:-2]) + termination
            dst_path = session_path / "raw_ephys_data" / probe_label.group() / wiring_name
            shutil.copy(probe_wiring_file_path, dst_path)


def multi_parts_flags_creation(root_paths: Union[list, str, Path]) -> List[Path]:
    """
    Creates the sequence files to run spike sorting in batches
    A sequence file is a json file with the following fields:
     sha1: a unique hash of the metafiles involved
     probe: a string with the probe name
     index: the index within the sequence
     nrecs: the length of the sequence
     files: a list of files
    :param root_paths:
    :return:
    """
    from one.alf import io as alfio
    # "001/raw_ephys_data/probe00/_spikeglx_ephysData_g0_t0.imec0.ap.meta",
    if isinstance(root_paths, str) or isinstance(root_paths, Path):
        root_paths = [root_paths]
    recordings = {}
    for root_path in root_paths:
        for meta_file in root_path.rglob("*.ap.meta"):
            # we want to make sure that the file is just under session_path/raw_ephys_data/{probe_label}
            session_path = alfio.files.get_session_path(meta_file)
            raw_ephys_path = session_path.joinpath('raw_ephys_data')
            if meta_file.parents[1] != raw_ephys_path:
                log.warning(f"{meta_file} is not in a probe directory and will be skipped")
                continue
            # stack the meta-file in the probe label key of the recordings dictionary
            plabel = meta_file.parts[-2]
            recordings[plabel] = recordings.get(plabel, []) + [meta_file]
    # once we have all of the files
    for k in recordings:
        nrecs = len(recordings[k])
        recordings[k].sort()
        # the identifier of the overarching recording sequence is the hash of hashes of the files
        m = hashlib.sha1()
        for i, meta_file in enumerate(recordings[k]):
            hash = hashfile.sha1(meta_file)
            m.update(hash.encode())
        # writes the sequence files
        for i, meta_file in enumerate(recordings[k]):
            sequence_file = meta_file.parent.joinpath(meta_file.name.replace('ap.meta', 'sequence.json'))
            with open(sequence_file, 'w+') as fid:
                json.dump(dict(sha1=m.hexdigest(), probe=k, index=i, nrecs=len(recordings[k]),
                               files=list(map(str, recordings[k]))), fid)
            log.info(f"{k}: {i}/{nrecs} written sequence file {recordings}")
    return recordings


class WindowsInhibitor:
    """Prevent OS sleep/hibernate in windows; code from:
    https://github.com/h3llrais3r/Deluge-PreventSuspendPlus/blob/master/preventsuspendplus/core.py
    API documentation:
    https://msdn.microsoft.com/en-us/library/windows/desktop/aa373208(v=vs.85).aspx"""
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001

    def __init__(self):
        pass

    def inhibit(self):
        print("Preventing Windows from going to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(WindowsInhibitor.ES_CONTINUOUS | WindowsInhibitor.ES_SYSTEM_REQUIRED)

    def uninhibit(self):
        print("Allowing Windows to go to sleep")
        ctypes.windll.kernel32.SetThreadExecutionState(WindowsInhibitor.ES_CONTINUOUS)

import json
import shutil
from pathlib import Path

import alf.folders as folders
import ibllib.io.params as params


# TODO: Tests!!!!!!
def cli_ask_default(prompt: str, default: str):
    dflt = " [default: {}]: "
    dflt.format(default)
    ans = input(prompt + dflt) or default
    return ans


def cli_ask_options(prompt: str, options: list, default_idx: int = 0) -> str:
    parsed_options = [str(x) for x in options]
    if default_idx is not None:
        parsed_options[default_idx] = f"[{parsed_options[default_idx]}]"
    options_str = ' (' + ' | '.join(parsed_options) + ')> '
    ans = input(prompt + options_str) or str(options[default_idx])
    if ans not in [str(x) for x in options]:
        return cli_ask_options(prompt, options, default_idx=default_idx)
    return ans


def behavior_exists(session_path: str) -> bool:
    session_path = Path(session_path)
    behavior_path = session_path / 'raw_behavior_data'
    if behavior_path.exists():
        return True
    return False


def check_transfer(src_session_path: str or Path, dst_session_path: str or Path):
    src_files = sorted([x for x in Path(src_session_path).rglob('*') if x.is_file()])
    dst_files = sorted([x for x in Path(dst_session_path).rglob('*') if x.is_file()])
    for s, d in zip(src_files, dst_files):
        assert(s.name == d.name)
        assert(s.stat().st_size == d.stat().st_size)
    return


def rename_session(session_path: str) -> Path:
    session_path = Path(folders.session_path(session_path))
    if session_path is None:
        return
    mouse = session_path.parts[-3]
    date = session_path.parts[-2]
    sess = session_path.parts[-1]
    new_mouse = input(
        f"Please insert mouse NAME [current value: {mouse}]> ") or mouse
    new_date = input(
        f"Please insert new session DATE [current value: {date}]> ") or date
    new_sess = input(
        f"Please insert new session NUMBER [current value: {sess}]> ") or sess
    new_session_path = Path(*session_path.parts[:-3]) / new_mouse / new_date / new_sess

    shutil.move(str(session_path), str(new_session_path))
    print(session_path, '--> renamed to:')
    print(new_session_path)

    return new_session_path


def transfer_folder(src: Path, dst: Path, force: bool = False) -> None:
    print(f"Attempting to copy:\n{src}\n--> {dst}")
    if force:
        print(f"Removing {dst}")
        shutil.rmtree(dst, ignore_errors=True)
    print(f"Copying all files:\n{src}\n--> {dst}")
    shutil.copytree(src, dst)
    # If folder was created delete the src_flag_file
    if check_transfer(src, dst) is None:
        print("All files copied")


def load_params_dict(params_fname: str) -> dict:
    params_fpath = Path(params.getfile(params_fname))
    if not params_fpath.exists():
        return None
    with open(params_fpath, 'r') as f:
        out = json.load(f)
    return out


def load_videopc_params():
    if not load_params_dict('videopc_params'):
        create_videopc_params()
    return load_params_dict('videopc_params')


def load_ephyspc_params():
    if not load_params_dict('ephyspc_params'):
        create_ephyspc_params()
    return load_params_dict('ephyspc_params')


def create_videopc_params(force=False):
    if Path(params.getfile('videopc_params')).exists() and not force:
        print(f"{params.getfile('videopc_params')} exists already, exiting...")
        print(Path(params.getfile('videopc_params')).exists())
        return
    data_folder_path = cli_ask_default(
        r"Where's your LOCAL 'Subjects' data folder?", r"D:\iblrig_data\Subjects")
    remote_data_folder_path = cli_ask_default(
        r"Where's your REMOTE 'Subjects' data folder?",
        r"\\iblserver.champalimaud.pt\ibldata\Subjects")
    body_cam_idx = cli_ask_default("Please select the index of the BODY camera", '0')
    left_cam_idx = cli_ask_default("Please select the index of the LEFT camera", '1')
    right_cam_idx = cli_ask_default("Please select the index of the RIGHT camera", '2')

    param_dict = {
        'DATA_FOLDER_PATH': data_folder_path,
        'REMOTE_DATA_FOLDER_PATH': remote_data_folder_path,
        'BODY_CAM_IDX': body_cam_idx,
        'LEFT_CAM_IDX': left_cam_idx,
        'RIGHT_CAM_IDX': right_cam_idx,
    }
    params.write('videopc_params', param_dict)
    print(f"Created {params.getfile('videopc_params')}")
    print(param_dict)
    return param_dict


def create_ephyspc_params(force=False):
    if Path(params.getfile('ephyspc_params')).exists() and not force:
        print(f"{params.getfile('ephyspc_params')} exists already, exiting...")
        print(Path(params.getfile('ephyspc_params')).exists())
        return
    data_folder_path = cli_ask_default(
        r"Where's your LOCAL 'Subjects' data folder?", r"D:\iblrig_data\Subjects")
    remote_data_folder_path = cli_ask_default(
        r"Where's your REMOTE 'Subjects' data folder?",
        r"\\iblserver.champalimaud.pt\ibldata\Subjects")
    probe_type_00 = cli_ask_options("What's the type of PROBE 00?", ['3A' '3B1' '3B2'])
    probe_type_01 = cli_ask_options("What's the type of PROBE 01?", ['3A' '3B1' '3B2'])
    param_dict = {
        'DATA_FOLDER_PATH': data_folder_path,
        'REMOTE_DATA_FOLDER_PATH': remote_data_folder_path,
        'PROBE_TYPE_00': probe_type_00,
        'PROBE_TYPE_01': probe_type_01,
    }
    params.write('ephyspc_params', param_dict)
    print(f"Created {params.getfile('ephyspc_params')}")
    print(param_dict)
    return param_dict


def confirm_video_remote_folder(local_folder=False, remote_folder=False, force=False):
    pars = load_videopc_params()

    if not local_folder:
        local_folder = pars['DATA_FOLDER_PATH']
    if not remote_folder:
        remote_folder = pars['REMOTE_DATA_FOLDER_PATH']
    local_folder = Path(local_folder)
    remote_folder = Path(remote_folder)
    # Check for Subjects folder
    local_folder = folders.subjects_data_folder(local_folder, rglob=True)
    remote_folder = folders.subjects_data_folder(remote_folder, rglob=True)

    print('LOCAL:', local_folder)
    print('REMOTE:', remote_folder)
    src_session_paths = [x.parent for x in local_folder.rglob("transfer_me.flag")]

    if not src_session_paths:
        print("Nothing to transfer, exiting...")
        return

    for session_path in src_session_paths:
        print(f"\nFound session: {session_path}")
        flag_file = session_path / 'transfer_me.flag'
        msg = f"Transfer to {remote_folder} with the same name?"
        resp = input(msg + "\n[y]es/[r]ename/[s]kip/[e]xit\n ^\n> ") or 'y'
        resp = resp.lower()
        print(resp)
        if resp not in ['y', 'r', 's', 'e', 'yes', 'rename', 'skip', 'exit']:
            return confirm_video_remote_folder(
                local_folder=local_folder, remote_folder=remote_folder)
        elif resp == 'y' or resp == 'yes':
            remote_session_path = remote_folder / Path(*session_path.parts[-3:])
            transfer_folder(
                session_path / 'raw_video_data',
                remote_session_path / 'raw_video_data',
                force=force)
            flag_file.unlink()
        elif resp == 'r' or resp == 'rename':
            new_session_path = rename_session(session_path)
            remote_session_path = remote_folder / Path(*new_session_path.parts[-3:])
            transfer_folder(
                new_session_path / 'raw_video_data',
                remote_session_path / 'raw_video_data')
            flag_file.unlink()
        elif resp == 's' or resp == 'skip':
            continue
        elif resp == 'e' or resp == 'exit':
            return


def confirm_ephys_remote_folder(local_folder=False, remote_folder=False):
    pars = load_ephyspc_params()

    if not local_folder:
        local_folder = pars['DATA_FOLDER_PATH']
    if not remote_folder:
        remote_folder = pars['REMOTE_DATA_FOLDER_PATH']
    local_folder = Path(local_folder)
    remote_folder = Path(remote_folder)
    # Check for Subjects folder
    local_folder = folders.subjects_data_folder(local_folder, rglob=True)
    remote_folder = folders.subjects_data_folder(remote_folder, rglob=True)

    print('LOCAL:', local_folder)
    print('REMOTE:', remote_folder)
    src_session_paths = [x.parent for x in local_folder.rglob("transfer_me.flag")]

    if not src_session_paths:
        print("Nothing to transfer, exiting...")
        return

    for session_path in src_session_paths:
        print(f"\nFound session: {session_path}")
        flag_file = session_path / 'transfer_me.flag'
        msg = f"Transfer to {remote_folder} with the same name?"
        resp = input(msg + "\n[y]es/[r]ename/[s]kip/[e]xit\n ^\n> ") or 'y'
        resp = resp.lower()
        print(resp)
        if resp not in ['y', 'r', 's', 'e', 'yes', 'rename', 'skip', 'exit']:
            return confirm_ephys_remote_folder(
                local_folder=local_folder, remote_folder=remote_folder)
        elif resp == 'y' or resp == 'yes':
            remote_session_path = remote_folder / Path(*session_path.parts[-3:])
            if not behavior_exists(remote_session_path):
                print(f"No behavior folder found in {remote_session_path}: skipping session...")
                continue
            transfer_folder(
                session_path / 'raw_ephys_data',
                remote_session_path / 'raw_ephys_data',
                force=False)
            flag_file.unlink()
            if (remote_session_path / 'extract_me.flag').exists():
                (remote_session_path / 'extract_me.flag').unlink()
            (remote_session_path / 'extract_ephys.flag').touch()
            (remote_session_path / 'ephys_qc.flag').touch()
        elif resp == 'r' or resp == 'rename':
            new_session_path = rename_session(session_path)
            remote_session_path = remote_folder / Path(*new_session_path.parts[-3:])
            if not behavior_exists(remote_session_path):
                print(f"No behavior folder found in {remote_session_path}: skipping session...")
                continue
            transfer_folder(
                new_session_path / 'raw_ephys_data',
                remote_session_path / 'raw_ephys_data')
            flag_file.unlink()
            (remote_session_path / 'extract_ephys.flag').touch()
            (remote_session_path / 'ephys_qc.flag').touch()
            if (remote_session_path / 'extract_me.flag').exists():
                (remote_session_path / 'extract_me.flag').unlink()
        elif resp == 's' or resp == 'skip':
            continue
        elif resp == 'e' or resp == 'exit':
            return

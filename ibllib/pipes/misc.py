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
        # TODO:Add here rename ephys files and move ephys files
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


def rename_ephys_files(session_folder: str) -> None:
    """rename_ephys_files is system agnostic (3A, 3B1, 3B2).
    Renames all ephys files to Alyx compatible filenames. Uses get_new_filename.

    :param session_folder: Session folder path
    :type session_folder: str
    :return: None - Changes names of files on filesystem
    :rtype: None
    """
    session_path = Path(session_folder)
    ap_files = session_path.rglob('*.ap.*')
    lf_files = session_path.rglob('*.lf.*')
    nidq_files = session_path.rglob('*.nidaq.*')

    for apf in ap_files:
        new_filename = get_new_filename(apf.name)
        shutil.move(str(apf), str(apf.parent / new_filename))

    for lff in lf_files:
        new_filename = get_new_filename(lff.name)
        shutil.move(str(lff), str(lff.parent / new_filename))

    for nidqf in nidq_files:
        new_filename = get_new_filename(nidqf.name)
        shutil.move(str(nidqf), str(nidqf.parent / new_filename))


def get_new_filename(filename: str) -> str:
    """get_new_filename is system agnostic (3A, 3B1, 3B2).
    Gets an alyx compatible filename from any spikeglx ephys file.

    :param filename: Name of an ephys file
    :type filename: str
    :return: New name for ephys file
    :rtype: str
    """
    root = '_spikeglx_ephysData'
    # gt = '_g0_t0' or 'error
    if '_g0_t0' in filename:
        gt = '_g0_t0'
    elif '_g0_t0' not in filename:
        raise(NotImplementedError)

    # ext = 'bin' or 'meta'
    if '.bin' in filename:
        ext = 'bin'
    elif '.meta' in filename:
        ext = 'meta'

    if '.nidq.' in filename:
        return '.'.join([root, gt, 'nidq', ext])

    # probe = 'imec0' or 'imec1'
    if '.imec0.' in filename:
        probe = 'imec0'
    elif '.imec1.' in filename:
        probe = 'imec1'
    elif '.imec.' in filename:
        probe = 'imec'

    # freq = 'ap' or 'lf'
    if '.ap.' in filename:
        freq = 'ap'
    elif '.lf.' in filename:
        freq = 'lf'

    return '.'.join([root, gt, probe, freq, ext])


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
    raw_ephys_data_path = session_path / 'raw_ephys_data'
    nidq_files = session_path.rglob('*.nidaq.*')
    # All nidq files go in the raw_eohys_data folder
    for nidqf in nidq_files:
        shutil.move(str(nidqf), str(raw_ephys_data_path / nidqf.name))
    probe00_path = raw_ephys_data_path / 'probe00'
    probe01_path = raw_ephys_data_path / 'probe01'
    # 3A system imec only
    imec_files = session_path.rglob('*.imec.*')
    for imf in imec_files:
        if 'probe00' in imf.name:
            shutil.move(str(imf), str(probe00_path / imf.name))
        elif 'probe01' in imf.name:
            shutil.move(str(imf), str(probe01_path / imf.name))

    # 3B system
    imec0_files = session_path.rglob('*.imec0.*')
    imec1_files = session_path.rglob('*.imec1.*')
    # All imec 0 in probe00 folder
    for i0f in imec0_files:
        shutil.move(str(i0f), str(probe00_path / i0f.name))
    # All imec 1 in probe01 folder
    for i1f in imec1_files:
        shutil.move(str(i1f), str(probe01_path / i1f.name))

# TODO: ADD wiring files from params folder that is not there yet.
# TODO: params_folder for ephyspc implementation
import json
import logging
import shutil
import hashlib
from pathlib import Path
import re
from typing import Union, List

import iblutil.io.params as params
from one.alf.spec import is_uuid_string, is_session_path
from one.alf.files import get_session_path
from one.api import ONE

from iblutil.io import hashfile
import ibllib.io.flags as flags
import ibllib.io.raw_data_loaders as raw
import ibllib.io.spikeglx as spikeglx
from ibllib.io.misc import delete_empty_folders

log = logging.getLogger("ibllib")


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
    return False


def check_transfer(src_session_path: str, dst_session_path: str):
    """
    Check all the files in the source directory match those in the destination directory.
    :param src_session_path: The source directory that was copied
    :param dst_session_path: The copy target directory
    :return:
    """
    src_files = sorted([x for x in Path(src_session_path).rglob('*') if x.is_file()])
    dst_files = sorted([x for x in Path(dst_session_path).rglob('*') if x.is_file()])
    assert len(src_files) == len(dst_files), 'Not all files transferred'
    for s, d in zip(src_files, dst_files):
        assert s.name == d.name, 'file name mismatch'
        assert s.stat().st_size == d.stat().st_size, 'file size mismatch'


def rename_session(session_path: str) -> Path:
    """
    Rename a session.  Prompts the user for the new subject name, data and number then moves
    session path to new session path.
    :param session_path: A session path to rename
    :return: The renamed session path
    """
    session_path = get_session_path(session_path)
    if session_path is None:
        raise ValueError('Session path not valid ALF session folder')
    mouse = session_path.parts[-3]
    date = session_path.parts[-2]
    sess = session_path.parts[-1]
    new_mouse = input(f"Please insert subject NAME [current value: {mouse}]> ") or mouse
    new_date = input(f"Please insert new session DATE [current value: {date}]> ") or date
    new_sess = input(f"Please insert new session NUMBER [current value: {sess}]> ") or sess
    new_session_path = Path(*session_path.parts[:-3]) / new_mouse / new_date / new_sess.zfill(3)
    assert is_session_path(new_session_path), 'invalid subject, date or number'

    shutil.move(str(session_path), str(new_session_path))
    print(session_path, "--> renamed to:")
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


def confirm_video_remote_folder(local_folder=False, remote_folder=False, force=False):
    pars = load_videopc_params()

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
        msg = f"Transfer to {remote_folder} with the same name?"
        resp = input(msg + "\n[y]es/[r]ename/[s]kip/[e]xit\n ^\n> ") or "y"
        resp = resp.lower()
        print(resp)
        if resp not in ["y", "r", "s", "e", "yes", "rename", "skip", "exit"]:
            return confirm_video_remote_folder(
                local_folder=local_folder, remote_folder=remote_folder, force=force
            )
        elif resp == "y" or resp == "yes":
            pass
        elif resp == "r" or resp == "rename":
            session_path = rename_session(session_path)
        elif resp == "s" or resp == "skip":
            continue
        elif resp == "e" or resp == "exit":
            return

        remote_session_path = remote_folder / Path(*session_path.parts[-3:])
        if not behavior_exists(remote_session_path):
            print(f"No behavior folder found in {remote_session_path}: skipping session...")
            return
        transfer_folder(
            session_path / "raw_video_data", remote_session_path / "raw_video_data", force=force
        )
        flag_file = session_path / "transfer_me.flag"
        flag_file.unlink()
        create_video_transfer_done_flag(remote_session_path)
        check_create_raw_session_flag(remote_session_path)


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
    :param session_path:
    :return: list of strings
    """
    plabels = []
    raw_ephys_folder = session_path.joinpath('raw_ephys_data')
    for meta_file in raw_ephys_folder.rglob('*.ap.meta'):
        if meta_file.parents[1] != raw_ephys_folder:
            continue
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
        print("Session not found on Alyx: please create session before creating insertions")
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


def check_create_raw_session_flag(session_folder: str) -> None:
    session_path = Path(session_folder)
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
    match = re.match(pattern, parts[0])
    if not match:  # py 3.8
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

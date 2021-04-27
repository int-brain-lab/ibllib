import json
import logging
import shutil
from pathlib import Path

import alf.folders as folders
import ibllib.io.flags as flags
import ibllib.io.params as params
import ibllib.io.raw_data_loaders as raw
import ibllib.io.spikeglx as spikeglx
from alf.io import is_uuid_string
from oneibl.one import ONE

log = logging.getLogger("ibllib")


def cli_ask_default(prompt: str, default: str):
    dflt = " [default: {}]: "
    dflt = dflt.format(default)
    ans = input(prompt + dflt) or default
    return ans


def cli_ask_options(prompt: str, options: list, default_idx: int = 0) -> str:
    parsed_options = [str(x) for x in options]
    if default_idx is not None:
        parsed_options[default_idx] = f"[{parsed_options[default_idx]}]"
    options_str = " (" + " | ".join(parsed_options) + ")> "
    ans = input(prompt + options_str) or str(options[default_idx])
    if ans not in [str(x) for x in options]:
        return cli_ask_options(prompt, options, default_idx=default_idx)
    return ans


def behavior_exists(session_path: str) -> bool:  # testOK
    session_path = Path(session_path)
    behavior_path = session_path / "raw_behavior_data"
    if behavior_path.exists():
        return True
    return False


def check_transfer(src_session_path: str, dst_session_path: str):
    # missing fail test
    src_files = sorted([x for x in Path(src_session_path).rglob("*") if x.is_file()])
    dst_files = sorted([x for x in Path(dst_session_path).rglob("*") if x.is_file()])
    assert_msg = "src_files != dst_files"
    assert len(src_files) == len(dst_files), assert_msg
    for s, d in zip(src_files, dst_files):
        assert s.name == d.name, assert_msg
        assert s.stat().st_size == d.stat().st_size, assert_msg


def rename_session(session_path: str) -> Path:
    session_path = Path(folders.session_path(session_path))
    if session_path is None:
        return
    mouse = session_path.parts[-3]
    date = session_path.parts[-2]
    sess = session_path.parts[-1]
    new_mouse = input(f"Please insert mouse NAME [current value: {mouse}]> ") or mouse
    new_date = input(f"Please insert new session DATE [current value: {date}]> ") or date
    new_sess = input(f"Please insert new session NUMBER [current value: {sess}]> ") or sess
    new_session_path = Path(*session_path.parts[:-3]) / new_mouse / new_date / new_sess

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
        probe_type_00 = "3A"
        probe_type_01 = "3B"
    else:
        data_folder_path = cli_ask_default(
            r"Where's your LOCAL 'Subjects' data folder?", r"D:\iblrig_data\Subjects"
        )
        remote_data_folder_path = cli_ask_default(
            r"Where's your REMOTE 'Subjects' data folder?",
            r"\\iblserver.champalimaud.pt\ibldata\Subjects",
        )
        probe_type_00 = cli_ask_options("What's the type of PROBE 00?", ["3A", "3B"])
        probe_type_01 = cli_ask_options("What's the type of PROBE 01?", ["3A", "3B"])
    param_dict = {
        "DATA_FOLDER_PATH": data_folder_path,
        "REMOTE_DATA_FOLDER_PATH": remote_data_folder_path,
        "PROBE_TYPE_00": probe_type_00,
        "PROBE_TYPE_01": probe_type_01,
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
    local_folder = folders.subjects_data_folder(local_folder, rglob=True)
    remote_folder = folders.subjects_data_folder(remote_folder, rglob=True)

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
    local_folder = folders.subjects_data_folder(local_folder, rglob=True)
    remote_folder = folders.subjects_data_folder(remote_folder, rglob=True)

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


def create_alyx_probe_insertions(
    session_path: str,
    force: bool = False,
    one: object = None,
    model: str = None,
    labels: list = None,
):
    if one is None:
        one = ONE()
    if is_uuid_string(session_path):
        eid = session_path
    else:
        eid = one.eid_from_path(session_path)
    if eid is None:
        print("Session not found on Alyx: please create session before creating insertions")
    if model is None:
        probe_model = spikeglx.get_neuropixel_version_from_folder(session_path)
        pmodel = "3B2" if probe_model == "3B" else probe_model
    else:
        pmodel = model
    raw_ephys_data_path = Path(session_path) / "raw_ephys_data"
    if labels is None:
        probe_labels = [
            x.name
            for x in Path(raw_ephys_data_path).glob("*")
            if x.is_dir() and ("00" in x.name or "01" in x.name)
        ]
    else:
        probe_labels = labels

    # create the qc fields in the json field
    qc_dict = {}
    qc_dict.update({"qc": "NOT_SET"})
    qc_dict.update({"extended_qc": {}})

    # create the dictionary
    for plabel in probe_labels:
        insdict = {"session": eid, "name": plabel, "model": pmodel, "json": qc_dict}
        # search for the corresponding insertion in Alyx
        alyx_insertion = one.alyx.rest(
            "insertions", "list", session=insdict["session"], name=insdict["name"]
        )
        # if it doesn't exist, create it
        if len(alyx_insertion) == 0:
            alyx_insertion = one.alyx.rest("insertions", "create", data=insdict)
        else:
            iid = alyx_insertion[0]["id"]
            if force:
                alyx_insertion = one.alyx.rest("insertions", "update", id=iid, data=insdict)
            else:
                alyx_insertion = alyx_insertion[0]


def create_ephys_flags(session_folder: str):
    session_path = Path(session_folder)
    flags.write_flag_file(session_path.joinpath("extract_ephys.flag"))
    flags.write_flag_file(session_path.joinpath("raw_ephys_qc.flag"))
    probe00_path = session_path / "raw_ephys_data" / "probe00"
    probe01_path = session_path / "raw_ephys_data" / "probe01"
    flags.write_flag_file(probe00_path.joinpath("spike_sorting.flag"))
    flags.write_flag_file(probe01_path.joinpath("spike_sorting.flag"))


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
    if video.exists() and is_biased:
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


def get_new_filename(filename: str) -> str:  # testOK
    """get_new_filename is system agnostic (3A, 3B1, 3B2).
    Gets an alyx compatible filename from any spikeglx ephys file.

    :param filename: Name of an ephys file
    :type filename: str
    :return: New name for ephys file
    :rtype: str
    """
    root = "_spikeglx_ephysData"
    # gt = '_g0_t0' or 'error
    gt = None
    for g in range(10):
        for t in range(10):
            if f"_g{g}_t{t}" in filename:
                gt = f"_g{g}_t{t}"
    else:
        if gt is None:
            raise NotImplementedError(
                f"Can not parse {filename} as an ephys file with gX_tY where"
                f"XY are single digits"
            )

    # ext = 'bin' or 'meta'
    if ".bin" in filename:
        ext = "bin"
    elif ".meta" in filename:
        ext = "meta"

    if ".nidq." in filename:
        return ".".join([root + gt, "nidq", ext])

    # probe = 'imec0' or 'imec1'
    if ".imec0." in filename:
        probe = "imec0"
    elif ".imec1." in filename:
        probe = "imec1"
    elif ".imec." in filename:
        probe = "imec"

    # freq = 'ap' or 'lf'
    if ".ap." in filename:
        freq = "ap"
    elif ".lf." in filename:
        freq = "lf"

    return ".".join([root + gt, probe, freq, ext])


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

    probe00_path = raw_ephys_data_path / "probe00"
    probe01_path = raw_ephys_data_path / "probe01"
    # 3A system imec only
    imec_files = session_path.rglob("*.imec.*")
    for imf in imec_files:
        if "probe00" in str(imf):
            shutil.move(str(imf), str(probe00_path / imf.name))
        elif "probe01" in str(imf):
            shutil.move(str(imf), str(probe01_path / imf.name))
    # 3B system
    imec0_files = session_path.rglob("*.imec0.*")
    imec1_files = session_path.rglob("*.imec1.*")
    nidq_files = session_path.rglob("*.nidq.*")
    # All imec 0 in probe00 folder
    for i0f in imec0_files:
        shutil.move(str(i0f), str(probe00_path / i0f.name))
    # All imec 1 in probe01 folder
    for i1f in imec1_files:
        shutil.move(str(i1f), str(probe01_path / i1f.name))
    # NIDAq files
    nidq_files = session_path.rglob("*.nidq.*")
    for nidqf in nidq_files:
        shutil.move(str(nidqf), str(raw_ephys_data_path / nidqf.name))
    # Delete all empty folders recursively
    from ibllib.io.misc import delete_empty_folders

    delete_empty_folders(raw_ephys_data_path, dry=False, recursive=True)


def create_custom_ephys_wirings(iblscripts_folder: str):
    iblscripts_path = Path(iblscripts_folder)
    PARAMS = load_ephyspc_params()
    if PARAMS["PROBE_TYPE_00"] != PARAMS["PROBE_TYPE_01"]:
        print("Having different probe types not supported")
        raise (NotImplementedError)

    params_path = iblscripts_path.parent / "iblscripts_params"
    params_path.mkdir(parents=True, exist_ok=True)
    wirings_path = iblscripts_path / "deploy" / "ephyspc" / "wirings"
    if "3A" in PARAMS["PROBE_TYPE_00"]:
        shutil.copy(
            str(wirings_path / "3A.wiring.json"), str(params_path / "3A_probe00.wiring.json")
        )
        shutil.copy(
            str(wirings_path / "3A.wiring.json"), str(params_path / "3A_probe01.wiring.json")
        )
        print(f"Created 3A.wiring.json in {params_path}")
    if "3B" in PARAMS["PROBE_TYPE_00"]:
        shutil.copy(
            str(wirings_path / "3B.wiring.json"), str(params_path / "3B_probe00.wiring.json")
        )
        shutil.copy(
            str(wirings_path / "3B.wiring.json"), str(params_path / "3B_probe01.wiring.json")
        )
        print(f"Created 3B.wiring.json in {params_path}")
        shutil.copy(str(wirings_path / "nidq.wiring.json"), str(params_path / "nidq.wiring.json"))
        print(f"Created nidq.wiring.json in {params_path}")
    print(f"\nYou can now modify your wirings files from folder {params_path}")


def get_iblscripts_folder():
    return str(Path().cwd().parent.parent)


def copy_wiring_files(session_folder, iblscripts_folder):
    """Run after moving files to probe folders"""
    PARAMS = load_ephyspc_params()
    if PARAMS["PROBE_TYPE_00"] != PARAMS["PROBE_TYPE_01"]:
        print("Having different probe types is not supported")
        raise (NotImplementedError)
    session_path = Path(session_folder)
    iblscripts_path = Path(iblscripts_folder)
    iblscripts_params_path = iblscripts_path.parent / "iblscripts_params"
    wirings_path = iblscripts_path / "deploy" / "ephyspc" / "wirings"
    probe00_path = session_path / "raw_ephys_data" / "probe00"
    probe01_path = session_path / "raw_ephys_data" / "probe01"
    termination = ".wiring.json"
    # Determine if custom wiring applies
    custom = True if iblscripts_params_path.exists() else False
    # Determine system
    ephys_system = PARAMS["PROBE_TYPE_00"]
    # Define where to get the files from
    if custom:
        src_wiring_path = iblscripts_params_path
        probe00_wiring_file_path = src_wiring_path / f"{ephys_system}_probe00.wiring.json"
        probe01_wiring_file_path = src_wiring_path / f"{ephys_system}_probe01.wiring.json"
    else:
        src_wiring_path = wirings_path
        probe00_wiring_file_path = src_wiring_path / f"{ephys_system}.wiring.json"
        probe01_wiring_file_path = src_wiring_path / f"{ephys_system}.wiring.json"

    if ephys_system == "3B":
        # Copy nidq file
        nidq_files = session_path.rglob("*.nidq.bin")
        for nidqf in nidq_files:
            nidq_wiring_name = ".".join(str(nidqf.name).split(".")[:-1]) + termination
            shutil.copy(
                str(src_wiring_path / "nidq.wiring.json"),
                str(session_path / "raw_ephys_data" / nidq_wiring_name),
            )
    # If system is either (3A OR 3B) copy a wiring file for each ap.bin file
    for binf in session_path.rglob("*.ap.bin"):
        wiring_name = ".".join(str(binf.name).split(".")[:-2]) + termination
        if "probe00" in str(binf):
            shutil.copy(str(probe00_wiring_file_path), str(probe00_path / wiring_name))
        if "probe01" in str(binf):
            shutil.copy(str(probe01_wiring_file_path), str(probe01_path / wiring_name))

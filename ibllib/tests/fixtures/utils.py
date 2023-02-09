#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Friday, October 9th 2020, 12:02:56 pm
import json
from pathlib import Path

from ibllib.io import session_params


def create_fake_session_folder(
    root_data_path, lab="fakelab", mouse="fakemouse", date="1900-01-01", num="001", increment=True
):
    root_data_path = Path(root_data_path)
    session_path = root_data_path / lab / "Subjects" / mouse / date / num
    if session_path.exists() and increment:
        num = str(int(num) + 1).zfill(3)
        return create_fake_session_folder(
            root_data_path, lab=lab, mouse=mouse, date=date, num=num, increment=increment
        )
    session_path = root_data_path / lab / "Subjects" / mouse / date / num

    session_path.mkdir(exist_ok=True, parents=True)
    return session_path


def create_fake_raw_ephys_data_folder(session_path, populate=True):
    """create_fake_raw_ephys_data_folder creates raw_ephys_data folder
    can populate with empty files with expected names

    :param session_path: [description]
    :type session_path: [type]
    :param populate: [description], defaults to True
    :type populate: bool, optional
    :return: [description]
    :rtype: [type]
    """
    session_path = Path(session_path)
    raw_ephys_data_path = session_path / "raw_ephys_data"
    raw_ephys_data_path.mkdir(exist_ok=True, parents=True)
    if populate:
        file_list = [
            "probe00/_spikeglx_ephysData_g0_t0.imec0.sync.npy",
            "probe00/_spikeglx_ephysData_g0_t0.imec0.lf.meta",
            "probe00/_spikeglx_ephysData_g0_t0.imec0.ap.cbin",
            "probe00/_spikeglx_ephysData_g0_t0.imec0.timestamps.npy",
            "probe00/_spikeglx_ephysData_g0_t0.imec0.lf.cbin",
            "probe00/_spikeglx_ephysData_g0_t0.imec0.ap.ch",
            "probe00/_spikeglx_ephysData_g0_t0.imec0.wiring.json",
            "probe00/_spikeglx_sync.times.probe00.npy",
            "probe00/_spikeglx_sync.channels.probe00.npy",
            "probe00/_spikeglx_sync.polarities.probe00.npy",
            "probe01/_spikeglx_ephysData_g0_t0.imec1.sync.npy",
            "probe01/_spikeglx_ephysData_g0_t0.imec1.lf.meta",
            "probe01/_spikeglx_ephysData_g0_t0.imec1.timestamps.cbin",
            "probe01/_spikeglx_ephysData_g0_t0.imec1.ap.cbin",
            "probe01/_spikeglx_ephysData_g0_t0.imec1.lf.cbin",
            "probe01/_spikeglx_ephysData_g0_t0.imec1.ap.ch",
            "probe01/_spikeglx_ephysData_g0_t0.imec1.wiring.json",
            "probe02/_spikeglx_ephysData_g0_t0.imec.ap.meta",  # 3A
            "probe02/_spikeglx_ephysData_g0_t0.imec.lf.meta",
            "probe02/_spikeglx_ephysData_g0_t0.imec.ap.bin",
            "probe02/_spikeglx_ephysData_g0_t0.imec.lf.bin",
            "_spikeglx_ephysData_g0_t0.nidq.sync.npy",
            "_spikeglx_ephysData_g0_t0.nidq.meta",
            "_spikeglx_ephysData_g0_t0.nidq.cbin",
            "_spikeglx_ephysData_g0_t0.nidq.ch",
            "_spikeglx_ephysData_g0_t0.wiring.json",
        ]
        for f in file_list:
            fpath = raw_ephys_data_path / Path(f)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.touch()

    return session_path


def populate_raw_spikeglx(session_path,
                          model='3B', legacy=False, user_label='my_run', n_probes=2):
    """
    Touch file tree to emulate files saved by SpikeGLX
    :param session_path: The raw ephys data path to place files
    :param model: Probe model file structure ('3A' or '3B')
    :param legacy: If true, the emulate older SpikeGLX version where all files are saved
    into a single folder
    :param user_label: User may input any name into SpikeGLX and filenames will include this
    :param n_probes: Number of probe datafiles to touch
    :return:

    Examples:
        populate_raw_spikeglx('3A_folder', model='3A', legacy=True, n_probes=1)
        3A_folder
            └───raw_ephys_folder
                    my_run_probe00_g0_t0.imec.ap.bin
                    my_run_probe00_g0_t0.imec.ap.meta
                    my_run_probe00_g0_t0.imec.lf.bin
                    my_run_probe00_g0_t0.imec.lf.meta

        populate_raw_spikeglx('3B_folder', model='3B', n_probes=3)
        3B_folder
            └───my_run_g0_t0
                    my_run_g0_t0.imec0.ap.bin
                    my_run_g0_t0.imec0.ap.meta
                    my_run_g0_t0.imec0.lf.bin
                    my_run_g0_t0.imec0.lf.meta
                    my_run_g0_t0.imec1.ap.bin
                    my_run_g0_t0.imec1.ap.meta
                    my_run_g0_t0.imec1.lf.bin
                    my_run_g0_t0.imec1.lf.meta
                    my_run_g0_t0.imec2.ap.bin
                    my_run_g0_t0.imec2.ap.meta
                    my_run_g0_t0.imec2.lf.bin
                    my_run_g0_t0.imec2.lf.meta
                    my_run_g0_t0.nidq.bin
                    my_run_g0_t0.nidq.meta

    See also: http://billkarsh.github.io/SpikeGLX/help/parsing/
    """
    for i in range(n_probes):
        label = (user_label + f'_probe{i:02}') if legacy and model == '3A' else user_label
        root = session_path.joinpath('raw_ephys_folder' if legacy else f'{user_label}_g0_t0')
        root.mkdir(exist_ok=True, parents=True)
        for ext in ('meta', 'bin'):
            for freq in ('lf', 'ap'):
                filename = f'{label}_g0_t0.imec{i if model == "3B" else ""}.{freq}.{ext}'
                root.joinpath(filename).touch()
            if model == '3B':
                root.joinpath(f'{label}_g0_t0.nidq.{ext}').touch()


def create_fake_raw_video_data_folder(session_path, populate=True, write_pars_stub=False):
    """
    Create the folder structure for a raw video session with three cameras.
    Creates a raw_video_data folder and optionally, touches some files and writes a experiment
    description stub to a _devices folder.

    Parameters
    ----------
    session_path : str, pathlib.Path
        The session path in which to create the folders.
    populate : bool
        If true, touch some raw video files.
    write_pars_stub : bool, str, dict
        If true, write an experiment description stub containing behaviour settings. If a string,
        the stub filename will contain this.  If a dict, the key is used as the filename; the value,
        the file contents.

    Example
    -------
    >>> create_fake_raw_video_data_folder(session_path, populate=False, write_pars_stub=False)
    >>> create_fake_raw_video_data_folder(session_path, write_pars_stub='hostname_19826354')
    """
    session_path = Path(session_path)
    raw_video_data_path = session_path / "raw_video_data"
    raw_video_data_path.mkdir(exist_ok=True, parents=True)
    if populate:
        file_list = [
            "_iblrig_leftCamera.raw.mp4",
            "_iblrig_rightCamera.raw.mp4",
            "_iblrig_bodyCamera.raw.mp4",
            "_iblrig_leftCamera.timestamps.ssv",
            "_iblrig_rightCamera.timestamps.ssv",
            "_iblrig_bodyCamera.timestamps.ssv",
            "_iblrig_leftCamera.GPIO.bin",
            "_iblrig_rightCamera.GPIO.bin",
            "_iblrig_bodyCamera.GPIO.bin",
            "_iblrig_leftCamera.frame_counter.bin",
            "_iblrig_rightCamera.frame_counter.bin",
            "_iblrig_bodyCamera.frame_counter.bin",
            "_iblrig_VideoCodeFiles.raw.zip",
        ]
        for f in file_list:
            fpath = raw_video_data_path / Path(f)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.touch()

    if write_pars_stub:
        if isinstance(write_pars_stub, dict):
            (name, data), = write_pars_stub.items()
        else:
            name = write_pars_stub if isinstance(write_pars_stub, str) else 'video'
            d = {'collection': 'raw_video_data', 'sync_label': 'frame2ttl'}
            data = {
                'devices': {'cameras': {k: d.copy() for k in ('body', 'left', 'right')}},
                'version': session_params.SPEC_VERSION
            }
        file_device = session_path.joinpath(f'_ibl_experiment.description_{name}.yaml')
        file_device.parent.mkdir(exist_ok=True)
        session_params.write_yaml(file_device, data)
    return raw_video_data_path


def create_fake_alf_folder_dlc_data(session_path, populate=True):
    session_path = Path(session_path)
    alf_path = session_path / "alf"
    alf_path.mkdir(exist_ok=True, parents=True)
    if populate:
        file_list = [
            "_ibl_leftCamera.dlc.pqt",
            "_ibl_rightCamera.dlc.pqt",
            "_ibl_bodyCamera.dlc.pqt",
            "_ibl_leftCamera.times.npy",
            "_ibl_rightCamera.times.npy",
            "_ibl_bodyCamera.times.npy",
            "_ibl_leftCamera.features.npy",
            "_ibl_rightCamera.features.npy",
        ]
        for f in file_list:
            fpath = alf_path / Path(f)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.touch()


def create_fake_raw_behavior_data_folder(
    session_path, populate=True, task="ephysCW", folder="raw_behavior_data", write_pars_stub=False
):
    """Create the folder structure for a raw behaviour session.

    Creates a raw_behavior_data folder and optionally, touches some files and writes a experiment
    description stub to a `_devices` folder.

    Parameters
    ----------
    session_path : pathlib.Path
        The session path in which to create the folders.
    populate : bool
        If true, touch some raw behaviour files.
    task : str
        The name of the task protocol, if 'ephys' or 'passive' extra files are touched.
    write_pars_stub : bool, str, dict
        If true, write an experiment description stub containing behaviour settings. If a string,
        the stub will be named as such.  If a dict, the key is used as the filename; the value,
        the file contents.

    Returns
    -------
    pathlib.Path
        The raw behaviour data path.
    """
    raw_behavior_data_path = session_path / folder
    raw_behavior_data_path.mkdir(exist_ok=True, parents=True)
    ephysCW = [
        "_iblrig_ambientSensorData.raw.jsonable",
        "_iblrig_encoderEvents.raw.ssv",
        "_iblrig_encoderPositions.raw.ssv",
        "_iblrig_encoderTrialInfo.raw.ssv",
        "_iblrig_micData.raw.wav",
        "_iblrig_stimPositionScreen.raw.csv",
        "_iblrig_syncSquareUpdate.raw.csv",
        "_iblrig_taskCodeFiles.raw.zip",
        "_iblrig_taskData.raw.jsonable",
        "_iblrig_taskSettings.raw.json",
        "online_plot.png",
    ]
    passiveCW = [
        "_iblrig_encoderEvents.raw.ssv",
        "_iblrig_encoderPositions.raw.ssv",
        "_iblrig_encoderTrialInfo.raw.ssv",
        "_iblrig_RFMapStim.raw.bin",
        "_iblrig_stimPositionScreen.raw.csv",
        "_iblrig_syncSquareUpdate.raw.csv",
        "_iblrig_taskCodeFiles.raw.zip",
        "_iblrig_taskSettings.raw.json",
    ]

    if populate:
        file_list = []
        if "ephys" in task:
            file_list = ephysCW
        elif "passive" in task:
            file_list = passiveCW
            (session_path / "passive_data_for_ephys.flag").touch()
        else:
            print(f"Not implemented: Task {task}")

        for f in file_list:
            fpath = raw_behavior_data_path / Path(f)
            fpath.parent.mkdir(parents=True, exist_ok=True)
            fpath.touch()

    if write_pars_stub:
        if isinstance(write_pars_stub, dict):
            (name, data), = write_pars_stub.items()
        else:
            name = write_pars_stub if isinstance(write_pars_stub, str) else 'behaviour'
            data = {
                'devices': {'microphone': {'microphone': {'collection': folder, 'sync_label': None}}},
                'procedures': ['Behavior training/tasks'],
                'projects': ['ibl_neuropixel_brainwide_01'],
                'tasks': [{task: {'collection': folder}}],
                'version': session_params.SPEC_VERSION
            }
            if 'ephys' in task:
                data['tasks'][0][task]['sync_label'] = 'frame2ttl'
            else:
                data['sync'] = {'bpod': {'collection': 'raw_behavior_data', 'extension': 'jsonable'}}
                data['tasks'][0][task]['sync_label'] = 'bpod'

        file_device = session_path.joinpath(f'_ibl_experiment.description_{name}.yaml')
        file_device.parent.mkdir(exist_ok=True)
        session_params.write_yaml(file_device, data)

    return raw_behavior_data_path


def populate_task_settings(fpath: Path, patch: dict):
    with fpath.open("w") as f:
        json.dump(patch, f, indent=1)


def create_fake_complete_ephys_session(
    root_data_path, lab="fakelab", mouse="fakemouse", date="1900-01-01", num="001", increment=True
):
    session_path = create_fake_session_folder(
        root_data_path, lab=lab, mouse=mouse, date=date, num=num, increment=increment
    )
    _mouse, _date, _num = session_path.parts[-3:]
    create_fake_raw_ephys_data_folder(session_path, populate=True)
    create_fake_raw_video_data_folder(session_path, populate=True)
    create_fake_raw_behavior_data_folder(session_path, populate=True, task="ephys")
    create_fake_raw_behavior_data_folder(
        session_path, populate=True, task="passive", folder="raw_passive_data"
    )
    fpath = Path(session_path) / "raw_passive_data" / "_iblrig_taskSettings.raw.json"
    passive_settings = {
        "CORRESPONDING_EPHYS_SESSION":
            f"C:\\some\\root\\folder\\Subjects\\{_mouse}\\{_date}\\{_num}"
    }
    populate_task_settings(fpath, passive_settings)
    if session_path.joinpath("passive_data_for_ephys.flag").exists():
        session_path.joinpath("passive_data_for_ephys.flag").unlink()

    return session_path


def create_fake_ephys_recording_bad_passive_transfer_sessions(
    root_data_path, lab="fakelab", mouse="fakemouse", date="1900-01-01", num="001", increment=True
):
    session_path = create_fake_session_folder(
        root_data_path, lab=lab, mouse=mouse, date=date, num=num, increment=increment
    )
    _mouse, _date, _num = session_path.parts[-3:]
    create_fake_raw_ephys_data_folder(session_path, populate=True)
    create_fake_raw_video_data_folder(session_path, populate=True)
    create_fake_raw_behavior_data_folder(session_path, populate=True, task="ephys")

    passive_session_path = create_fake_session_folder(
        root_data_path, lab=lab, mouse=mouse, date=date, num=num, increment=increment
    )
    create_fake_raw_behavior_data_folder(passive_session_path, populate=True, task="passive")
    fpath = Path(passive_session_path) / "raw_behavior_data" / "_iblrig_taskSettings.raw.json"
    passive_settings = {
        "CORRESPONDING_EPHYS_SESSION":
            f"C:\\some\\root\\folder\\Subjects\\{_mouse}\\{_date}\\{_num}"
    }
    populate_task_settings(fpath, passive_settings)

    return session_path, passive_session_path

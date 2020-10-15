#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Friday, October 9th 2020, 12:02:56 pm
import json
from pathlib import Path


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


def create_fake_raw_video_data_folder(session_path, populate=True):
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


def create_fake_raw_behavior_data_folder(
    session_path, populate=True, task="ephysCW", folder="raw_behavior_data"
):
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

    return session_path


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


if __name__ == "__main__":
    pass
    # root_data_path = "/home/nico/Projects/IBL/scratch/serverpc/_mnt_s0_Data"
    # # Create a fake ephys session
    # session_path = create_fake_session_folder(
    #     root_data_path, lab="fakelab", mouse="fakemouse", date="1900-01-01", num="001"
    # )
    # create_fake_raw_ephys_data_folder(session_path, populate=True)
    # create_fake_raw_video_data_folder(session_path, populate=True)
    # create_fake_raw_behavior_data_folder(session_path, populate=True, task="ephys")
    # # create a fake passive session that was not transferred
    # passive_session_path = create_fake_session_folder(
    #     root_data_path, lab="fakelab", mouse="fakemouse", date="1900-01-01", num="002"
    # )
    # create_fake_raw_behavior_data_folder(passive_session_path, populate=True, task="passive")

    # fpath = Path(passive_session_path) / "raw_behavior_data" / "_iblrig_taskSettings.raw.json"
    # passive_settings = {"CORRESPONDING_EPHYS_SESSION": str(session_path)}
    # populate_task_settings(fpath, passive_settings)

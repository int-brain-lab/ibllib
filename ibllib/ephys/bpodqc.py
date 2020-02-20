import os
from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

import ibllib.io.raw_data_loaders as raw
import ibllib.plots as plots
from ibllib.ephys.ephysqc import _qc_from_path
from ibllib.io.extractors.training_trials import (
    get_choice,
    get_goCueOnset_times,
    get_goCueTrigger_times,
    get_response_times,
    get_stimOn_times,
    get_stimOnTrigger_times,
    get_feedback_times,
    get_feedbackType,
    get_intervals,
    get_port_events,
)


def get_bpod_fronts(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})

    BNC1_fronts = np.array([[np.nan, np.nan]])
    BNC2_fronts = np.array([[np.nan, np.nan]])
    for tr in data:
        BNC1_fronts = np.append(BNC1_fronts, np.array([[x, 1] for x in tr["behavior_data"]["Events timestamps"].get('BNC1High', [np.nan])]), axis=0)
        BNC1_fronts = np.append(BNC1_fronts, np.array([[x, -1] for x in tr["behavior_data"]["Events timestamps"].get('BNC1Low', [np.nan])]), axis=0)
        BNC2_fronts = np.append(BNC2_fronts, np.array([[x, 1] for x in tr["behavior_data"]["Events timestamps"].get('BNC2High', [np.nan])]), axis=0)
        BNC2_fronts = np.append(BNC2_fronts, np.array([[x, -1] for x in tr["behavior_data"]["Events timestamps"].get('BNC2Low', [np.nan])]), axis=0)

    BNC1_fronts = BNC1_fronts[1:, :]
    BNC1_fronts = BNC1_fronts[BNC1_fronts[:, 0].argsort()]
    BNC2_fronts = BNC2_fronts[1:, :]
    BNC2_fronts = BNC2_fronts[BNC2_fronts[:, 0].argsort()]

    BNC1 = {"times": BNC1_fronts[:, 0], "polarities": BNC1_fronts[:, 1]}
    BNC2 = {"times": BNC2_fronts[:, 0], "polarities": BNC2_fronts[:, 1]}

    return (BNC1, BNC2)


# --------------------------------------------------------------------------- #
def get_itiIn_times(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})

    itiIn_times = np.array(
        [tr["behavior_data"]["States timestamps"]["exit_state"][0][0] for tr in data]
    )

    if raw.save_bool(save, "_ibl_trials.itiIn_times.npy"):
        lpath = os.path.join(session_path, "alf", "_ibl_trials.itiIn_times.npy")
        np.save(lpath, itiIn_times)

    return itiIn_times


def get_stimFreezeTrigger_times(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})

    freeze_reward = np.array(
        [
            True
            if np.all(~np.isnan(tr["behavior_data"]["States timestamps"]["freeze_reward"][0]))
            else False
            for tr in data
        ]
    )
    freeze_error = np.array(
        [
            True
            if np.all(~np.isnan(tr["behavior_data"]["States timestamps"]["freeze_error"][0]))
            else False
            for tr in data
        ]
    )
    no_go = np.array(
        [
            True
            if np.all(~np.isnan(tr["behavior_data"]["States timestamps"]["no_go"][0]))
            else False
            for tr in data
        ]
    )
    assert np.sum(freeze_error) + np.sum(freeze_reward) + np.sum(no_go) == len(data)

    stimFreezeTrigger = np.array([])
    for r, e, n, tr in zip(freeze_reward, freeze_error, no_go, data):
        if n:
            stimFreezeTrigger = np.append(stimFreezeTrigger, np.nan)
            continue
        state = "freeze_reward" if r else "freeze_error"
        stimFreezeTrigger = np.append(
            stimFreezeTrigger, tr["behavior_data"]["States timestamps"][state][0][0]
        )

    if raw.save_bool(save, "_ibl_trials.stimFreeze_times.npy"):
        lpath = os.path.join(session_path, "alf", "_ibl_trials.stimFreeze_times.npy")
        np.save(lpath, stimFreezeTrigger)

    return stimFreezeTrigger


def get_stimOffTrigger_times(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})

    stimOffTrigger_times = np.array(
        [tr["behavior_data"]["States timestamps"]["hide_stim"][0][0] for tr in data]
    )
    no_goTrigger_times = np.array(
        [tr["behavior_data"]["States timestamps"]["no_go"][0][0] for tr in data]
    )
    # Stim off triggers are either in their own state or in the no_go state if the mouse did not move
    assert all(~np.isnan(no_goTrigger_times) == np.isnan(stimOffTrigger_times))
    stimOffTrigger_times[~np.isnan(no_goTrigger_times)] = no_goTrigger_times[
        ~np.isnan(no_goTrigger_times)
    ]

    if raw.save_bool(save, "_ibl_trials.stimOffTrigger_times.npy"):
        lpath = Path(session_path).joinpath("alf", "_ibl_trials.stimOffTrigger_times.npy")
        np.save(lpath, stimOffTrigger_times)

    return stimOffTrigger_times


def get_stimOff_times_from_state(session_path, save=False, data=False, settings=False):
    """ Will return NaN is trigger state == 0.1 secs
    """

    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})

    hide_stim_state = np.array(
        [tr["behavior_data"]["States timestamps"]["hide_stim"][0] for tr in data]
    )
    stimOff_times = np.array([])
    for s in hide_stim_state:
        x = s[0] - s[1]
        if np.isnan(x) or np.abs(x) > 0.0999:
            stimOff_times = np.append(stimOff_times, np.nan)
        else:
            stimOff_times = np.append(stimOff_times, s[1])

    if raw.save_bool(save, "_ibl_trials.stimOff_times.npy"):
        lpath = Path(session_path).joinpath("alf", "_ibl_trials.stimOff_times.npy")
        np.save(lpath, stimOff_times)

    return stimOff_times


def get_stimOnOffFreeze_times_from_BNC1(session_path, save=False, data=False, settings=False):
    """Get stim onset offset and freeze using the FPGA specifications"""
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})

    choice = get_choice(session_path, data=data, settings=settings)
    f2TTL = [get_port_events(tr, name="BNC1") for tr in data]
    stimOn_times = np.array([])
    stimOff_times = np.array([])
    stimFreeze_times = np.array([])

    for tr in f2TTL:
        if tr and len(tr) >= 2:
            # 2nd order criteria:
            # stimOn -> Closest one to stimOnTrigger?
            # stimOff -> Closest one to stimOffTrigger?
            # stimFreeze -> Closest one to stimFreezeTrigger?
            stimOn_times = np.append(stimOn_times, tr[0])
            stimOff_times = np.append(stimOff_times, tr[-1])
            stimFreeze_times = np.append(stimFreeze_times, tr[-2])
        else:
            stimOn_times = np.append(stimOn_times, np.nan)
            stimOff_times = np.append(stimOff_times, np.nan)
            stimFreeze_times = np.append(stimFreeze_times, np.nan)

    # In no_go trials no stimFreeze happens jsut stim Off
    stimFreeze_times[choice == 0] = np.nan

    if raw.save_bool(save, "_ibl_trials.stimOn_times.npy"):
        lpath = Path(session_path).joinpath("alf", "_ibl_trials.stimOn_times.npy")
        np.save(lpath, stimOn_times)
    if raw.save_bool(save, "_ibl_trials.stimOff_times.npy"):
        lpath = Path(session_path).joinpath("alf", "_ibl_trials.stimOff_times.npy")
        np.save(lpath, stimOff_times)
    if raw.save_bool(save, "_ibl_trials.stimFreeze_times.npy"):
        lpath = Path(session_path).joinpath("alf", "_ibl_trials.stimFreeze_times.npy")
        np.save(lpath, stimFreeze_times)

    return stimOn_times, stimOff_times, stimFreeze_times


def get_bonsai_screen_data(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})
    path = Path(session_path).joinpath("raw_behavior_data", "_iblrig_stimPositionScreen.raw.csv")
    screen_data = pd.read_csv(path, sep=" ", header=None)

    return screen_data


def get_bonsai_sync_square_update_times(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})

    path = Path(session_path).joinpath("raw_behavior_data", "_iblrig_syncSquareUpdate.raw.csv")
    if path.exists():
        sync_square_update_times = pd.read_csv(path, sep=",", header=None)
        return sync_square_update_times

    return


def get_error_tone_in_trigger(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})

    error_tone_in_trigger = np.zeros(len(data)) * np.nan

    for i, tr in enumerate(data):
        nogo = tr["behavior_data"]["States timestamps"]["no_go"][0][0]
        error = tr["behavior_data"]["States timestamps"]["error"][0][0]
        if np.all(~np.isnan(nogo)):
            error_tone_in_trigger[i] = nogo
        elif np.all(~np.isnan(error)):
            error_tone_in_trigger[i] = error

    return error_tone_in_trigger


def _get_trimmed_data_from_pregenerated_files(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})
    num = settings.get("PRELOADED_SESSION_NUM", None)
    if num is None:
        num = settings.get("PREGENERATED_SESSION_NUM", None)
    if num is None:
        fn = settings.get("SESSION_LOADED_FILE_PATH", None)
        fn = PureWindowsPath(fn).name
        num = "".join([d for d in fn if d.isdigit()])
        if num == "":
            raise ValueError("Can't extract left probability behaviour.")
    # Load the pregenerated file
    sessions_folder = Path(raw.__file__).parent.joinpath("extractors", "ephys_sessions")
    fname = f"session_{num}_ephys_pcqs.npy"
    pcqsp = np.load(sessions_folder.joinpath(fname))
    pos = pcqsp[:, 0]
    con = pcqsp[:, 1]
    pos = pos[: len(data)]
    con = con[: len(data)]
    contrastRight = con.copy()
    contrastLeft = con.copy()
    contrastRight[pos < 0] = np.nan
    contrastLeft[pos > 0] = np.nan
    qui = pcqsp[:, 2]
    qui = qui[: len(data)]
    phase = pcqsp[:, 3]
    phase = phase[: len(data)]
    pLeft = pcqsp[:, 4]
    pLeft = pLeft[: len(data)]

    if raw.save_bool(save, "_ibl_trials.contrastLeft.npy"):
        lpath = os.path.join(session_path, "alf", "_ibl_trials.contrastLeft.npy")
        np.save(lpath, contrastLeft)

    if raw.save_bool(save, "_ibl_trials.contrastRight.npy"):
        rpath = os.path.join(session_path, "alf", "_ibl_trials.contrastRight.npy")
        np.save(rpath, contrastRight)

    if raw.save_bool(save, "_ibl_trials.probabilityLeft.npy"):
        lpath = Path(session_path).joinpath("alf", "_ibl_trials.probabilityLeft.npy")
        np.save(lpath, pLeft)

    return {"position": pos, "contrast": con, "quiescence": qui, "phase": phase, "prob_left": pLeft}


def load_bpod_data(session_path):
    # one.load(
    #     eid, dataset_types=["_iblrig_taskData.raw", "_iblrig_taskSettings.raw"], download_only=True
    # )
    # session_path = one.path_from_eid(eid)
    data = raw.load_data(session_path)
    settings = raw.load_settings(session_path)
    stimOn_times, stimOff_times, stimFreeze_times = get_stimOnOffFreeze_times_from_BNC1(
        session_path
    )

    out = {
        "position": None,
        "contrast": None,
        "quiescence": None,
        "phase": None,
        "prob_left": None,
        "choice": get_choice(session_path, save=False, data=data, settings=settings),
        "stimOnTrigger_times": get_stimOnTrigger_times(
            session_path, save=False, data=data, settings=settings
        ),
        "stimOffTrigger_times": get_stimOffTrigger_times(
            session_path, save=False, data=data, settings=settings
        ),
        "stimFreezeTrigger_times": get_stimFreezeTrigger_times(
            session_path, save=False, data=data, settings=settings
        ),
        "goCueTrigger_times": get_goCueTrigger_times(
            session_path, save=False, data=data, settings=settings
        ),
        "goCue_times": get_goCueOnset_times(session_path, save=False, data=data, settings=settings),
        "error_tone_in_trigger": get_error_tone_in_trigger(
            session_path, save=False, data=data, settings=settings
        ),
        "response_times": get_response_times(
            session_path, save=False, data=data, settings=settings
        ),
        "stimOn_times_BNC1": stimOn_times,
        "stimOff_times_BNC1": stimOff_times,
        "stimFreeze_times_BNC1": stimFreeze_times,
        "stimOn_times": get_stimOn_times(session_path, save=False, data=data, settings=settings),
        "stimOff_times": get_stimOff_times_from_state(
            session_path, save=False, data=data, settings=settings
        ),
        "feedback_times": get_feedback_times(
            session_path, save=False, data=data, settings=settings
        ),
        "feedbackType": get_feedbackType(session_path, save=False, data=data, settings=settings),
        "intervals": get_intervals(session_path, save=False, data=data, settings=settings),
        "itiIn_times": get_itiIn_times(session_path, save=False, data=data, settings=settings),
    }
    out.update(
        _get_trimmed_data_from_pregenerated_files(
            session_path, save=False, data=data, settings=settings
        )
    )
    # get valve_time and error_tone_in_time from feedback_times
    correct = np.sign(out["position"]) + np.sign(out["choice"]) == 0
    error_tone_in = out["feedback_times"].copy()
    valve_open = out["feedback_times"].copy()
    error_tone_in[correct] = np.nan
    valve_open[~correct] = np.nan
    out.update({"error_tone_in": error_tone_in, "valve_open": valve_open})
    return out


def get_bpodqc_frame(session_path):
    bpod = load_bpod_data(session_path)
    correct = np.sign(bpod["position"]) + np.sign(bpod["choice"]) == 0
    error_tone_in = bpod["feedback_times"].copy()
    valve_time = bpod["feedback_times"].copy()
    error_tone_in[correct] = np.nan
    valve_time[~correct] = np.nan

    GOCUE_STIMON_DELAY = 0.01  # -> 0.1
    FEEDBACK_STIMFREEZE_DELAY = 0.01  # -> 0.1
    VALVE_STIM_OFF_DELAY = 1
    VALVE_STIM_OFF_JITTER = 0.1
    ITI_IN_STIM_OFF_JITTER = 0.1
    ERROR_STIM_OFF_DELAY = 2
    ERROR_STIM_OFF_JITTER = 0.1  # -> 0.2
    RESPONSE_FEEDBACK_DELAY = 0.0005

    frame = {
        # Translate bpod data to match ephs_fpga_frame
        "ready_tone_in": bpod["goCue_times"],
        "goCueTrigger_times": bpod["goCueTrigger_times"],
        "goCue_times": bpod["goCue_times"],
        "error_tone_in_trigger": bpod["error_tone_in_trigger"],
        "error_tone_in": bpod["error_tone_in"],
        "valve_open": bpod["valve_open"],
        "feedback_times": bpod["feedback_times"],
        "stimOnTrigger_times": bpod["stimOnTrigger_times"],
        "stimOn_times": bpod["stimOn_times_BNC1"],
        "stimOffTrigger_times": bpod["stimOffTrigger_times"],
        "stimOff_times": bpod["stimOff_times_BNC1"],
        "stimFreezeTrigger_times": bpod["stimFreezeTrigger_times"],
        "stim_freeze": bpod["stimFreeze_times_BNC1"],
        "iti_in": bpod["itiIn_times"],
        "intervals_0": bpod["intervals"][:, 0],
        "intervals_1": bpod["intervals"][:, 1],
        "response_times": bpod["response_times"],
        # qc from here
        "n_feedback": np.int32(~np.isnan(bpod["valve_open"]) + ~np.isnan(bpod["error_tone_in"])),
        "stimOn_times_nan": ~np.isnan(bpod["stimOn_times_BNC1"]),
        "goCue_times_nan": ~np.isnan(bpod["goCue_times"]),
        "stimOn_times_before_goCue_times": bpod["goCue_times"] - bpod["stimOn_times_BNC1"] > 0,
        "stimOn_times_goCue_times_delay": (
            np.abs(bpod["goCue_times"] - bpod["stimOn_times_BNC1"]) <= GOCUE_STIMON_DELAY
        ),
        "stim_freeze_before_feedback": bpod["feedback_times"] - bpod["stimFreeze_times_BNC1"] > 0,
        "stim_freeze_feedback_delay": (
            np.abs(bpod["feedback_times"] - bpod["stimFreeze_times_BNC1"])
            <= FEEDBACK_STIMFREEZE_DELAY
        ),
        "stimOff_delay_valve": np.less(
            np.abs(bpod["stimOff_times"] - bpod["valve_open"] - VALVE_STIM_OFF_DELAY),
            VALVE_STIM_OFF_JITTER,
            out=np.ones(len(bpod["stimOn_times"]), dtype=np.bool),
            where=~np.isnan(bpod["valve_open"]),
        ),
        "iti_in_delay_stim_off": np.less(
            np.abs(bpod["stimOff_times_BNC1"] - bpod["itiIn_times"]),
            ITI_IN_STIM_OFF_JITTER,
            out=np.ones(len(bpod["stimOn_times"]), dtype=np.bool),
            where=~np.isnan(bpod["error_tone_in"]),
        ),  # FIXME: where= is wrong, no_go trials have a error tone but longer delay!
        "stimOff_delay_noise": np.less(
            np.abs(bpod["stimOff_times_BNC1"] - bpod["error_tone_in"] - ERROR_STIM_OFF_DELAY),
            ERROR_STIM_OFF_JITTER,
            out=np.ones(len(bpod["stimOn_times"]), dtype=np.bool),
            where=~np.isnan(bpod["error_tone_in"]),
        ),
        "response_times_nan": ~np.isnan(bpod["response_times"]),
        "response_times_increase": np.diff(np.append([0], bpod["response_times"])) > 0,
        "response_times_goCue_times_diff": bpod["response_times"] - bpod["goCue_times"] > 0,
        "response_before_feedback": bpod["feedback_times"] - bpod["response_times"] > 0,
        "response_feedback_delay": (
            bpod["feedback_times"] - bpod["response_times"] < RESPONSE_FEEDBACK_DELAY
        ),
    }
    #
    bpodqc_frame = pd.DataFrame.from_dict(frame)
    return bpodqc_frame


# --------------------------------------------------------------------------- #
def check_trigger_response(session_path):
    bpod = load_bpod_data(session_path)
    # get diff from triggers to detected events
    goCue_diff = bpod["goCueTrigger_times"] - bpod["goCue_times"]
    stimOn_diff = bpod["stimOnTrigger_times"] - bpod["stimOn_times_BNC1"]
    stimOff_diff = bpod["stimOffTrigger_times"] - bpod["stimOff_times_BNC1"]
    stimFreeze_diff = bpod["stimFreezeTrigger_times"] - bpod["stimFreeze_times_BNC1"]


def check_response_feedback(session_path):
    bpod = load_bpod_data(session_path)

    resp_feedback_diff = bpod["response_times"] - bpod["feedback_times"]


def check_iti_stimOffTrig(session_path):
    bpod = load_bpod_data(session_path)

    iti_stimOff_diff = bpod["stimOffTrigger_times"] - bpod["itiIn_times"]
    # 0.1 means that stimOff lasted more than 0.1 to go off.or BNC1 was not detected
    # 2 sec means it was a no go trial and the trig was at the beginning of the no_go state
    # that lasts for 2 sec after which itiState enters


def check_feedback_stim_off_delay(session_path):
    bpod = load_bpod_data(session_path)

    valve_stim_off_diff = bpod["feedback_times"] - bpod["stimOffTrigger_times"]


# --------------------------------------------------------------------------- #
def count_qc_failures(session_path):
    fpgaqc_frame = _qc_from_path(session_path, display=True)
    bpodqc_frame = get_bpodqc_frame(session_path)
    qc_fields = [
        "n_feedback",
        "stimOn_times_nan",
        "goCue_times_nan",
        "stimOn_times_before_goCue_times",
        "stimOn_times_goCue_times_delay",
        "stim_freeze_before_feedback",
        "stim_freeze_feedback_delay",
        "stimOff_delay_valve",
        "iti_in_delay_stim_off",
        "stimOff_delay_noise",
        "response_times_nan",
        "response_times_increase",
        "response_times_goCue_times_diff",
    ]
    for k in qc_fields:
        print("FPGA nFailed", k, ":", sum(fpgaqc_frame[k] == False), len(fpgaqc_frame[k]))
        print("BPOD nFailed", k, ":", sum(bpodqc_frame[k] == False), len(bpodqc_frame[k]))


def plot_bpod_session(session_path):
    BNC1, BNC2 = get_bpod_fronts(session_path)
    plt.ion()
    f, ax = plt.subplots()
    width = 0.5
    ymax = 5
    plots.squares(BNC1['times'], BNC1['polarities'] * 0.4 + 1, ax=ax, c='k')
    plots.squares(BNC2['times'], BNC2['polarities'] * 0.4 + 2, ax=ax, c='k')
    plots.vertical_lines(bpodqc_frame['ready_tone_in'], ymin=0, ymax=ymax,
                         ax=ax, label='ready_tone_in', color='b', linewidth=width)
    plots.vertical_lines(bpodqc_frame['intervals_0'], ymin=0, ymax=ymax,
                         ax=ax, label='start_trial', color='m', linewidth=width)
    plots.vertical_lines(bpodqc_frame['error_tone_in_trigger'], ymin=0, ymax=ymax,
                         ax=ax, label='error_tone_in_trigger', color='r', alpha=0.5, linewidth=width)
    plots.vertical_lines(bpodqc_frame['error_tone_in'], ymin=0, ymax=ymax,
                         ax=ax, label='error_tone_in', color='r', linewidth=width)
    plots.vertical_lines(bpodqc_frame['valve_open'], ymin=0, ymax=ymax,
                         ax=ax, label='valve_open', color='g', linewidth=width)
    plots.vertical_lines(bpodqc_frame['stimFreezeTrigger_times'], ymin=0, ymax=ymax,
                         ax=ax, label='stimFreezeTrigger_times', color='k', alpha=0.5, linewidth=width)
    plots.vertical_lines(bpodqc_frame['stim_freeze'], ymin=0, ymax=ymax,
                         ax=ax, label='stim_freeze', color='y', linewidth=width)
    plots.vertical_lines(bpodqc_frame['stimOffTrigger_times'], ymin=0, ymax=ymax,
                         ax=ax, label='stimOffTrigger_times', color='c', alpha=0.5, linewidth=width)
    plots.vertical_lines(bpodqc_frame['stimOff_times'], ymin=0, ymax=ymax,
                         ax=ax, label='stimOff_times', color='c', linewidth=width)
    plots.vertical_lines(bpodqc_frame['stimOnTrigger_times'], ymin=0, ymax=ymax,
                         ax=ax, label='stimOnTrigger_times', color='tab:orange', alpha=0.5, linewidth=width)
    plots.vertical_lines(bpodqc_frame['stimOn_times'], ymin=0, ymax=ymax,
                         ax=ax, label='stimOn_times', color='tab:orange', linewidth=width)
    ax.legend()
    ax.set_yticklabels(['', 'f2ttl', 'audio', ''])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim([0, 3])
# Make decorator for sesison_path based QC to accept eid's
# Check input if valid eid
# Download relevant datasets
# Get the path and feed  it to the func [sess_path = one.path_from_eid(eid)]


if __name__ == "__main__":
    from ibllib.ephys.bpodqc import *
    from ibllib.ephys.ephysqc import _qc_from_path
    import matplotlib.pyplot as plt

    # Guido's 3B
    gsession_path = "/home/nico/Projects/IBL/github/iblapps/scratch/TestSubjects/_iblrig_test_mouse/2020-02-11/001"
    # Alex's 3A
    asession_path = "/home/nico/Projects/IBL/scratch/TestSubjects/_iblrig_test_mouse/2020-02-18/006"

    session_path = gsession_path
    bpod = load_bpod_data(session_path)
    fpgaqc_frame = _qc_from_path(session_path)
    bpodqc_frame = get_bpodqc_frame(session_path)

    bla = [(k, all(fpgaqc_frame[k] == bpodqc_frame[k])) for k in fpgaqc_frame if k in bpodqc_frame]

    count_qc_failures(session_path)
    plot_bpod_session(session_path)

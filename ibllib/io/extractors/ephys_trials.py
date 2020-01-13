import os
import tempfile
from pathlib import Path

import numpy as np

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.biased_trials import (
    get_choice,
    get_contrastLR,
    get_feedbackType,
    get_goCueOnset_times,
    get_goCueTrigger_times,
    get_intervals,
    get_iti_duration,
    get_response_times,
    get_rewardVolume,
)


def get_probabilityLeft(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None:
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    elif settings["IBLRIG_VERSION_TAG"] == "":
        settings.update({"IBLRIG_VERSION_TAG": "100.0.0"})
    # LEN_BLOCKS is None up to v6.3.1 (also POSITIONS, CONTRASTS, et al.)
    if settings["LEN_BLOCKS"] is None:
        # Get from iblrig repo
        if "PRELOADED_SESSION_NUM" in settings:
            num = settings["PRELOADED_SESSION_NUM"]
        elif "PREGENERATED_SESSION_NUM" in settings:
            num = settings["PREGENERATED_SESSION_NUM"]

        master_branch = "https://raw.githubusercontent.com/int-brain-lab/iblrig/master/"
        sessions_folder = "tasks/_iblrig_tasks_ephysChoiceWorld/sessions/"
        fname0 = f"session_{num}_ephys_len_blocks.npy"
        fname1 = f"session_{num}_ephys_pcqs.npy"
        tmp = tempfile.mkdtemp()
        os.system(f"wget {master_branch}{sessions_folder}{fname0} --output-document={tmp}/{fname0}")
        os.system(f"wget {master_branch}{sessions_folder}{fname1} --output-document={tmp}/{fname1}")
        # Load the files
        len_blocks = np.load(f"{tmp}/{fname0}").tolist()
        pcqs = np.load(f"{tmp}/{fname1}")
        positions = [x[0] for x in pcqs]
        # Patch the settings file
        settings["LEN_BLOCKS"] = len_blocks
        settings["POSITIONS"] = positions
        # Cleanup
        (Path(tmp) / fname0).unlink()
        (Path(tmp) / fname1).unlink()
        Path(tmp).rmdir()
    # Continue
    pLeft = []
    prev = 0
    for bl, nt in zip(np.cumsum(settings["LEN_BLOCKS"]), settings["LEN_BLOCKS"]):
        prob = np.sum([1 for x in np.sign(settings["POSITIONS"][prev:bl]) if x < 0]) / nt
        pLeft.extend([prob] * nt)
        prev = bl
    # Trim to actual number of trials
    pLeft = pLeft[: len(data)]

    if raw.save_bool(save, "_ibl_trials.probabilityLeft.npy"):
        lpath = Path(session_path).joinpath("alf", "_ibl_trials.probabilityLeft.npy")
        np.save(lpath, pLeft)
    return pLeft


def extract_all(session_path, save=False, data=False, output_path=None):
    """
    Extract all behaviour data from Bpod whithin the specified folder.
    The timing information from FPGA is extracted in
    :func:`~ibllib.io.extractors.ephys_fpga`

    :param session_path: folder containing sessions
    :type session_path: str or pathlib.Path
    :param save: bool
    :param data: raw Bpod data dictionary
    :return: dictionary of trial related vectors (one row per trial)
    """
    if not data:
        data = raw.load_data(session_path)
    feedbackType = get_feedbackType(session_path, save=save, data=data)
    contrastLeft, contrastRight = get_contrastLR(session_path, save=save, data=data)
    probabilityLeft = get_probabilityLeft(session_path, save=save, data=data)
    choice = get_choice(session_path, save=save, data=data)
    rewardVolume = get_rewardVolume(session_path, save=save, data=data)
    iti_dur = get_iti_duration(session_path, save=save, data=data)
    # all files containing times need to be saved with a specific bpod tag later
    intervals = get_intervals(session_path, save=False, data=data)
    response_times = get_response_times(session_path, save=False, data=data)
    go_cue_trig_times = get_goCueTrigger_times(session_path, save=False, data=data)

    out = {
        "feedbackType": feedbackType,
        "contrastLeft": contrastLeft,
        "contrastRight": contrastRight,
        "probabilityLeft": probabilityLeft,
        "session_path": session_path,
        "choice": choice,
        "rewardVolume": rewardVolume,
        "iti_dur": iti_dur,
        "goCueTrigger_times": go_cue_trig_times,
        "intervals": intervals,
        "response_times": response_times,
    }
    if save:
        output_path = Path(output_path or Path(session_path).joinpath("alf"))
        np.save(output_path.joinpath("_ibl_trials.intervals_bpod.npy"), intervals)
        np.save(output_path.joinpath("_ibl_trials.response_times_bpod.npy"), response_times)
        np.save(output_path.joinpath("_ibl_trials.goCueTrigger_times_bpod.npy"), go_cue_trig_times)
    return out


if __name__ == "__main__":
    session_path = str(Path().cwd() / "../../../tests/ibllib/extractors/data/session_ephys")
    settings = raw.load_settings(session_path)
    settings.update({"LEN_BLOCKS": None})
    data = raw.load_data(session_path)
    pLeft = get_probabilityLeft(session_path, data=data, settings=settings)
    print(".")

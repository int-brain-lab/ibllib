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
    get_rewardVolume,
)


def get_probabilityLeft(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings["IBLRIG_VERSION_TAG"] == "":
        settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
    pLeft = []
    if settings["LEN_BLOCKS"] is None:
        # Get from iblrig repo
        pass
    else:
        prev = 0
        for bl, nt in zip(np.cumsum(settings["LEN_BLOCKS"]), settings["LEN_BLOCKS"]):
            prob = (np.sum([1 for x in np.sign(settings["POSITIONS"][prev:bl]) if x < 0]) / nt)
            pLeft.extend([prob] * nt)
            prev = bl
        # Trim to actual number of trials
        pLeft = pLeft[: len(data)]

    if raw.save_bool(save, "_ibl_trials.probabilityLeft.npy"):
        lpath = Path(session_path).joinpath("alf", "_ibl_trials.probabilityLeft.npy")
        np.save(lpath, pLeft)
    return pLeft


def extract_all(session_path, save=False, data=False):
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
    session_path = "/home/nico/Projects/IBL/github/iblrig_data/Subjects/_iblrig_fake_subject/2020-01-08/013"
    settings = raw.load_settings(session_path)
    pLeft = []
    prev = 0
    for bl, nt in zip(np.cumsum(settings["LEN_BLOCKS"]), settings["LEN_BLOCKS"]):
        prob = (
            np.sum([1 for x in np.sign(settings["POSITIONS"][prev:bl]) if x < 0]) / nt
        )
        pLeft.extend([prob] * nt)
        prev = bl

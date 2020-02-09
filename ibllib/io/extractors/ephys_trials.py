from pathlib import Path, PureWindowsPath

import numpy as np

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.biased_trials import (
    get_choice,
    get_contrastLR,
    get_feedbackType,
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
    num = settings.get("PRELOADED_SESSION_NUM", None)
    if num is None:
        num = settings.get("PREGENERATED_SESSION_NUM", None)
    if num is None:
        fn = settings.get('SESSION_LOADED_FILE_PATH', None)
        fn = PureWindowsPath(fn).name
        num = ''.join([d for d in fn if d.isdigit()])
        if num == '':
            raise ValueError("Can't extract left probability behaviour.")
    # Load the pregenerated file
    sessions_folder = Path(raw.__file__).parent.joinpath('extractors', 'ephys_sessions')
    fname = f"session_{num}_ephys_pcqs.npy"
    pcqsp = np.load(sessions_folder.joinpath(fname))
    pLeft = pcqsp[:, 4]
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

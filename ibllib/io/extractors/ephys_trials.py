from pathlib import Path

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.biased_trials import (
    get_feedbackType, get_probabilityLeft,
    get_choice, get_rewardVolume, get_iti_duration, get_contrastLR,
    get_goCueTrigger_times, get_goCueOnset_times, get_intervals)


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
    contrastLeft, contrastRight = get_contrastLR(
        session_path, save=save, data=data)
    probabilityLeft = get_probabilityLeft(session_path, save=save, data=data)
    choice = get_choice(session_path, save=save, data=data)
    rewardVolume = get_rewardVolume(session_path, save=save, data=data)
    iti_dur = get_iti_duration(session_path, save=save, data=data)
    go_cue_trig_times = get_goCueTrigger_times(session_path, save=save, data=data)
    go_cue_times = get_goCueOnset_times(session_path, save=save, data=data)
    intervals = get_intervals(session_path, save=save, data=data)
    out = {'feedbackType': feedbackType,
           'contrastLeft': contrastLeft,
           'contrastRight': contrastRight,
           'probabilityLeft': probabilityLeft,
           'session_path': session_path,
           'choice': choice,
           'rewardVolume': rewardVolume,
           'iti_dur': iti_dur,
           'goCue_times': go_cue_times,
           'goCueTrigger_times': go_cue_trig_times,
           'intervals': intervals}
    if save:
        file_intervals = Path(session_path) / 'alf' / '_ibl_trials.intervals.npy'
        file_intervals.rename(Path(session_path) / 'alf' / '_ibl_trials.intervalsBpod.npy')

    return out

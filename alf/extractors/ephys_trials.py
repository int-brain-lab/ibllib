import ibllib.io.raw_data_loaders as raw
from alf.extractors.biased_trials import (
    get_feedbackType, get_probaLR,
    get_choice, get_rewardVolume, get_iti_duration, get_contrastLR,
    get_goCueTrigger_times, get_goCueOnset_times)


def extract_all(session_path, save=False, data=False):
    if not data:
        data = raw.load_data(session_path)
    feedbackType = get_feedbackType(session_path, save=save, data=data)
    contrastLeft, contrastRight = get_contrastLR(
        session_path, save=save, data=data)
    probabilityLeft, _ = get_probaLR(session_path, save=save, data=data)
    choice = get_choice(session_path, save=save, data=data)
    rewardVolume = get_rewardVolume(session_path, save=save, data=data)
    iti_dur = get_iti_duration(session_path, save=save, data=data)
    go_cue_trig_times = get_goCueTrigger_times(session_path, save=save, data=data)
    go_cue_times = get_goCueOnset_times(session_path, save=save, data=data)
    out = {'feedbackType': feedbackType,
           'contrastLeft': contrastLeft,
           'contrastRight': contrastRight,
           'probabilityLeft': probabilityLeft,
           'session_path': session_path,
           'choice': choice,
           'rewardVolume': rewardVolume,
           'iti_dur': iti_dur,
           'goCue_times': go_cue_times,
           'goCueTrigger_times': go_cue_trig_times}
    return out

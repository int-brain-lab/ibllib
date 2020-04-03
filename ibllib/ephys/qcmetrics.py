import json
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ibllib.ephys.bpodqc as bpodqc
from ibllib.ephys.oneutils import search_lab_ephys_sessions, _to_eid, random_ephys_session
from oneibl.one import ONE
from alf.io import is_details_dict

plt.ion()


one = ONE()


def _load_df_from_details(details=None, func=None):
    """
    Applies a session level loader_func(eid) from session details dict from Alyx
    """
    if details is None or func is None:
        print("One or more required inputs are None.")
        return
    if is_details_dict(details):
        details = [details]
    data = []
    labels = []
    for i, det in enumerate(details):
        eid = _to_eid(det)
        data.append(func(eid))
        labels.append(det["lab"] + str(i))

    df = pd.DataFrame(data).transpose()
    df.columns = labels

    return df


def boxplots_from_df(
    df, ax=None, describe=False, title="", xlabel="Seconds (s)", xscale="symlog",
):
    if ax is None:
        f, ax = plt.subplots()

    if describe:
        desc = df.describe()
        print(json.dumps(json.loads(desc.to_json()), indent=1))
    # Plot
    p = sns.boxplot(data=df, ax=ax, orient="h")
    p.set_title(title)
    p.set_xlabel(xlabel)
    p.set(xscale=xscale)


def boxplot_metrics(eid, qcmetrics_frame=None):
    if qcmetrics_frame is None:
        qcmetrics_frame = get_qcmetrics_frame(eid)
    df = pd.DataFrame.from_dict({k: qcmetrics_frame[k] for k in qcmetrics_frame if "delays" in k})
    boxplots_from_df(df, describe=True)


class SessionBehaviorMetrics(object):
    def __init__(self, eid, from_one=False):
        self.eid = eid
        self.data = bpodqc.load_bpod_data(eid, fpga_time=from_one)
        self.from_one = from_one

    def compute_metrics(self):
        return


def bpod_data_loader(func):
    """ Checks if data is None loads eid data in case
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not kwargs or kwargs["data"] is None:
            kwargs["data"] = bpodqc.load_bpod_data(args[0])
        return func(*args, **kwargs)

    return wrapper


@bpod_data_loader
def get_qcmetrics_frame(eid, data=None, pass_crit=False):
    qcmetrics_frame = {
        "goCue_delays": load_goCue_delays(eid, data=data),  # (Point 25)
        "errorCue_delays": load_errorCue_delays(eid, data=data),  # (Point 26)
        "stimOn_delays": load_stimOn_delays(eid, data=data),  # (Point 27)
        "stimOff_delays": load_stimOff_delays(eid, data=data),  # (Point 28)
        "stimFreeze_delays": load_stimFreeze_delays(eid, data=data),  # (Point 29)
        "stimOn_goCue_delays": load_stimon_gocue_delays(eid, data=data),  # (Point 1)
        "response_feedback_delays": load_response_feddback_delays(eid, data=data),  # (Point 2)
        "response_stimFreeze_delays": load_response_stimFreeze_delays(eid, data=data),  # (Point 3)
        "stimOff_itiIn_delays": load_stimOff_itiIn_delays(eid, data=data),  # (Point 4)
        "wheel_freeze_during_quiescence": load_wheel_freeze_during_quiescence(
            eid, data=data
        ),  # (Point 5)
        "wheel_move_before_feedback": load_wheel_move_before_feedback(eid, data=data),  # (Point 6)
        "stimulus_move_before_goCue": load_stimulus_move_before_goCue(eid, data=data),  # (Point 7)
        "positive_feedback_stimOff_delays": load_positive_feedback_stimOff_delays(
            eid, data=data
        ),  # (Point 8)
        "negative_feedback_stimOff_delays": load_negative_feedback_stimOff_delays(
            eid, data=data
        ),  # (Point 9)
        "valve_pre_trial": load_valve_pre_trial(eid, data=data),  # (Point 11)
        "audio_pre_trial": load_audio_pre_trial(eid, data=data),  # (Point 12)
        "trial_event_sequence_error": load_trial_event_sequence_error(
            eid, data=data
        ),  # (Point 13)
        "trial_event_sequence_correct": load_trial_event_sequence_correct(
            eid, data=data
        ),  # (Point 14)
        "trial_length": load_trial_length(eid, data=data),  # (Point 15)
    }
    return qcmetrics_frame


@bpod_data_loader
def get_qccriteria_frame(eid, data=None, pass_crit=True):
    qccriteria_frame = {
        "nDatasetTypes": load_nDatasetTypes(eid, data=data, pass_crit=True),  # (Point 17)
        "intervals": load_intervals(eid, data=data, pass_crit=True),  # (Point 18)
        "stimOnTrigger_times": load_stimOnTrigger_times(
            eid, data=data, pass_crit=True
        ),  # (Point 19)
        "stimOn_times": load_stimOn_times(eid, data=data, pass_crit=True),  # (Point 20)
        "goCueTrigger_times": load_goCueTrigger_times(
            eid, data=data, pass_crit=True
        ),  # (Point 21)
        "goCue_times": load_goCue_times(eid, data=data, pass_crit=True),  # (Point 22)
        "response_times": load_response_times(eid, data=data, pass_crit=True),  # (Point 23)
        "feedback_times": load_feedback_times(eid, data=data, pass_crit=True),  # (Point 24)
        "goCue_delays": load_goCue_delays(eid, data=data, pass_crit=True),  # (Point 25)
        "errorCue_delays": load_errorCue_delays(eid, data=data, pass_crit=True),  # (Point 26)
        "stimOn_delays": load_stimOn_delays(eid, data=data, pass_crit=True),  # (Point 27)
        "stimOff_delays": load_stimOff_delays(eid, data=data, pass_crit=True),  # (Point 28)
        "stimFreeze_delays": load_stimFreeze_delays(eid, data=data, pass_crit=True),  # (Point 29)
        "stimOn_goCue_delays": load_stimon_gocue_delays(
            eid, data=data, pass_crit=True
        ),  # (Point 1)
        "response_feedback_delays": load_response_feddback_delays(
            eid, data=data, pass_crit=True
        ),  # (Point 2)
        "response_stimFreeze_delays": load_response_stimFreeze_delays(
            eid, data=data, pass_crit=True
        ),  # (Point 3)
        "stimOff_itiIn_delays": load_stimOff_itiIn_delays(
            eid, data=data, pass_crit=True
        ),  # (Point 4)
        "wheel_freeze_during_quiescence": load_wheel_freeze_during_quiescence(
            eid, data=data, pass_crit=True
        ),  # (Point 5)
        "wheel_move_before_feedback": load_wheel_move_before_feedback(
            eid, data=data, pass_crit=True
        ),  # (Point 6)
        "stimulus_move_before_goCue": load_stimulus_move_before_goCue(
            eid, data=data, pass_crit=True
        ),  # (Point 7)
        "positive_feedback_stimOff_delays": load_positive_feedback_stimOff_delays(
            eid, data=data, pass_crit=True
        ),  # (Point 8)
        "negative_feedback_stimOff_delays": load_negative_feedback_stimOff_delays(
            eid, data=data, pass_crit=True
        ),  # (Point 9)
        "valve_pre_trial": load_valve_pre_trial(eid, data=data, pass_crit=True),  # (Point 11)
        "audio_pre_trial": load_audio_pre_trial(eid, data=data, pass_crit=True),  # (Point 12)
        "trial_event_sequence_error": load_trial_event_sequence_error(
            eid, data=data, pass_crit=True
        ),  # (Point 13)
        "trial_event_sequence_correct": load_trial_event_sequence_correct(
            eid, data=data, pass_crit=True
        ),  # (Point 14)
        "trial_length": load_trial_length(eid, data=data, pass_crit=True),  # (Point 15)
    }
    return qccriteria_frame


# ---------------------------------------------------------------------------- #
@bpod_data_loader
def load_stimon_gocue_delays(eid, data=None, pass_crit=False):
    """ 1. StimOn and GoCue and should be within a 10 ms of each other on 99% of trials
    Variable name: stimOn_goCue_delays
    Metric: goCue_times - stimOn_times (from ONE)
    Criterion: (M<10 ms for 99%) of trials AND (M > 0 ms for 99% of trials)
    """
    metric = data["goCue_times"] - data["stimOn_times"]
    criteria = (metric < 0.01) & (metric > 0)
    return np.mean(criteria) if pass_crit else metric


@bpod_data_loader
def load_response_feddback_delays(eid, data=None, pass_crit=False):
    """ 2. response_time and feedback_time
    Variable name: response_feedback_delays
    Metric: Feedback_time - response_time
    Criterion: (M <10 ms for 99% of trials) AND ( M > 0 ms for 100% of trials)
    _one_load_delays_between_events(
        eid, "trials.response_times", "trials.feedback_times"
    )"""
    metric = data["feedback_times"] - data["response_times"]
    criteria = (metric < 0.01) & (metric > 0)
    return np.mean(criteria) if pass_crit else metric


@bpod_data_loader
def load_response_stimFreeze_delays(eid, data=None, pass_crit=False):
    """ 3. Stim freeze and response time
    Variable name: response_stimFreeze_delays
    Metric: stim_freeze - response_time
    Criterion: (M<100 ms for 99% of trials) AND (M > 0 ms for 100% of trials)
    response = one.load(eid, dataset_types=["trials.response_times"])[0]
    _, _, stimFreeze = bpodqc.get_stimOnOffFreeze_times_from_BNC1(eid)
    bpod2fpga = bpodqc.get_bpod2fpga_times_func(eid)
    stimFreeze = bpod2fpga(stimFreeze)
    if len(response) != len(stimFreeze):
        session_path = one.path_from_eid(eid)
        response = bpodqc.get_response_times(session_path, save=False)
    assert len(response) == len(stimFreeze)
    """
    metric = data["stimFreeze_times"] - data["response_times"]
    criteria = (metric < 0.1) & (metric > 0)
    # Remove no_go trials (stimFreeze triggered differently in no_go trials)
    nonogo_criteria = criteria[~data["choice"] == 0]
    return np.mean(nonogo_criteria) if pass_crit else metric


@bpod_data_loader
def load_stimOff_itiIn_delays(eid, data=None, pass_crit=False):
    """ 4. Start of iti_in should be within a very small tolerance of the stim off
    Variable name: stimOff_itiIn_delays
    Metric: iti_in - stim_off
    Criterion: (M<10 ms for 99% of trials) AND (M > 0 ms for 99% of trials)
    itiIn = bpodqc.get_itiIn_times(eid, save=False)
    _, stimOff, _ = bpodqc.get_stimOnOffFreeze_times_from_BNC1(eid, save=False)
    if len(itiIn) != len(stimOff):
        print(f"Length mismatch iniIn and stimOff: {len(itiIn)}, {len(stimOff)}")
    """
    metric = data["itiIn_times"] - data["stimOff_times"]
    criteria = (metric < 0.01) & (metric >= 0)
    # Remove no_go trials (stimOff triggered differently in no_go trials)
    nonogo_criteria = criteria[~data["choice"] == 0]
    return np.mean(nonogo_criteria) if pass_crit else metric


@bpod_data_loader
def load_wheel_freeze_during_quiescence(eid, data=None, pass_crit=False):
    """ 5. Wheel should not move more than 2 ticks each direction for at least 0.2 + 0.2-0.6
    amount of time (quiescent period; exact value in bpod['quiescence']) before go cue
    Variable name: wheel_freeze_during_quiescence
    Metric: max(abs( w(t) - w_start )) over interval
    interval = [quiescent_end_time-quiescent_duration+0.02, quiescent_end_time]
    Criterion: <2 ticks for 99% of trials
    """
    return


@bpod_data_loader
def load_wheel_move_before_feedback(eid, data=None, pass_crit=False):
    """ 6. Wheel should move before feedback
    Variable name: wheel_move_before_feedback
    Metric: max(abs( w(t) - w_start )) over interval [gocue_time, feedback_time]
    Criterion: >2 ticks for 99% of non-NoGo trials
    """
    return


@bpod_data_loader
def load_stimulus_move_before_goCue(eid, data=None, pass_crit=False):
    """ 7. No stimulus movements between trialstart_time and gocue_time-20 ms
    Variable name: stimulus_move_before_goCue
    Metric: count of any stimulus change events between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    ---
    goCue_times = one.load(eid, dataset_types="trials.goCue_times")
    bpod2fpga = bpodqc.get_bpod2fpga_times_func(eid)
    BNC1_times = bpod2fpga(BNC1['times'])
    """
    BNC1, _ = bpodqc.get_bpod_fronts(eid)
    s = BNC1["times"]
    metric = np.array([])
    for i, c in zip(data["intervals_0"], data["goCue_times"]):
        metric = np.append(metric, np.count_nonzero(s[s > i] < (c - 0.02)))

    criteria = metric == 0
    nonogo_criteria = criteria[~data["choice"] == 0]
    return np.mean(nonogo_criteria) if pass_crit else metric


@bpod_data_loader
def load_positive_feedback_stimOff_delays(eid, data=None, pass_crit=False):
    """ 8. Delay between valve and stim off should be 1s
    Variable name: positive_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 1s)
    Criterion: <100 ms on 99% of correct trials
    """
    metric = np.abs(data["stimOff_times"] - data["feedback_times"] - 1)
    criteria = metric[data["correct"]] < 0.1
    return np.mean(criteria) if pass_crit else metric


@bpod_data_loader
def load_negative_feedback_stimOff_delays(eid, data=None, pass_crit=False):
    """ 9.Delay between noise and stim off should be 2 second
    Variable name: negative_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 2s)
    Criterion: <100 ms on 99% of incorrect trials
    """
    metric = np.abs(data["stimOff_times"] - data["errorCue_times"] - 2)
    criteria = metric[data["outcome"] == -1] < 0.1
    return np.mean(criteria) if pass_crit else metric


# @bpod_data_loader
# def load_0(eid, data=None, pass_crit=False):
#     """ 10. Number of Bonsai command to change screen should match
#     Number of state change of frame2ttl
#     Variable name: syncSquare
#     Metric: (count of bonsai screen updates) - (count of frame2ttl)
#     Criterion: 0 on 99% of trials
#     """
#     pass


@bpod_data_loader
def load_valve_pre_trial(eid, data=None, pass_crit=False):
    """ 11. No valve outputs between trialstart_time and gocue_time-20 ms
    Variable name: valve_pre_trial
    Metric: count of valve events between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    metric = ~(data["valveOpen_times"] < data["goCue_times"])
    criteria = metric
    return np.mean(criteria) if pass_crit else metric


@bpod_data_loader
def load_audio_pre_trial(eid, data=None, pass_crit=False):
    """ 12. No audio outputs between trialstart_time and gocue_time-20 ms
    Variable name: audio_pre_trial
    Metric: count of audio events between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    _, BNC2 = bpodqc.get_bpod_fronts(eid)
    s = BNC2["times"]
    metric = np.array([], dtype=np.bool)
    for i, c in zip(data["intervals_0"], data["goCue_times"]):
        metric = np.append(metric, ~np.any(s[s > i] < (c - 0.02)))
    criteria = metric
    return np.mean(criteria) if pass_crit else metric


# Sequence of events:
@bpod_data_loader
def load_trial_event_sequence_error(eid, data=None, pass_crit=False):
    """ 13. on incorrect / miss trials : 2 audio events, 2 Bpod events (trial start, ITI)
    Variable name: trial_event_sequence_error
    Metric: Bpod (trial start) > audio (go cue) > audio (wrong) > Bpod (ITI)
    Criterion: All three boolean comparisons true on 99% of trials
    """
    t = ~data["correct"]
    metric = (
        (data["intervals_0"][t] < data["goCue_times"][t])
        & (data["goCue_times"][t] < data["errorCue_times"][t])
        & (data["errorCue_times"][t] < data["itiIn_times"][t])
    )
    criteria = metric
    return np.mean(criteria) if pass_crit else metric


@bpod_data_loader
def load_trial_event_sequence_correct(eid, data=None, pass_crit=False):
    """ 14. on correct trials : 1 audio events, 3 Bpod events (valve open, trial start, ITI)
    (ITI task version dependent on ephys)
    Variable name: trial_event_sequence_correct
    Metric: Bpod (trial start) > audio (go cue) > Bpod (valve) > Bpod (ITI)
    Criterion: All three boolean comparisons true on 99% of trials
    """
    t = data["correct"]
    metric = (
        (data["intervals_0"][t] < data["goCue_times"][t])
        & (data["goCue_times"][t] < data["valveOpen_times"][t])
        & (data["valveOpen_times"][t] < data["itiIn_times"][t])
    )
    criteria = metric
    return np.mean(criteria) if pass_crit else metric


@bpod_data_loader
def load_trial_length(eid, data=None, pass_crit=False):
    """ 15. Time between goCue and feedback <= 60s
    Variable name: trial_length
    Metric: (feedback_time - gocue_time) < 60.1 s AND (feedback_time - gocue_time) > 0 s
    Criterion: both true on 99% of trials
    """
    metric = (data["feedback_times"] - data["goCue_times"] < 60.1) & (
        data["feedback_times"] - data["goCue_times"] > 0
    )
    criteria = metric
    return np.mean(criteria) if pass_crit else metric


# @bpod_data_loader
# def load_1(eid, data=None, pass_crit=False):
#     """ 16. Between go tone and feedback, frame2ttl should be changing at ~60Hz
#     if wheel moves (exact frequency depending on velocity)
#     Variable name:
#     Metric:
#     Criterion:
#     """
#     pass


# Trigger response checks
@bpod_data_loader
def load_goCue_delays(eid, data=None, pass_crit=False):
    """ 25.Trigger response difference
    Variable name: goCue_delays
    Metric: goCue_times - goCueTrigger_times
    Criterion: 99% <= 1.5ms
    """
    metric = data["goCue_times"] - data["goCueTrigger_times"]
    criteria = metric <= 0.0015
    return np.mean(criteria) if pass_crit else metric


@bpod_data_loader
def load_errorCue_delays(eid, data=None, pass_crit=False):
    """ 26.Trigger response difference
    Variable name: errorCue_delays
    Metric: errorCue_times - errorCueTrigger_times
    Criterion: 99% <= 1.5ms
    """
    metric = data["errorCue_times"] - data["errorCueTrigger_times"]
    criteria = metric[data["feedbackType"] == -1] <= 0.0015
    return np.mean(criteria) if pass_crit else metric


@bpod_data_loader
def load_stimOn_delays(eid, data=None, pass_crit=False):
    """ 27. Trigger response difference
    Variable name: stimOn_delays
    Metric: stimOn_times - stiomOnTrigger_times
    Criterion: 99% <  150ms
    """
    metric = data["stimOn_times"] - data["stimOnTrigger_times"]
    criteria = (metric <= 0.15) & (metric > 0)
    return np.mean(criteria) if pass_crit else metric


@bpod_data_loader
def load_stimOff_delays(eid, data=None, pass_crit=False):
    """ 28.Trigger response difference
    Variable name: stimOff_delays
    Metric: stimOff_times - stimOffTrigger_times
    Criterion:99% <  150ms
    """
    metric = data["stimOff_times"] - data["stimOffTrigger_times"]
    criteria = metric <= 0.15
    return np.mean(criteria) if pass_crit else metric


@bpod_data_loader
def load_stimFreeze_delays(eid, data=None, pass_crit=False):
    """ 29.Trigger response difference
    Variable name: stimFreeze_delays
    Metric: stimFreeze_times - stimFreezeTrigger_times
    Criterion: 99% <  150ms
    """
    metric = data["stimFreeze_times"] - data["stimFreezeTrigger_times"]
    criteria = metric[~np.isnan(metric)] <= 0.15
    return np.mean(criteria) if pass_crit else metric


# Session level?
# bpod_ntrials = len(raw.load_data(one.path_from_eid(eid)))
@bpod_data_loader
def load_nDatasetTypes(eid, data=None, pass_crit=False):
    """ 17. Proportion of datasetTypes extracted
    Variable name: nDatasetTypes
    Metric: len(one.load(eid, offline=True, download_only=True)) / nExpetedDatasetTypes
    (hardcoded per task?)
    """
    return


@bpod_data_loader
def load_intervals(eid, data=None, pass_crit=False):
    """ 18. Proportion of ntrials from ONE to bpod
    Variable name: intervals
    Metric: len(one.load(eid, dataset_types=’trials.intervals’)) / bpod_ntrials
    """
    dset = one.load(eid, dataset_types="trials.intervals")[0]
    if dset is not None:
        return len(dset) / len(data["intervals_0"])
    return


@bpod_data_loader
def load_stimOnTrigger_times(eid, data=None, pass_crit=False):
    """ 19.Proportion of stimOnTrigger_times to bpod_ntrials
    Variable name: stimOnTrigger_times
    Metric: len(one.load(eid, dataset_types=’trials.stimOnTrigger_times’)) / bpod_ntrials
    """
    dset = one.load(eid, dataset_types="trials.stimOnTrigger_times")[0]
    if dset is not None:
        return len(dset) / len(data["intervals_0"])
    return


@bpod_data_loader
def load_stimOn_times(eid, data=None, pass_crit=False):
    """ 20.Proportion of stimOn_times to ntrials
    Variable name: stimOn_times
    Metric:
    """
    dset = one.load(eid, dataset_types="trials.stimOn_times")[0]
    if dset is not None:
        return len(dset) / len(data["intervals_0"])
    return


@bpod_data_loader
def load_goCueTrigger_times(eid, data=None, pass_crit=False):
    """ 21.Proportion of goCueTrigger_times to bpod_ntrials
    Variable name: goCueTrigger_times
    Metric:
    """
    dset = one.load(eid, dataset_types="trials.goCueTrigger_times")[0]
    if dset is not None:
        return len(dset) / len(data["intervals_0"])
    return


@bpod_data_loader
def load_goCue_times(eid, data=None, pass_crit=False):
    """ 22.Proportion of goCue_times to bpod_ntrials
    Variable name: goCue_times
    Metric:
    """
    dset = one.load(eid, dataset_types="trials.goCue_times")[0]
    if dset is not None:
        return len(dset) / len(data["intervals_0"])
    return


@bpod_data_loader
def load_response_times(eid, data=None, pass_crit=False):
    """ 23. Proportion of response_times to bpod_ntrials
    Variable name: response_times
    Metric:
    """
    dset = one.load(eid, dataset_types="trials.response_times")[0]
    if dset is not None:
        return len(dset) / len(data["intervals_0"])
    return


@bpod_data_loader
def load_feedback_times(eid, data=None, pass_crit=False):
    """ 24.Proportion of feedback_times to bpod_ntrials
    Variable name: feedback_times
    Metric:
    """
    dset = one.load(eid, dataset_types="trials.feedback_times")[0]
    if dset is not None:
        return len(dset) / len(data["intervals_0"])
    return


if __name__ == "__main__":
    eid, det = random_ephys_session("churchlandlab")
    print(det)
    # eid = "2e6e179c-fccc-4e8f-9448-ce5b6858a183"
    # det = {
    #     "subject": "CSHL060",
    #     "start_time": "2020-03-09T14:31:01",
    #     "number": 2,
    #     "lab": "churchlandlab",
    #     "project": "ibl_neuropixel_brainwide_01",
    #     "url":
    # "https://alyx.internationalbrainlab.org/sessions/2e6e179c-fccc-4e8f-9448-ce5b6858a183",
    #     "task_protocol": "_iblrig_tasks_ephysChoiceWorld6.4.0",
    #     "local_path":
    # "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL060/2020-03-09/002",
    # }
    # {
    #     "subject": "CSHL052",
    #     "start_time": "2020-02-21T13:24:45",
    #     "number": 1,
    #     "lab": "churchlandlab",
    #     "project": "ibl_neuropixel_brainwide_01",
    #     "url":
    # "https://alyx.internationalbrainlab.org/sessions/4b00df29-3769-43be-bb40-128b1cba6d35",
    #     "task_protocol": "_iblrig_tasks_ephysChoiceWorld6.2.5",
    #     "local_path":
    # "/home/nico/Downloads/FlatIron/churchlandlab/Subjects/CSHL052/2020-02-21/001",
    # }
    data = bpodqc.load_bpod_data(eid, fpga_time=False)
    metrics = get_qcmetrics_frame(eid, data=data)
    criteria = get_qccriteria_frame(eid, data=data)
    # trd = bpodqc.get_session_trigger_response_delays(eid)

    # boxplot_metrics(eid)

    # eid = "0deb75fb-9088-42d9-b744-012fb8fc4afb"
    # eid = "af74b29d-a671-4c22-a5e8-1e3d27e362f3"
    # # lab = 'zadorlab'
    # # ed = search_lab_ephys_sessions(lab, ['trials.stimOn_times', 'trials.goCue_times'])

    # # f, ax = plt.subplots()
    # labs = one.list(None, "lab")
    # eids = []
    # details = []
    # for lab in labs:
    #     ed = search_lab_ephys_sessions(
    #         lab,
    #         dstypes=[
    #             "_iblrig_taskData.raw",
    #         ],
    #         check_download=True,
    #     )
    #     if ed is not None:
    #         eids.extend(ed[0])
    #         details.extend(ed[1])
    # df = _load_df_from_details(details, func=load_stimOff_itiIn_delays)
    # boxplots_from_df(df, describe=True, title='itiIn - stimOff')
    # plt.show()
    # get_session_stimon_gocue_delays(eid)
    # get_response_feddback_delays(eid)
    # get_response_stimFreeze_delays(eid)  # FIXME:Have to fix timescales!!!!
    # bpod = bpodqc.load_bpod_data(eid)
    # response = one.load(eid, dataset_types=['trials.response_times'])[0]
    # plt.plot(response, '-o', label='response_one')
    # plt.plot(bpod['response_times'], '-o', label='response_bpod')
    # plt.show()
    # plt.legend(loc='best')

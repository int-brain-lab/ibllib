import json
from functools import wraps

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ibllib.ephys.bpodqc as bpodqc
from ibllib.ephys.oneutils import search_lab_ephys_sessions
from oneibl.one import ONE
from alf.io import is_details_dict, _to_eid

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
    df,
    ax=None,
    describe=False,
    title="",
    xlabel="Seconds (s)",
    xscale="symlog",
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


class SessionBehaviorMetrics(object):
    def __init__(self, eid, from_one=False):
        self.eid = eid
        self.data = bpodqc.load_bpod_data(eid, fpga_time=from_one)
        self.from_one = from_one


def bpod_data_loader(func):
    """ Checks if data is None loads eid data in case
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(args, kwargs)
        if not kwargs or kwargs['data'] is None:
            kwargs['data'] = bpodqc.load_bpod_data(args[0])
        return func(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------- #
@bpod_data_loader
def load_session_stimon_gocue_delays(eid, data=None):
    """ 1. StimOn and GoCue and should be within a 10 ms of each other on 99% of trials
    Variable name: stimOn_goCue_delays
    Metric: goCue_times - stimOn_times (from ONE)
    Criterion: (M<10 ms for 99%) of trials AND (M > 0 ms for 99% of trials)
    """
    return data['goCue_times'] - data['stimOn_times']


@bpod_data_loader
def load_session_response_feddback_delays(eid, data=None):
    """ 2. response_time and feedback_time
    Variable name: response_feedback_delays
    Metric: Feedback_time - response_time
    Criterion: (M <10 ms for 99% of trials) AND ( M > 0 ms for 100% of trials)
    _one_load_session_delays_between_events(
        eid, "trials.response_times", "trials.feedback_times"
    )"""
    return data['feedback_times'] - data['response_times']


@bpod_data_loader
def load_session_response_stimFreeze_delays(eid, data=None):
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
    return data['stimFreeze_times'] - data['response_times']


@bpod_data_loader
def load_session_stimOff_itiIn_delays(eid, data=None):
    """ 4. Start of iti_in should be within a very small tolerance of the stim off
    Variable name: stimOff_itiIn_delays
    Metric: iti_in - stim_off
    Criterion: (M<10 ms for 99% of trials) AND (M > 0 ms for 99% of trials)
    itiIn = bpodqc.get_itiIn_times(eid, save=False)
    _, stimOff, _ = bpodqc.get_stimOnOffFreeze_times_from_BNC1(eid, save=False)
    if len(itiIn) != len(stimOff):
        print(f"Length mismatch iniIn and stimOff: {len(itiIn)}, {len(stimOff)}")
    """
    return data['itiIn_times'] - data['stimOff_times']


@bpod_data_loader
def load_session_wheel_freeze_during_quiescence(eid, data=None):
    """ 5. Wheel should not move more than 2 ticks each direction for at least 0.2 + 0.2-0.6
    amount of time (quiescent period; exact value in bpod['quiescence']) before go cue
    Variable name: wheel_freeze_during_quiescence
    Metric: max(abs( w(t) - w_start )) over interval
    interval = [quiescent_end_time-quiescent_duration+0.02, quiescent_end_time]
    Criterion: <2 ticks for 99% of trials
    """
    pass


@bpod_data_loader
def load_wheel_move_before_feedback(eid, data=None):
    """ 6. Wheel should move before feedback
    Variable name: wheel_move_before_feedback
    Metric: max(abs( w(t) - w_start )) over interval [gocue_time, feedback_time]
    Criterion: >2 ticks for 99% of non-NoGo trials
    """
    pass


@bpod_data_loader
def load_session_stimulus_move_before_goCue(eid, data=None):
    """ 7. No stimulus movements between trialstart_time and gocue_time-20 ms
    Variable name: stimulus_move_before_goCue
    Metric: count if any stimulus change events between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    ---
    goCue_times = one.load(eid, dataset_types="trials.goCue_times")
    bpod2fpga = bpodqc.get_bpod2fpga_times_func(eid)
    BNC1_times = bpod2fpga(BNC1['times'])
    """
    BNC1, _ = bpodqc.get_bpod_fronts(eid)
    s = BNC1['times']
    out = np.array([])
    for i, c in zip(data['intervals_0'], data['goCue_times']):
        out = np.append(out, np.any(s[s > i] < c))
    return out


@bpod_data_loader
def load_session_positive_feedback_stimOff_delays(eid, data=None):
    """ 8. Delay between valve and stim off should be 1s
    Variable name: positive_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 1s)
    Criterion: <20 ms on 99% of correct trials
    """
    return np.abs(data['stimOff_times'] - data['feedback_times'] - 1)


@bpod_data_loader
def load_session_negative_feedback_stimOff_delays(eid, data=None):
    """ 9.Delay between noise and stim off should be 2 second
    Variable name: negative_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 2s)
    Criterion: <20 ms on 99% of incorrect trials
    """
    return np.abs(data['stimOff_times'] - data['errorCue_times'] - 2)


# @bpod_data_loader
# def load_session_0(eid, data=None):
#     """ 10. Number of Bonsai command to change screen should match
#     Number of state change of frame2ttl
#     Variable name: syncSquare
#     Metric: (count of bonsai screen updates) - (count of frame2ttl)
#     Criterion: 0 on 99% of trials
#     """
#     pass


@bpod_data_loader
def load_session_valve_pre_trial(eid, data=None):
    """ 11. No valve outputs between trialstart_time and gocue_time-20 ms
    Variable name: valve_pre_trial
    Metric: count of valve events between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    return data['valveOpen_times'] < data['goCue_times']


@bpod_data_loader
def load_session_audio_pre_trial(eid, data=None):
    """ 12. No audio outputs between trialstart_time and gocue_time-20 ms
    Variable name: audio_pre_trial
    Metric: count of audio events between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    _, BNC2 = bpodqc.get_bpod_fronts(eid)
    s = BNC2['times']
    out = np.array([], dtype=np.bool)
    for i, c in zip(data['intervals_0'], data['goCue_times']):
        out = np.append(out, np.any(s[s > i] < (c - 0.02)))
    return out


# Sequence of events:
@bpod_data_loader
def load_session_trial_event_sequence_error(eid, data=None):
    """ 13. on incorrect / miss trials : 2 audio events, 2 Bpod events (trial start, ITI)
    Variable name: trial_event_sequence_error
    Metric: Bpod (trial start) > audio (go cue) > audio (wrong) > Bpod (ITI)
    Criterion: All three boolean comparisons true on 99% of trials
    """
    t = ~data['correct']
    return (
        (data['intervals_0'][t] < data['goCue_times'][t]) &
        (data['goCue_times'][t] < data['errorCue_times'][t]) &
        (data['errorCue_times'][t] < data['itiIn_times'][t])
    )


@bpod_data_loader
def load_session_trial_event_sequence_correct(eid, data=None):
    """ 14. on correct trials : 1 audio events, 3 Bpod events (valve open, trial start, ITI)
    (ITI task version dependent on ephys)
    Variable name: trial_event_sequence_correct
    Metric: Bpod (trial start) > audio (go cue) > Bpod (valve) > Bpod (ITI)
    Criterion: All three boolean comparisons true on 99% of trials
    """
    t = data['correct']
    return (
        (data['intervals_0'][t] < data['goCue_times'][t]) &
        (data['goCue_times'][t] < data['valveOpen_times'][t]) &
        (data['valveOpen_times'][t] < data['itiIn_times'][t])
    )


@bpod_data_loader
def load_session_trial_length(eid, data=None):
    """ 15. Time between goCue and feedback <= 60s
    Variable name: trial_length
    Metric: (feedback_time - gocue_time) < 60.1 s AND (feedback_time - gocue_time) > 0 s
    Criterion: both true on 99% of trials
    """
    return (
        (data['feedback_times'] - data['goCue_times'] < 60.1) &
        (data['feedback_times'] - data['goCue_times'] > 0)
    )


# @bpod_data_loader
# def load_session_1(eid, data=None):
#     """ 16. Between go tone and feedback, frame2ttl should be changing at ~60Hz
#     if wheel moves (exact frequency depending on velocity)
#     Variable name:
#     Metric:
#     Criterion:
#     """
#     pass


# Session level?
# bpod_ntrials = len(raw.load_session_data(one.path_from_eid(eid)))
@bpod_data_loader
def load_session_nDatasetTypes(eid, data=None):
    """ 17. Proportion of datasetTypes extracted
    Variable name: nDatasetTypes
    Metric: len(one.load(eid, offline=True, download_session_only=True)) / nExpetedDatasetTypes
    (hardcoded per task?)
    """
    pass


@bpod_data_loader
def load_session_intervals(eid, data=None):
    """ 18. Proportion of ntrials from ONE to bpod
    Variable name: intervals
    Metric: len(one.load(eid, dataset_types=’trials.intervals’)) / bpod_ntrials
    """
    return len(one.load(eid, dataset_types='trials.intervals')[0]) / len(data['intervals_0'])


@bpod_data_loader
def load_session_stimOnTrigger_times(eid, data=None):
    """ 19.Proportion of stimOnTrigger_times to bpod_ntrials
    Variable name: stimOnTrigger_times
    Metric: len(one.load(eid, dataset_types=’trials.stimOnTrigger_times’)) / bpod_ntrials
    """
    return (
        len(one.load(eid, dataset_types='trials.stimOnTrigger_times')[0]) /
        len(data['intervals_0'])
    )


@bpod_data_loader
def load_session_stimOn_times(eid, data=None):
    """ 20.Proportion of stimOn_times to ntrials
    Variable name: stimOn_times
    Metric:
    """
    return len(one.load(eid, dataset_types='trials.stimOn_times')[0]) / len(data['intervals_0'])


@bpod_data_loader
def load_session_goCueTrigger_times(eid, data=None):
    """ 21.Proportion of goCueTrigger_times to bpod_ntrials
    Variable name: goCueTrigger_times
    Metric:
    """
    return (
        len(one.load(eid, dataset_types='trials.goCueTrigger_times')[0]) / len(data['intervals_0'])
    )


@bpod_data_loader
def load_session_goCue_times(eid, data=None):
    """ 22.Proportion of goCue_times to bpod_ntrials
    Variable name: goCue_times
    Metric:
    """
    return len(one.load(eid, dataset_types='trials.goCue_times')[0]) / len(data['intervals_0'])


@bpod_data_loader
def load_session_response_times(eid, data=None):
    """ 23. Proportion of response_times to bpod_ntrials
    Variable name: response_times
    Metric:
    """
    return len(one.load(eid, dataset_types='trials.response_times')[0]) / len(data['intervals_0'])


@bpod_data_loader
def load_session_feedback_times(eid, data=None):
    """ 24.Proportion of feedback_times to bpod_ntrials
    Variable name: feedback_times
    Metric:
    """
    return len(one.load(eid, dataset_types='trials.feedback_times')[0]) / len(data['intervals_0'])


# Trigger response checks
@bpod_data_loader
def load_session_goCue_delays(eid, data=None):
    """ 25.Trigger response difference
    Variable name: goCue_delays
    Metric: goCue_times - goCueTrigger_times
    Criterion: 99% <= 1ms
    """
    return data['goCue_times'] - data['goCueTrigger_times']


@bpod_data_loader
def load_session_errorCue_delays(eid, data=None):
    """ 26.Trigger response difference
    Variable name: errorCue_delays
    Metric: errorCue_times - errorCueTrigger_times
    Criterion: 99% <= 1ms
    """
    return data['errorCue_times'] - data['errorCueTrigger_times']


@bpod_data_loader
def load_session_stimOn_delays(eid, data=None):
    """ 27. Trigger response difference
    Variable name: stimOn_delays
    Metric: stimOn_times - stiomOnTrigger_times
    Criterion: 99% <  150ms
    """
    return data['stimOn_times'] - data['stimOnTrigger_times']


@bpod_data_loader
def load_session_stimOff_delays(eid, data=None):
    """ 28.Trigger response difference
    Variable name: stimOff_delays
    Metric: stimOff_times - stimOffTrigger_times
    Criterion:99% <  150ms
    """
    return data['stimOff_times'] - data['stimOffTrigger_times']


@bpod_data_loader
def load_session_stimFreeze_delays(eid, data=None):
    """ 29.Trigger response difference
    Variable name: stimFreeze_delays
    Metric: stimFreeze_times - stimFreezeTrigger_times
    Criterion: 99% <  150ms
    """
    return data['stimFreeze_times'] - data['stimFreezeTrigger_times']


if __name__ == "__main__":
    eid = "0deb75fb-9088-42d9-b744-012fb8fc4afb"
    eid = "af74b29d-a671-4c22-a5e8-1e3d27e362f3"
    # lab = 'zadorlab'
    # ed = search_lab_ephys_sessions(lab, ['trials.stimOn_times', 'trials.goCue_times'])

    # f, ax = plt.subplots()
    labs = one.list(None, "lab")
    eids = []
    details = []
    for lab in labs:
        ed = search_lab_ephys_sessions(
            lab,
            dstypes=[
                "_iblrig_taskData.raw",
            ],
            check_download=True,
        )
        if ed is not None:
            eids.extend(ed[0])
            details.extend(ed[1])
    df = _load_df_from_details(details, func=load_session_stimOff_itiIn_delays)
    boxplots_from_df(df, describe=True, title='itiIn - stimOff')
    plt.show()
    # get_session_stimon_gocue_delays(eid)
    # get_response_feddback_delays(eid)
    # get_response_stimFreeze_delays(eid)  # FIXME:Have to fix timescales!!!!
    # bpod = bpodqc.load_bpod_data(eid)
    # response = one.load(eid, dataset_types=['trials.response_times'])[0]
    # plt.plot(response, '-o', label='response_one')
    # plt.plot(bpod['response_times'], '-o', label='response_bpod')
    # plt.show()
    # plt.legend(loc='best')

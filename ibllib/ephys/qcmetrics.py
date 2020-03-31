import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import ibllib.ephys.bpodqc as bpodqc
from ibllib.ephys.ephysqc import _qc_from_path
from oneibl.one import ONE

# plt.ion()

one = ONE(
    username="niccolo",
    password="ItTakesBrains!",
    base_url="https://alyx.internationalbrainlab.org",
)


def _one_load_session_delays_between_events(eid, dstype1, dstype2):
    """ Returns difference between times of 2 different dataset types
    Func is called with eid and dstypes in temporal order, returns delay between
    event1 and event 2, i.e. event_time2 - event_time1
    """
    event_times1, event_times2 = one.load(eid, dataset_types=[dstype1, dstype2])
    if all(np.isnan(event_times1)) or all(np.isnan(event_times2)):
        print(
            f"{eid}\nall {dstype1} nan: {all(np.isnan(event_times1))}",
            f"\nall {dstype2} nan: {all(np.isnan(event_times2))}",
        )
        return
    delay_between_events = event_times2 - event_times1
    return delay_between_events


def search_lab_ephys_sessions(lab: str, dstypes: list, nlatest: int = 3, det: bool = True):
    ephys_sessions0, session_details0 = one.search(
        task_protocol="_iblrig_tasks_ephysChoiceWorld6.4.0",
        dataset_types=dstypes,
        limit=1000,
        details=True,
        lab=lab,
    )
    ephys_sessions1, session_details1 = one.search(
        task_protocol="_iblrig_tasks_ephysChoiceWorld6.2.5",
        dataset_types=dstypes,
        limit=1000,
        details=True,
        lab=lab,
    )
    ephys_sessions = list(ephys_sessions0) + list(ephys_sessions1)
    session_details = list(session_details0) + list(session_details1)
    print(f"Processing {lab}")
    # Check if you found anything
    if ephys_sessions == []:
        print(f"No sessions found for {lab}")
        return
    out_sessions = []
    out_details = []
    for esess, edets in zip(ephys_sessions, session_details):
        dstypes_data = one.load(esess, dataset_types=dstypes)
        # Check if dstypes have all NaNs
        skip_esess = False
        for dsname, dsdata in zip(dstypes, dstypes_data):
            if 'raw' in dsname:
                continue
            if np.all(np.isnan(dsdata)):
                print(f"Skipping {esess}, one or more dstypes are all NaNs")
                skip_esess = True
        if skip_esess:
            continue
        # Check if all dstypes have the same length
        if not all(len(x) == len(dstypes_data[0]) for x in dstypes_data):
            print(f"Skipping {esess}, one or more dstypes have different lengths")
            continue
        out_sessions.append(esess)
        out_details.append(edets)
        if len(out_details) == nlatest:
            break
    return out_sessions, out_details if det else out_sessions


def _load_df_from_details(details=None, func=None):
    """
    Applies a session level func(eid) from session details dict from Alyx
    """
    if details is None or func is None:
        return
    if isinstance(details, dict):
        details = [details]
    data = []
    labels = []
    for i, det in enumerate(details):
        eid = det["url"][-36:]
        data.append(func(eid))
        labels.append(det["lab"] + str(i))

    df = pd.DataFrame(data).transpose()
    df.columns = labels

    return df


# ---------------------------------------------------------------------------- #
# 1.
# Variable name: stimOn_goCue_delays
# Metric: goCue_times - stimOn_times (from ONE)
# Criterion: (M<10 ms for 99%) of trials AND (M > 0 ms for 99% of trials)
def load_session_stimon_gocue_delays(eid):
    return _one_load_session_delays_between_events(
        eid, "trials.stimOn_times", "trials.goCue_times"
    )


def process_session_stimon_gocue_delays(details):
    if details is None:
        return
    df = _load_df_from_details(details, func=load_session_stimon_gocue_delays)
    return df


def plot_session_stimon_gocue_delays(details: list, ax=None, describe=False):
    if details is None:
        return
    if ax is None:
        f, ax = plt.subplots()

    # Load and process data
    df = process_session_stimon_gocue_delays(details)
    if describe:
        desc = df.describe()
        print(json.dumps(json.loads(desc.to_json()), indent=1))
    # Plot
    p = sns.boxplot(data=df, ax=ax, orient="h")
    p.set_title("goCue - stimOn")
    p.set_xlabel("Seconds (s)")
    p.set(xscale="symlog")


# 2.
# Variable name: response_feedback_delays
# Metric: Feedback_time - response_time
# Criterion: (M <10 ms for 99% of trials) AND ( M > 0 ms for 100% of trials)
def load_session_response_feddback_delays(eid):
    return _one_load_session_delays_between_events(
        eid, "trials.response_times", "trials.feedback_times"
    )


def process_session_response_feedback_delays(details):
    if details is None:
        return
    df = _load_df_from_details(details, func=load_session_response_feddback_delays)
    return df


def plot_session_response_feedback_delays(details: list, ax=None, describe=False):
    if details is None:
        return
    if ax is None:
        f, ax = plt.subplots()
    # Load and process data
    df = process_session_stimon_gocue_delays(details)
    if describe:
        desc = df.describe()
        print(json.dumps(json.loads(desc.to_json()), indent=1))
    # Plot
    p = sns.boxplot(data=df, ax=ax, orient="h")
    p.set_title("feedback - response")
    p.set_xlabel("Seconds (s)")
    p.set(xscale="symlog")


# 3.
# Variable name: response_stimFreeze_delays
# Metric: stim_freeze - response_time
# Criterion: (M<100 ms for 99% of trials) AND (M > 0 ms for 100% of trials)
def load_session_response_stimFreeze_delays(eid):
    response = one.load(eid, dataset_types=["trials.response_times"])[0]
    _, _, stimFreeze = bpodqc.get_stimOnOffFreeze_times_from_BNC1(eid)
    bpod2fpga = bpodqc.get_bpod2fpga_times_func(eid)
    stimFreeze = bpod2fpga(stimFreeze)
    if len(response) != len(stimFreeze):
        session_path = one.path_from_eid(eid)
        response = bpodqc.get_response_times(session_path, save=False)
    assert len(response) == len(stimFreeze)
    return stimFreeze - response


def process_session_response_stimFreeze_delays(details):
    if details is None:
        return
    df = _load_df_from_details(details, func=load_session_response_stimFreeze_delays)
    return df


def plot_session_response_stimFreeze_delays(details: list, ax=None, describe=False):
    if details is None:
        return
    if ax is None:
        f, ax = plt.subplots()
    # Load and process data
    df = process_session_response_stimFreeze_delays(details)
    if describe:
        desc = df.describe()
        print(json.dumps(json.loads(desc.to_json()), indent=1))
    # Plot
    p = sns.boxplot(data=df, ax=ax, orient="h")
    p.set_title("stimFreeze - response")
    p.set_xlabel("Seconds (s)")
    p.set(xscale="symlog")


# 4.
# Variable name: stimOff_itiIn_delays
# Metric: iti_in - stim_off
# Criterion: (M<10 ms for 99% of trials) AND (M > 0 ms for 99% of trials)
def load_session_stimOff_itiIn_delays(eid):
    itiIn = bpodqc.get_itiIn_times(eid, save=False)
    _, stimOff, _ = bpodqc.get_stimOnOffFreeze_times_from_BNC1(eid, save=False)
    if len(itiIn) != len(stimOff):
        print(f"Length mismatch iniIn and stimOff: {len(itiIn)}, {len(stimOff)}")
    return itiIn - stimOff


def process_session_stimOff_itiIn_delays(details):
    """process a list of sessions from details objects"""
    if details is None:
        return
    df = _load_df_from_details(details, func=load_session_stimOff_itiIn_delays)
    return df


def plot_session_stimOff_itiIn_delays(details: list, ax=None, describe=False):
    if details is None:
        return
    if ax is None:
        f, ax = plt.subplots()
    # Load and process data
    df = process_session_stimOff_itiIn_delays(details)
    if describe:
        desc = df.describe()
        print(json.dumps(json.loads(desc.to_json()), indent=1))
    # Plot
    p = sns.boxplot(data=df, ax=ax, orient="h")
    p.set_title("itiIn - stimOff")
    p.set_xlabel("Seconds (s)")
    p.set(xscale="symlog")


# 5.
# Variable name: wheel_freeze_during_quiescence
# Metric: max(abs( w(t) - w_start )) over interval
# interval = [quiescent_end_time-quiescent_duration+0.02, quiescent_end_time]
# Criterion: <2 ticks for 99% of trials
def load_session_wheel_freeze_during_quiescence(eid):
    pass


def process_session_wheel_freeze_during_quiescence(details):
    pass


def plot_wheel_freeze_during_quiescence(details: list, ax=None, describe=False):
    pass


# 6.
# Variable name: wheel_move_before_feedback
# Metric: max(abs( w(t) - w_start )) over interval [gocue_time, feedback_time]
# Criterion: >2 ticks for 99% of non-NoGo trials
def load_wheel_move_before_feedback(eid):
    pass


def process_wheel_move_before_feedback(details):
    pass


def plot_wheel_move_before_feedback(details: list, ax=None, describe=False):
    pass


# 7.
# Variable name: stimulus_move_before_goCue
# Metric: count of stimulus change events between trialstart_time and (gocue_time-20ms)
# Criterion: 0 on 99% of trials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 8.
# Variable name: positive_feedback_stimOff_delays
# Metric: abs((stimoff_time - feedback_time) - 1s)
# Criterion: <20 ms on 99% of correct trials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 9.Delay between noise and stim off should be 2 second ✓ Ephys ✘ Bpod
# Variable name: negative_feedback_stimOff_delays
# Metric: abs((stimoff_time - feedback_time) - 2s)
# Criterion: <20 ms on 99% of incorrect trials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 10. Number of Bonsai command to change screen should match Number of state change of frame2ttl ✓ Ephys ✘ Bpod
# Variable name: syncSquare
# Metric: (count of bonsai screen updates) - (count of frame2ttl)
# Criterion: 0 on 99% of trials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 11. No valve outputs between trialstart_time and gocue_time-20 ms ✘ Ephys ✘ Bpod
# Variable name: valve_pre_trial
# Metric: count of valve events between trialstart_time and (gocue_time-20ms)
# Criterion: 0 on 99% of trials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 12. No audio outputs between trialstart_time and gocue_time-20 ms ✘ Ephys ✘ Bpod
# Variable name: audio_pre_trial
# Metric: count of audio events between trialstart_time and (gocue_time-20ms)
# Criterion: 0 on 99% of trials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass

# Sequence of events:
# 13. on incorrect / miss trials : 2 audio events, 2 Bpod events (trial start, ITI) ✘ Ephys ✘ Bpod
# Variable name: trial_event_sequence_error
# Metric: Bpod (trial start) > audio (go cue) > audio (wrong) > Bpod (ITI)
# Criterion: All three boolean comparisons true on 99% of trials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 14. on correct trials : 1 audio events, 3 Bpod events (valve open, trial start, ITI) ✘ Ephys ✘ Bpod (ITI task version dependent on ephys)
# Variable name: trial_event_sequence_correct
# Metric: Bpod (trial start) > audio (go cue) > Bpod (valve) > Bpod (ITI)
# Criterion: All three boolean comparisons true on 99% of trials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 15. Time between goCue and feedback <= 60s ✘ Ephys ✘ Bpod
# Variable name: trial_length
# Metric: (feedback_time - gocue_time) < 60.1 s AND (feedback_time - gocue_time) > 0 s
# Criterion: both true on 99% of trials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 16. Between go tone and feedback, frame2ttl should be changing at ~60Hz if wheel moves (exact frequency depending on velocity) ✘ Ephys ✘ Bpod
# Variable name:
# Metric:
# Criterion:
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# Session level?
# bpod_ntrials = len(raw.load_data(one.path_from_eid(eid)))
# 17. Proportion of datasetTypes extracted
# Variable name: nDatasetTypes
# Metric: len(one.load(eid, offline=True, download_only=True)) / nExpetedDatasetTypes (hardcoded per task?)
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 18. Proportion of ntrials from ONE to bpod
# Variable name: intervals
# Metric: len(one.load(eid, dataset_types=’trials.intervals’)) / bpod_ntrials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 19.Proportion of stimOnTrigger_times to bpod_ntrials
# Variable name: stimOnTrigger_times
# Metric: len(one.load(eid, dataset_types=’trials.stimOnTrigger_times’)) / bpod_ntrials
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 20.Proportion of stimOn_times to ntrials
# Variable name: stimOn_times
# Metric:
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 21.Proportion of goCueTrigger_times to bpod_ntrials
# Variable name: goCueTrigger_times
# Metric:
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 22.Proportion of goCue_times to bpod_ntrials
# Variable name: goCue_times
# Metric:
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 23. Proportion of response_times to bpod_ntrials
# Variable name: response_times
# Metric:
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 24.Proportion of feedback_times to bpod_ntrials
# Variable name: feedback_times
# Metric:
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# Trigger response checks

# 25.Trigger response difference
# Variable name: goCue_delays
# Metric: goCue_times - goCueTrigger_times
# Criterion: 99% <= 1ms
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 26.Trigger response difference
# Variable name: errorCue_delays
# Metric: errorCue_times - errorCueTrigger_times
# Criterion: 99% <= 1ms
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 27. Trigger response difference
# Variable name: stimOn_delays
# Metric: stimOn_times - stiomOnTrigger_times
# Criterion: 99% <  150ms
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 28.Trigger response difference
# Variable name: stimOff_delays
# Metric: stimOff_times - stimOffTrigger_times
# Criterion:99% <  150ms
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


# 29.Trigger response difference
# Variable name: stimFreeze_delays
# Metric: stimFreeze_times - stimFreezeTrigger_times
# Criterion: 99% <  150ms
def load_(eid):
    pass


def process_(details):
    pass


def plot_(details: list, ax=None, describe=False):
    pass


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
            [
                "trials.intervals",
                "_iblrig_taskData.raw",
            ],
        )
        if ed is not None:
            eids.extend(ed[0])
            details.extend(ed[1])
    plot_session_stimOff_itiIn_delays(details, describe=True)
    plt.show()
    #  get_session_stimon_gocue_delays(eid)
    # get_response_feddback_delays(eid)
    # get_response_stimFreeze_delays(eid)  # FIXME:Have to fix timescales!!!!
    # bpod = bpodqc.load_bpod_data(eid)
    # response = one.load(eid, dataset_types=['trials.response_times'])[0]
    # plt.plot(response, '-o', label='response_one')
    # plt.plot(bpod['response_times'], '-o', label='response_bpod')
    # plt.show()
    # plt.legend(loc='best')

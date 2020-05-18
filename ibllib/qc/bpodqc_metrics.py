import numpy as np

import ibllib.qc.bpodqc_extractors as bpodqc
from brainbox.behavior.wheel import cm_to_rad, traces_by_trial
from ibllib.io.extractors.training_wheel import get_wheel_position
from ibllib.qc.bpodqc_extractors import bpod_data_loader
from ibllib.qc.oneutils import random_ephys_session
from oneibl.one import ONE

one = ONE(printout=False)


@bpod_data_loader
def get_bpodqc_metrics_frame(eid, data=None, apply_criteria=False):
    """Plottable metrics based on timings"""
    qcmetrics_frame = {
        "_bpod_goCue_delays": load_goCue_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 25)
        "_bpod_errorCue_delays": load_errorCue_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 26)
        "_bpod_stimOn_delays": load_stimOn_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 27)
        "_bpod_stimOff_delays": load_stimOff_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 28)
        "_bpod_stimFreeze_delays": load_stimFreeze_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 29)
        "_bpod_stimOn_goCue_delays": load_stimOn_goCue_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 1)
        "_bpod_response_feedback_delays": load_response_feedback_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 2)
        "_bpod_response_stimFreeze_delays": load_response_stimFreeze_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 3)
        "_bpod_stimOff_itiIn_delays": load_stimOff_itiIn_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 4)
        "_bpod_wheel_freeze_during_quiescence": load_wheel_freeze_during_quiescence(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 5)
        "_bpod_wheel_move_before_feedback": load_wheel_move_before_feedback(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 6)
        "_bpod_wheel_move_during_closed_loop": load_wheel_move_during_closed_loop(
            eid, data=data, apply_criteria=apply_criteria
        ),
        "_bpod_stimulus_move_before_goCue": load_stimulus_move_before_goCue(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 7)
        "_bpod_positive_feedback_stimOff_delays": load_positive_feedback_stimOff_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 8)
        "_bpod_negative_feedback_stimOff_delays": load_negative_feedback_stimOff_delays(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 9)
        "_bpod_valve_pre_trial": load_valve_pre_trial(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 11)
        "_bpod_audio_pre_trial": load_audio_pre_trial(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 12)
        "_bpod_error_trial_event_sequence": load_error_trial_event_sequence(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 13)
        "_bpod_correct_trial_event_sequence": load_correct_trial_event_sequence(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 14)
        "_bpod_trial_length": load_trial_length(
            eid, data=data, apply_criteria=apply_criteria
        ),  # (Point 15)
    }
    return qcmetrics_frame


# SINGLE METRICS
# ---------------------------------------------------------------------------- #
def load_stimOn_goCue_delays(eid, data, apply_criteria=False):
    """ 1. StimOn and GoCue and should be within a 10 ms of each other on 99% of trials
    Variable name: stimOn_goCue_delays
    Metric: stimOn_times - goCue_times
    Criteria: (M<10 ms for 99%) of trials AND (M > 0 ms for 99% of trials)
    """
    metric = data["goCue_times"] - data["stimOn_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 0.01) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_response_feedback_delays(eid, data, apply_criteria=False):
    """ 2. response_time and feedback_time
    Variable name: response_feedback_delays
    Metric: Feedback_time - response_time
    Criterion: (M <10 ms for 99% of trials) AND ( M > 0 ms for 100% of trials)
    _one_load_delays_between_events(
        eid, "trials.response_times", "trials.feedback_times"
    )"""
    metric = data["feedback_times"] - data["response_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = ((metric[~nans] < 0.01) & (metric[~nans] > 0)).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_response_stimFreeze_delays(eid, data, apply_criteria=False):
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
    # Find NaNs (if any of the values are nan operation will be nan)
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    # Test for valid values
    passed[~nans] = ((metric[~nans] < 0.1) & (metric[~nans] > 0)).astype(np.float)
    # Finally remove no_go trials (stimFreeze triggered differently in no_go trials)
    # should account for all the nans
    passed[data["choice"] == 0] = np.nan
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_stimOff_itiIn_delays(eid, data, apply_criteria=False):
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
    passed = ((metric < 0.01) & (metric >= 0)).astype(np.float)
    # Remove no_go trials (stimOff triggered differently in no_go trials)
    metric[data["choice"] == 0] = np.nan
    passed[data["choice"] == 0] = np.nan
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_wheel_freeze_during_quiescence(eid, data=None, apply_criteria=False):
    """ 5. Wheel should not move more than 2 ticks each direction for at least 0.2 + 0.2-0.6
    amount of time (quiescent period; exact value in bpod['quiescence']) before go cue
    Variable name: wheel_freeze_during_quiescence
    Metric: abs(min(W - w_t0), max(W - w_t0)) where W is wheel pos over interval
    np.max(Metric) to get highest displaceente in any direction
    interval = [goCueTrigger_time-quiescent_duration,goCueTrigger_time]
    Criterion: <2 degrees for 99% of trials
    #TODO: check loading!z
    """
    # Load Bpod wheel data
    wheel_data = get_wheel_position(one.path_from_eid(eid))
    assert np.all(np.diff(wheel_data["re_ts"]) > 0)
    assert data['quiescence'].size == data["goCueTrigger_times"].size
    # Get tuple of wheel times and positions over each trial's quiescence period
    qevt_start_times = data["goCueTrigger_times"] - data['quiescence']
    traces = traces_by_trial(
        wheel_data["re_ts"],
        wheel_data["re_pos"],
        start=qevt_start_times,
        end=data["goCueTrigger_times"],
    )

    # metric = np.zeros_like(data['quiescence'])
    # for i, trial in enumerate(traces):
    #     pos = trial[1]
    #     if pos.size > 1:
    #         metric[i] = np.abs(pos.max() - pos.min())
    # -OR-
    metric = np.zeros((len(data['quiescence']), 2))  # (n_trials, n_directions)
    for i, trial in enumerate(traces):
        t, pos = trial
        # Get the last position before the period began
        if pos.size > 1:
            # Find the position of the preceding sample and subtract it
            origin = wheel_data["re_pos"][wheel_data["re_ts"] < t[0]][-1]
            # Find the absolute min and max relative to the last sample
            metric[i, :] = np.abs([np.min(pos - origin), np.max(pos - origin)])
    # Reduce to the largest displacement found in any direction
    metric = np.max(metric, axis=1)
    metric = 180 * metric / np.pi  # convert to degrees from radians
    criterion = 2  # Position shouldn't change more than 2 in either direction
    passed = (metric < criterion).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_wheel_move_before_feedback(eid, data, apply_criteria=False):
    """ 6. Wheel should move within 100ms of feedback
    Variable name: wheel_move_before_feedback
    Metric: (w_t - 0.05) - (w_t + 0.05) where t = feedback_time
    Criterion: != 0 for 99% of non-NoGo trials
    """
    # Load Bpod wheel data
    wheel_data = get_wheel_position(one.path_from_eid(eid))
    assert np.all(np.diff(wheel_data["re_ts"]) > 0)
    # Get tuple of wheel times and positions within 100ms of feedback
    traces = traces_by_trial(
        wheel_data["re_ts"],
        wheel_data["re_pos"],
        start=data["feedback_times"] - 0.05,
        end=data["feedback_times"] + 0.05,
    )
    metric = np.zeros_like(data["feedback_times"])
    # For each trial find the displacement
    for i, trial in enumerate(traces):
        pos = trial[1]
        if pos.size > 1:
            metric[i] = pos[-1] - pos[0]

    # except no-go trials
    metric[data["choice"] == 0] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan

    passed[~nans] = (metric[~nans] != 0).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_wheel_move_during_closed_loop(eid, data, apply_criteria=False):
    """ Wheel should move a sufficient amount during the closed-loop period
    Variable name: wheel_move_during_closed_loop
    Metric: abs(w_resp - w_t0) - threshold_displacement, where w_resp = position at response
      time, w_t0 = position at go cue time, threshold_displacement = displacement required to move
      35 visual degrees
    Criterion: displacement < 1 visual degree for 99% of non-NoGo trials
    """
    # Load Bpod wheel data
    wheel_data = get_wheel_position(one.path_from_eid(eid))
    assert np.all(np.diff(wheel_data["re_ts"]) > 0)

    # Get tuple of wheel times and positions over each trial's closed-loop period
    traces = traces_by_trial(
        wheel_data["re_ts"],
        wheel_data["re_pos"],
        start=data["goCueTrigger_times"],
        end=data["response_times"],
    )

    metric = np.zeros_like(data["feedback_times"])
    # For each trial find the absolute displacement
    for i, trial in enumerate(traces):
        t, pos = trial
        # Find the position of the preceding sample and subtract it
        origin = wheel_data["re_pos"][wheel_data["re_ts"] < t[0]][-1]
        if pos.size > 0:
            metric[i] = np.abs(pos - origin).max()

    # Load gain and thresholds for each trial
    gain = one.alyx.rest("sessions", "read", id=eid)['json']['STIM_GAIN']
    gain = np.array([gain] * len(data['position']))
    thresh = data['position']
    # abs displacement, s, in mm required to move 35 visual degrees
    s_mm = np.abs(thresh / gain)  # don't care about direction
    criterion = cm_to_rad(s_mm * 1e-1)  # convert abs displacement to radians (wheel pos is in rad)
    metric = metric - criterion  # difference should be close to 0
    rad_per_deg = cm_to_rad(1 / gain * 1e-1)
    passed = (np.abs(metric) < rad_per_deg).astype(np.float)  # less than 1 visual degree off
    metric[data["choice"] == 0] = np.nan  # except no-go trials
    passed[data["choice"] == 0] = np.nan  # except no-go trials
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_stimulus_move_before_goCue(eid, data, apply_criteria=False):
    """ 7. No stimulus movements between trialstart_time and gocue_time-20 ms
    Variable name: stimulus_move_before_goCue
    Metric: count of any stimulus change events between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    ---
    goCue_times = one.load(eid, dataset_types="trials.goCue_times")
    bpod2fpga = bpodqc.get_bpod2fpga_times_func(eid)
    BNC1_times = bpod2fpga(BNC1['times'])
    """
    # XXX: Check this!
    BNC1, _ = bpodqc.get_bpod_fronts(one.path_from_eid(eid))
    s = BNC1["times"]
    metric = np.array([])
    for i, c in zip(data["intervals_0"], data["goCue_times"]):
        metric = np.append(metric, np.count_nonzero(s[s > i] < (c - 0.02)))

    passed = (metric == 0).astype(np.float)
    # Remove no go trials
    passed[data["choice"] == 0] = np.nan
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_positive_feedback_stimOff_delays(eid, data, apply_criteria=False):
    """ 8. Delay between valve and stim off should be 1s
    Variable name: positive_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 1s)
    Criterion: <150 ms on 99% of correct trials
    """
    metric = np.abs(data["stimOff_times"] - data["feedback_times"] - 1)
    metric[~data["correct"]] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 0.15).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_negative_feedback_stimOff_delays(eid, data, apply_criteria=False):
    """ 9.Delay between noise and stim off should be 2 second
    Variable name: negative_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 2s)
    Criterion: <150 ms on 99% of incorrect trials
    """
    metric = np.abs(data["stimOff_times"] - data["errorCue_times"] - 2)
    # Find NaNs (if any of the values are nan operation will be nan)
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    # Apply criteria
    passed[~nans] = (metric[~nans] < 0.15).astype(np.float)
    # Remove no negative feedback trials
    metric[~data["outcome"] == -1] = np.nan
    passed[~data["outcome"] == -1] = np.nan
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


# @bpod_data_loader
# def load_0(eid, data=None, apply_criteria=False):
#     """ 10. Number of Bonsai command to change screen should match
#     Number of state change of frame2ttl
#     Variable name: syncSquare
#     Metric: (count of bonsai screen updates) - (count of frame2ttl)
#     Criterion: 0 on 99% of trials
#     """
#     pass


def load_valve_pre_trial(eid, data, apply_criteria=False):
    """ 11. No valve outputs between trialstart_time and gocue_time-20 ms
    Variable name: valve_pre_trial
    Metric: Check if valve events exist between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    metric = data["valveOpen_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    # Apply criteria
    passed[~nans] = ~(metric[~nans] < (data["goCue_times"][~nans] - 0.02))
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_audio_pre_trial(eid, data, apply_criteria=False):
    """ 12. No audio outputs between trialstart_time and gocue_time-20 ms
    Variable name: audio_pre_trial
    Metric: Check if audio events exist between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    _, BNC2 = bpodqc.get_bpod_fronts(one.path_from_eid(eid))
    s = BNC2["times"]
    metric = np.array([], dtype=np.bool)
    for i, c in zip(data["intervals_0"], data["goCue_times"]):
        metric = np.append(metric, np.any(s[s > i] < (c - 0.02)))
    passed = (~metric).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


# Sequence of events:
def load_error_trial_event_sequence(eid, data, apply_criteria=False):
    """ 13. on incorrect / miss trials : 2 audio events, 2 Bpod events (trial start, ITI)
    Variable name: error_trial_event_sequence
    Metric: Bpod (trial start) > audio (go cue) > audio (wrong) > Bpod (ITI)
    Criterion: All three boolean comparisons true on 99% of trials
    XXX: figure out single metric to use
    """
    a = np.less(
        data["intervals_0"],
        data["goCue_times"],
        where=(~np.isnan(data["intervals_0"]) & ~np.isnan(data["goCue_times"])),
    )
    b = np.less(
        data["goCue_times"],
        data["errorCue_times"],
        where=(~np.isnan(data["goCue_times"]) & ~np.isnan(data["errorCue_times"])),
    )
    c = np.less(
        data["errorCue_times"],
        data["itiIn_times"],
        where=(~np.isnan(data["errorCue_times"]) & ~np.isnan(data["itiIn_times"])),
    )
    metric = a & b & c
    metric = np.float64(metric)
    # Look only at incorrect or missed trials
    metric[data["correct"]] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = metric[~nans]
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_correct_trial_event_sequence(eid, data, apply_criteria=False):
    """ 14. on correct trials : 1 audio events, 3 Bpod events (valve open, trial start, ITI)
    (ITI task version dependent on ephys)
    Variable name: correct_trial_event_sequence
    Metric: Bpod (trial start) > audio (go cue) > Bpod (valve) > Bpod (ITI)
    Criterion: All three boolean comparisons true on 99% of trials
    XXX: figure out single metric to use
    """
    a = np.less(
        data["intervals_0"],
        data["goCue_times"],
        where=(~np.isnan(data["intervals_0"]) & ~np.isnan(data["goCue_times"])),
    )
    b = np.less(
        data["goCue_times"],
        data["valveOpen_times"],
        where=(~np.isnan(data["goCue_times"]) & ~np.isnan(data["valveOpen_times"])),
    )
    c = np.less(
        data["valveOpen_times"],
        data["itiIn_times"],
        where=(~np.isnan(data["valveOpen_times"]) & ~np.isnan(data["itiIn_times"])),
    )
    metric = a & b & c
    metric = np.float64(metric)
    # Look only at correct trials
    metric[~data["correct"]] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = metric[~nans]
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_trial_length(eid, data, apply_criteria=False):
    """ 15. Time between goCue and feedback <= 60s
    Variable name: trial_length
    Metric: (feedback_time - gocue_time)
    Criteria: M < 60.1 s AND M > 0 s both (true on 99% of trials)
    """
    metric = data["feedback_times"] - data["goCue_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 60.1) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


# @bpod_data_loader
# def load_1(eid, data=None, apply_criteria=False):
#     """ 16. Between go tone and feedback, frame2ttl should be changing at ~60Hz
#     if wheel moves (exact frequency depending on velocity)
#     Variable name:
#     Metric:
#     Criterion:
#     """
#     pass


# Trigger response checks
def load_goCue_delays(eid, data, apply_criteria=False):
    """ 25.Trigger response difference
    Variable name: goCue_delays
    Metric: goCue_times - goCueTrigger_times
    Criterion: 99% <= 1.5ms
    """
    metric = data["goCue_times"] - data["goCueTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = metric[~nans] <= 0.0015
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_errorCue_delays(eid, data, apply_criteria=False):
    """ 26.Trigger response difference
    Variable name: errorCue_delays
    Metric: errorCue_times - errorCueTrigger_times
    Criterion: 99% <= 1.5ms
    """
    metric = data["errorCue_times"] - data["errorCueTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = metric[~nans] <= 0.0015
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_stimOn_delays(eid, data, apply_criteria=False):
    """ 27. Trigger response difference
    Variable name: stimOn_delays
    Metric: stimOn_times - stiomOnTrigger_times
    Criterion: 99% <  150ms
    """
    metric = data["stimOn_times"] - data["stimOnTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_stimOff_delays(eid, data, apply_criteria=False):
    """ 28.Trigger response difference
    Variable name: stimOff_delays
    Metric: stimOff_times - stimOffTrigger_times
    Criterion:99% <  150ms
    """
    metric = data["stimOff_times"] - data["stimOffTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = metric[~nans] <= 0.15
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_stimFreeze_delays(eid, data, apply_criteria=False):
    """ 29.Trigger response difference
    Variable name: stimFreeze_delays
    Metric: stimFreeze_times - stimFreezeTrigger_times
    Criterion: 99% <  150ms
    """
    metric = data["stimFreeze_times"] - data["stimFreezeTrigger_times"]
    passed = np.zeros_like(metric) * np.nan
    passed[~np.isnan(metric)] = metric[~np.isnan(metric)] <= 0.15
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


def load_reward_volumes(eid, data, apply_criteria=False):
    """ xx.Reward volume tests
    Variable name: rewardVolume
    Metric: len(set(rewardVolume)) <= 2 & np.all(rewardVolume <= 3)
    Criterion: 100%
    """
    metric = data["rewardVolume"]
    val = np.min(np.unique(np.nonzero(metric)))
    vals = np.ones(len(metric)) * val
    passed = ((metric >= 1.5) & (metric == vals) & (metric <= 3)).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return passed if apply_criteria else metric


class BpodqcMetrics(object):
    def __init__(self, eid):
        self.eid = eid
        self.data = bpodqc.load_bpod_data(eid, fpga_time=False)

    def compute_metrics(self):
        return

    def apply_criteria(self):
        pass

    def get_extended_qc_frame(self):
        pass

    def patch_alyx_extended_qc(self, frame):
        pass


if __name__ == "__main__":
    eid, det = random_ephys_session("churchlandlab")
    data = bpodqc.load_bpod_data(eid, fpga_time=False)
    metrics = get_bpodqc_metrics_frame(eid, data=data, apply_criteria=False)
    criteria = get_bpodqc_metrics_frame(eid, data=data, apply_criteria=True)
    mean_criteria = {k: np.nanmean(v) for k, v in criteria.items()}

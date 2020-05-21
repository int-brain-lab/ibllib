import logging
import numpy as np

import ibllib.qc.bpodqc_extractors as bpodqc
from brainbox.behavior.wheel import cm_to_rad, traces_by_trial
from ibllib.io.extractors.training_wheel import get_wheel_position
from ibllib.qc.bpodqc_extractors import load_bpod_data, BpodQCExtractor
from ibllib.qc.oneutils import random_ephys_session, check_parse_json
from oneibl.one import ONE

log = logging.getLogger("ibllib")


class BpodQCMetricsFrame(object):
    def __init__(self, eid, one=False, trial_data=None):
        self.one = one or ONE(printout=False)
        self.eid = eid or None
        self.session_path = self.one.path_from_eid(eid)
        self.trial_data = BpodQCExtractor(self.session_path)

        self.metrics = get_bpodqc_metrics_frame()


# XXX: change apply_criteria pattern, always return (metrics, passed)
# XXX: change apply_criteria pattern, always return (metrics, passed)
# XXX: change apply_criteria pattern, always return (metrics, passed)
# XXX: change apply_criteria pattern, always return (metrics, passed)
# XXX: change apply_criteria pattern, always return (metrics, passed)
# XXX: change apply_criteria pattern, always return (metrics, passed)
# XXX: change apply_criteria pattern, always return (metrics, passed)
# XXX: change apply_criteria pattern, always return (metrics, passed)
# XXX: change apply_criteria pattern, always return (metrics, passed)
# XXX: change apply_criteria pattern, always return (metrics, passed)
def get_bpodqc_metrics_frame(trial_data=None):
    one = ONE(printout=False)
    """Plottable metrics based on timings"""
    session_path = one.path_from_eid(eid)
    BNC1, BNC2 = bpodqc.get_bpod_fronts(session_path)

    gain = check_parse_json(one.alyx.rest("sessions", "read", id=eid)["json"])["STIM_GAIN"]

    if trial_data is None:
        trial_data = load_bpod_data(session_path)
    qcmetrics_frame = {
        "_bpod_goCue_delays": load_goCue_delays(trial_data),
        "_bpod_errorCue_delays": load_errorCue_delays(trial_data),
        "_bpod_stimOn_delays": load_stimOn_delays(trial_data),
        "_bpod_stimOff_delays": load_stimOff_delays(trial_data),
        "_bpod_stimFreeze_delays": load_stimFreeze_delays(trial_data),
        "_bpod_stimOn_goCue_delays": load_stimOn_goCue_delays(trial_data),
        "_bpod_response_feedback_delays": load_response_feedback_delays(trial_data),
        "_bpod_response_stimFreeze_delays": load_response_stimFreeze_delays(trial_data),
        "_bpod_stimOff_itiIn_delays": load_stimOff_itiIn_delays(trial_data),
        "_bpod_positive_feedback_stimOff_delays": load_positive_feedback_stimOff_delays(trial_data),
        "_bpod_negative_feedback_stimOff_delays": load_negative_feedback_stimOff_delays(trial_data),
        "_bpod_valve_pre_trial": load_valve_pre_trial(trial_data),
        "_bpod_error_trial_event_sequence": load_error_trial_event_sequence(trial_data),
        "_bpod_correct_trial_event_sequence": load_correct_trial_event_sequence(trial_data),
        "_bpod_trial_length": load_trial_length(trial_data),
        # Wheel trial_data loading
        "_bpod_wheel_freeze_during_quiescence": load_wheel_freeze_during_quiescence(
            trial_data, session_path=session_path
        ),
        "_bpod_wheel_move_before_feedback": load_wheel_move_before_feedback(
            trial_data, session_path=session_path
        ),
        "_bpod_wheel_move_during_closed_loop": load_wheel_move_during_closed_loop(
            trial_data, session_path=session_path, gain=gain
        ),
        # Bpod fronts loading
        "_bpod_stimulus_move_before_goCue": load_stimulus_move_before_goCue(trial_data, BNC1=BNC1),
        "_bpod_audio_pre_trial": load_audio_pre_trial(trial_data, BNC2=BNC2),
    }
    return qcmetrics_frame


# SINGLE METRICS
# ---------------------------------------------------------------------------- #
def load_stimOn_goCue_delays(trial_data):
    """ 1. StimOn and GoCue and should be within a 10 ms of each other on 99% of trials
    Variable name: stimOn_goCue_delays
    Metric: stimOn_times - goCue_times
    Criteria: (M<10 ms for 99%) of trials AND (M > 0 ms for 99% of trials)
    """
    metric = trial_data["goCue_times"] - trial_data["stimOn_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 0.01) & (metric[~nans] > 0)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_response_feedback_delays(trial_data):
    """ 2. response_time and feedback_time
    Variable name: response_feedback_delays
    Metric: Feedback_time - response_time
    Criterion: (M <10 ms for 99% of trials) AND ( M > 0 ms for 100% of trials)
    """
    metric = trial_data["feedback_times"] - trial_data["response_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = ((metric[~nans] < 0.01) & (metric[~nans] > 0)).astype(np.float)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_response_stimFreeze_delays(trial_data):
    """ 3. Stim freeze and response time
    Variable name: response_stimFreeze_delays
    Metric: stim_freeze - response_time
    Criterion: (M<100 ms for 99% of trials) AND (M > 0 ms for 100% of trials)
    """
    metric = trial_data["stimFreeze_times"] - trial_data["response_times"]
    # Find NaNs (if any of the values are nan operation will be nan)
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    # Test for valid values
    passed[~nans] = ((metric[~nans] < 0.1) & (metric[~nans] > 0)).astype(np.float)
    # Finally remove no_go trials (stimFreeze triggered differently in no_go trials)
    # should account for all the nans
    passed[trial_data["choice"] == 0] = np.nan
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_stimOff_itiIn_delays(trial_data):
    """ 4. Start of iti_in should be within a very small tolerance of the stim off
    Variable name: stimOff_itiIn_delays
    Metric: iti_in - stim_off
    Criterion: (M<10 ms for 99% of trials) AND (M > 0 ms for 99% of trials)
    """
    metric = trial_data["itiIn_times"] - trial_data["stimOff_times"]
    passed = ((metric < 0.01) & (metric >= 0)).astype(np.float)
    # Remove no_go trials (stimOff triggered differently in no_go trials)
    metric[trial_data["choice"] == 0] = np.nan
    passed[trial_data["choice"] == 0] = np.nan
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_wheel_freeze_during_quiescence(trial_data, wheel_data):
    """ 5. Wheel should not move more than 2 ticks each direction for at least 0.2 + 0.2-0.6
    amount of time (quiescent period; exact value in bpod['quiescence']) before go cue
    Variable name: wheel_freeze_during_quiescence
    Metric: abs(min(W - w_t0), max(W - w_t0)) where W is wheel pos over interval
    np.max(Metric) to get highest displaceente in any direction
    interval = [goCueTrigger_time-quiescent_duration,goCueTrigger_time]
    Criterion: <2 degrees for 99% of trials
    """
    assert np.all(np.diff(wheel_data["re_ts"]) > 0)
    assert trial_data["quiescence"].size == trial_data["goCueTrigger_times"].size
    # Get tuple of wheel times and positions over each trial's quiescence period
    qevt_start_times = trial_data["goCueTrigger_times"] - trial_data["quiescence"]
    traces = traces_by_trial(
        wheel_data["re_ts"],
        wheel_data["re_pos"],
        start=qevt_start_times,
        end=trial_data["goCueTrigger_times"],
    )

    # metric = np.zeros_like(trial_data['quiescence'])
    # for i, trial in enumerate(traces):
    #     pos = trial[1]
    #     if pos.size > 1:
    #         metric[i] = np.abs(pos.max() - pos.min())
    # -OR-
    metric = np.zeros((len(trial_data["quiescence"]), 2))  # (n_trials, n_directions)
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
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_wheel_move_before_feedback(trial_data, session_path=None):
    """ 6. Wheel should move within 100ms of feedback
    Variable name: wheel_move_before_feedback
    Metric: (w_t - 0.05) - (w_t + 0.05) where t = feedback_time
    Criterion: != 0 for 99% of non-NoGo trials
    """
    if session_path is None:
        log.warning("No session_path in function call, retruning None")
        return None
    # Load Bpod wheel trial_data
    wheel_data = get_wheel_position(session_path)
    assert np.all(np.diff(wheel_data["re_ts"]) > 0)
    # Get tuple of wheel times and positions within 100ms of feedback
    traces = traces_by_trial(
        wheel_data["re_ts"],
        wheel_data["re_pos"],
        start=trial_data["feedback_times"] - 0.05,
        end=trial_data["feedback_times"] + 0.05,
    )
    metric = np.zeros_like(trial_data["feedback_times"])
    # For each trial find the displacement
    for i, trial in enumerate(traces):
        pos = trial[1]
        if pos.size > 1:
            metric[i] = pos[-1] - pos[0]

    # except no-go trials
    metric[trial_data["choice"] == 0] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan

    passed[~nans] = (metric[~nans] != 0).astype(np.float)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_wheel_move_during_closed_loop(trial_data, session_path=None, gain=None):
    """ Wheel should move a sufficient amount during the closed-loop period
    Variable name: wheel_move_during_closed_loop
    Metric: abs(w_resp - w_t0) - threshold_displacement, where w_resp = position at response
      time, w_t0 = position at go cue time, threshold_displacement = displacement required to move
      35 visual degrees
    Criterion: displacement < 1 visual degree for 99% of non-NoGo trials
    """
    if session_path is None:
        log.warning("No session_path input in function call, retruning None")
        return None
    if gain is None:
        log.warning("No gain input in function call, retruning None")
        return None

    # Load Bpod wheel trial_data
    wheel_data = get_wheel_position(session_path)
    assert np.all(np.diff(wheel_data["re_ts"]) > 0)

    # Get tuple of wheel times and positions over each trial's closed-loop period
    traces = traces_by_trial(
        wheel_data["re_ts"],
        wheel_data["re_pos"],
        start=trial_data["goCueTrigger_times"],
        end=trial_data["response_times"],
    )

    metric = np.zeros_like(trial_data["feedback_times"])
    # For each trial find the absolute displacement
    for i, trial in enumerate(traces):
        t, pos = trial
        # Find the position of the preceding sample and subtract it
        origin = wheel_data["re_pos"][wheel_data["re_ts"] < t[0]][-1]
        if pos.size > 0:
            metric[i] = np.abs(pos - origin).max()

    # Load gain and thresholds for each trial
    gain = np.array([gain] * len(trial_data["position"]))
    thresh = trial_data["position"]
    # abs displacement, s, in mm required to move 35 visual degrees
    s_mm = np.abs(thresh / gain)  # don't care about direction
    criterion = cm_to_rad(s_mm * 1e-1)  # convert abs displacement to radians (wheel pos is in rad)
    metric = metric - criterion  # difference should be close to 0
    rad_per_deg = cm_to_rad(1 / gain * 1e-1)
    passed = (np.abs(metric) < rad_per_deg).astype(np.float)  # less than 1 visual degree off
    metric[trial_data["choice"] == 0] = np.nan  # except no-go trials
    passed[trial_data["choice"] == 0] = np.nan  # except no-go trials
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_positive_feedback_stimOff_delays(trial_data):
    """ 8. Delay between valve and stim off should be 1s
    Variable name: positive_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 1s)
    Criterion: <150 ms on 99% of correct trials
    """
    metric = np.abs(trial_data["stimOff_times"] - trial_data["feedback_times"] - 1)
    metric[~trial_data["correct"]] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 0.15).astype(np.float)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_negative_feedback_stimOff_delays(trial_data):
    """ 9.Delay between noise and stim off should be 2 second
    Variable name: negative_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 2s)
    Criterion: <150 ms on 99% of incorrect trials
    """
    metric = np.abs(trial_data["stimOff_times"] - trial_data["errorCue_times"] - 2)
    # Find NaNs (if any of the values are nan operation will be nan)
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    # Apply criteria
    passed[~nans] = (metric[~nans] < 0.15).astype(np.float)
    # Remove no negative feedback trials
    metric[~trial_data["outcome"] == -1] = np.nan
    passed[~trial_data["outcome"] == -1] = np.nan
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


# def load_0(trial_data, session_path=None):
#     """ 10. Number of Bonsai command to change screen should match
#     Number of state change of frame2ttl
#     Variable name: syncSquare
#     Metric: (count of bonsai screen updates) - (count of frame2ttl)
#     Criterion: 0 on 99% of trials
#     """
#     pass


def load_valve_pre_trial(trial_data):
    """ 11. No valve outputs between trialstart_time and gocue_time-20 ms
    Variable name: valve_pre_trial
    Metric: Check if valve events exist between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    metric = trial_data["valveOpen_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    # Apply criteria
    passed[~nans] = ~(metric[~nans] < (trial_data["goCue_times"][~nans] - 0.02))
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


# Sequence of events:
def load_error_trial_event_sequence(trial_data):
    """ 13. on incorrect / miss trials : 2 audio events, 2 Bpod events (trial start, ITI)
    Variable name: error_trial_event_sequence
    Metric: Bpod (trial start) > audio (go cue) > audio (wrong) > Bpod (ITI)
    Criterion: All three boolean comparisons true on 99% of trials
    XXX: figure out single metric to use
    """
    a = np.less(
        trial_data["intervals_0"],
        trial_data["goCue_times"],
        where=(~np.isnan(trial_data["intervals_0"]) & ~np.isnan(trial_data["goCue_times"])),
    )
    b = np.less(
        trial_data["goCue_times"],
        trial_data["errorCue_times"],
        where=(~np.isnan(trial_data["goCue_times"]) & ~np.isnan(trial_data["errorCue_times"])),
    )
    c = np.less(
        trial_data["errorCue_times"],
        trial_data["itiIn_times"],
        where=(~np.isnan(trial_data["errorCue_times"]) & ~np.isnan(trial_data["itiIn_times"])),
    )
    metric = a & b & c
    metric = np.float64(metric)
    # Look only at incorrect or missed trials
    metric[trial_data["correct"]] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = metric[~nans]
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_correct_trial_event_sequence(trial_data):
    """ 14. on correct trials : 1 audio events, 3 Bpod events (valve open, trial start, ITI)
    (ITI task version dependent on ephys)
    Variable name: correct_trial_event_sequence
    Metric: Bpod (trial start) > audio (go cue) > Bpod (valve) > Bpod (ITI)
    Criterion: All three boolean comparisons true on 99% of trials
    XXX: figure out single metric to use
    """
    a = np.less(
        trial_data["intervals_0"],
        trial_data["goCue_times"],
        where=(~np.isnan(trial_data["intervals_0"]) & ~np.isnan(trial_data["goCue_times"])),
    )
    b = np.less(
        trial_data["goCue_times"],
        trial_data["valveOpen_times"],
        where=(~np.isnan(trial_data["goCue_times"]) & ~np.isnan(trial_data["valveOpen_times"])),
    )
    c = np.less(
        trial_data["valveOpen_times"],
        trial_data["itiIn_times"],
        where=(~np.isnan(trial_data["valveOpen_times"]) & ~np.isnan(trial_data["itiIn_times"])),
    )
    metric = a & b & c
    metric = np.float64(metric)
    # Look only at correct trials
    metric[~trial_data["correct"]] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = metric[~nans]
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_trial_length(trial_data):
    """ 15. Time between goCue and feedback <= 60s
    Variable name: trial_length
    Metric: (feedback_time - gocue_time)
    Criteria: M < 60.1 s AND M > 0 s both (true on 99% of trials)
    """
    metric = trial_data["feedback_times"] - trial_data["goCue_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 60.1) & (metric[~nans] > 0)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


# def load_1(trial_data, session_path=None):
#     """ 16. Between go tone and feedback, frame2ttl should be changing at ~60Hz
#     if wheel moves (exact frequency depending on velocity)
#     Variable name:
#     Metric:
#     Criterion:
#     """
#     pass


# Trigger response checks
def load_goCue_delays(trial_data):
    """ 25.Trigger response difference
    Variable name: goCue_delays
    Metric: goCue_times - goCueTrigger_times
    Criterion: 99% <= 1.5ms
    """
    metric = trial_data["goCue_times"] - trial_data["goCueTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = metric[~nans] <= 0.0015
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_errorCue_delays(trial_data):
    """ 26.Trigger response difference
    Variable name: errorCue_delays
    Metric: errorCue_times - errorCueTrigger_times
    Criterion: 99% <= 1.5ms
    """
    metric = trial_data["errorCue_times"] - trial_data["errorCueTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = metric[~nans] <= 0.0015
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_stimOn_delays(trial_data):
    """ 27. Trigger response difference
    Variable name: stimOn_delays
    Metric: stimOn_times - stiomOnTrigger_times
    Criterion: 99% <  150ms
    """
    metric = trial_data["stimOn_times"] - trial_data["stimOnTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_stimOff_delays(trial_data):
    """ 28.Trigger response difference
    Variable name: stimOff_delays
    Metric: stimOff_times - stimOffTrigger_times
    Criterion:99% <  150ms
    """
    metric = trial_data["stimOff_times"] - trial_data["stimOffTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = metric[~nans] <= 0.15
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_stimFreeze_delays(trial_data):
    """ 29.Trigger response difference
    Variable name: stimFreeze_delays
    Metric: stimFreeze_times - stimFreezeTrigger_times
    Criterion: 99% <  150ms
    """
    metric = trial_data["stimFreeze_times"] - trial_data["stimFreezeTrigger_times"]
    passed = np.zeros_like(metric) * np.nan
    passed[~np.isnan(metric)] = metric[~np.isnan(metric)] <= 0.15
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_reward_volumes(trial_data):
    """ xx.Reward volume tests
    Variable name: rewardVolume
    Metric: len(set(rewardVolume)) <= 2 & np.all(rewardVolume <= 3)
    Criterion: 100%
    """
    metric = trial_data["rewardVolume"]
    val = np.min(np.unique(np.nonzero(metric)))
    vals = np.ones(len(metric)) * val
    passed = ((metric >= 1.5) & (metric == vals) & (metric <= 3)).astype(np.float)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_stimulus_move_before_goCue(trial_data, BNC1=None):
    """ 7. No stimulus movements between trialstart_time and gocue_time-20 ms
    Variable name: stimulus_move_before_goCue
    Metric: count of any stimulus change events between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    if BNC1 is None:
        log.warning("No BNC1 input in function call, retruning None")
        return None
    s = BNC1["times"]
    metric = np.array([])
    for i, c in zip(trial_data["intervals_0"], trial_data["goCue_times"]):
        metric = np.append(metric, np.count_nonzero(s[s > i] < (c - 0.02)))

    passed = (metric == 0).astype(np.float)
    # Remove no go trials
    passed[trial_data["choice"] == 0] = np.nan
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_audio_pre_trial(trial_data, BNC2=None):
    """ 12. No audio outputs between trialstart_time and gocue_time-20 ms
    Variable name: audio_pre_trial
    Metric: Check if audio events exist between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    if BNC2 is None:
        log.warning("No BNC2 input in function call, retruning None")
        return None
    s = BNC2["times"]
    metric = np.array([], dtype=np.bool)
    for i, c in zip(trial_data["intervals_0"], trial_data["goCue_times"]):
        metric = np.append(metric, np.any(s[s > i] < (c - 0.02)))
    passed = (~metric).astype(np.float)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


class BpodQCMetrics(object):
    def __init__(self, eid, one=None, trial_data=None):
        self.eid = eid
        self.one = one or ONE(printout=False)
        self.session_path = self.one.path_from_eid(eid)
        self.trial_data = trial_data

        self.metrics = None
        self.passed = None

    def load_raw_data(self):
        self.trial_data = bpodqc.load_bpod_data(self.eid, fpga_time=False)
        self.BNC1, self.BNC2 = bpodqc.get_bpod_fronts(self.session_path)
        self.gain = check_parse_json(self.one.alyx.rest("sessions", "read", id=eid)["json"])[
            "STIM_GAIN"
        ]

    def compute_metrics(self):

        self.metrics = None
        return

    def apply_criteria(self):
        pass

    def get_extended_qc_frame(self):
        pass

    def patch_alyx_extended_qc(self, frame):
        pass


if __name__ == "__main__":
    from pyinstrument import Profiler

    eid, det = random_ephys_session("churchlandlab")
    trial_data = bpodqc.load_bpod_data(eid, fpga_time=False)

    profiler = Profiler()
    profiler.start()

    # code you want to profile
    metrics = get_bpodqc_metrics_frame(eid, trial_data=trial_data)
    criteria = get_bpodqc_metrics_frame(eid, trial_data=trial_data, apply_criteria=True)
    mean_criteria = {k: np.nanmean(v) for k, v in criteria.items()}

    profiler.stop()

    print(profiler.output_text(unicode=True, color=True))

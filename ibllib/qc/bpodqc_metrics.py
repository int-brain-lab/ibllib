"""Behaviour QC
This module runs a list of quality control metrics on the behaviour data.

Examples:
    # Running on a rig computer and updating QC fields in Alyx:
    from ibllib.qc.bpodqc_metrics import BpodQC
    BpodQC('path/to/session').run(update=True)

    # Downloading the required data and inspecting the QC on a different computer:
    from ibllib.qc.bpodqc_metrics import BpodQC
    qc = BpodQC(eid, download_data=True)
    outcome, results = qc.run()

    # Inspecting individual test outcomes
    from ibllib.qc.bpodqc_metrics import BpodQC
    qc = BpodQC(eid, download_data=True)
    outcome, results, outcomes = qc.compute().compute_session_status()
"""
import logging

import numpy as np

from brainbox.behavior.wheel import cm_to_rad, traces_by_trial
from ibllib.qc.bpodqc_extractors import BpodQCExtractor
from ibllib.io.extractors.training_wheel import WHEEL_RADIUS_CM
from ibllib.io.extractors.ephys_fpga import WHEEL_TICKS
from . import base

log = logging.getLogger('ibllib')


class BpodQC(base.QC):
    def __init__(self, session_path_or_eid, one=None, download_data=False):
        super().__init__(session_path_or_eid, one, log=log)

        # Data
        self.download_data = download_data
        self.extractor = None

        # Metrics and passed trials
        self.metrics = None
        self.passed = None
        self.criteria = {"PASS": 0.99,
                         "WARNING": 0.95,
                         "FAIL": 0}

    def load_data(self):
        self.extractor = BpodQCExtractor(
            self.session_path, one=self.one, ensure_data=self.download_data)

    def compute(self):
        """Compute and store the QC metrics
        Runs the QC on the session and stores a map of the metrics for each datapoint for each
        test, and a map of which datapoints passed for each test
        :return:
        """
        if self.extractor is None:
            self.load_data()
        self.log.info(f"Session {self.session_path}: Running QC on behavior data...")
        self.metrics, self.passed = get_bpodqc_metrics_frame(
            self.extractor.data,
            self.extractor.settings["STIM_GAIN"],  # The wheel gain
            self.extractor.BNC1,
            self.extractor.BNC2,
            self.extractor.wheel_encoding or 'X1'
        )
        return

    def run(self, update=False):
        if self.metrics is None:
            self.compute()
        self.outcome, results, _ = self.compute_session_status()
        if update:
            self.update_extended_qc(results)
            self.update(self.outcome, 'behavior')
        return self.outcome, results

    def compute_session_status(self):
        """
        :return: Overall session QC outcome as a string
        :return: A map of QC tests and the proportion of data points that passed them
        :return: A map of QC tests and their outcomes
        """
        MAX_BOUND, MIN_BOUND = (1, 0)
        results = {k: np.nanmean(v) for k, v in self.passed.items()}

        # Ensure criteria are in order
        criteria = self.criteria.items()
        criteria = {k: v for k, v in sorted(criteria, key=lambda x: x[1], reverse=True)}
        indices = []

        for v in results.values():
            if v is None or np.isnan(v):
                indices.append(int(-1))
            elif (v > MAX_BOUND) or (v < MIN_BOUND):
                raise ValueError("Values out of bound")
            else:
                passed = v >= np.fromiter(criteria.values(), dtype=float)
                indices.append(int(np.argmax(passed)))

        key_map = lambda x: 'NOT_SET' if x < 0 else list(criteria.keys())[x]
        # Criteria map is in order of severity so the max index is our overall QC outcome
        session_outcome = key_map(max(indices))
        outcomes = dict(zip(results.keys(), map(key_map, indices)))

        return session_outcome, results, outcomes


def get_bpodqc_metrics_frame(data, wheel_gain, BNC1, BNC2, re_encoding='X1'):
    """Plottable metrics based on timings"""

    qcmetrics_frame = {
        "_bpod_goCue_delays": check_goCue_delays(data),
        "_bpod_errorCue_delays": check_errorCue_delays(data),
        "_bpod_stimOn_delays": check_stimOn_delays(data),
        "_bpod_stimOff_delays": check_stimOff_delays(data),
        "_bpod_stimFreeze_delays": check_stimFreeze_delays(data),
        "_bpod_stimOn_goCue_delays": check_stimOn_goCue_delays(data),
        "_bpod_response_feedback_delays": check_response_feedback_delays(data),
        "_bpod_response_stimFreeze_delays": check_response_stimFreeze_delays(data),
        "_bpod_stimOff_itiIn_delays": check_stimOff_itiIn_delays(data),
        "_bpod_positive_feedback_stimOff_delays": check_positive_feedback_stimOff_delays(data),
        "_bpod_negative_feedback_stimOff_delays": check_negative_feedback_stimOff_delays(data),
        "_bpod_valve_pre_trial": check_valve_pre_trial(data),
        "_bpod_error_trial_event_sequence": check_error_trial_event_sequence(data),
        "_bpod_correct_trial_event_sequence": check_correct_trial_event_sequence(data),
        "_bpod_trial_length": check_trial_length(data),
        # Wheel trial_data loading
        "_bpod_wheel_integrity": check_wheel_integrity(data, re_encoding),
        "_bpod_wheel_freeze_during_quiescence": check_wheel_freeze_during_quiescence(data),
        "_bpod_wheel_move_before_feedback": check_wheel_move_before_feedback(data),
        "_bpod_wheel_move_during_closed_loop": check_wheel_move_during_closed_loop(
            data, wheel_gain
        ),
        # Bpod fronts loading
        "_bpod_stimulus_move_before_goCue": check_stimulus_move_before_goCue(data, BNC1=BNC1),
        "_bpod_audio_pre_trial": check_audio_pre_trial(data, BNC2=BNC2),
    }
    # Split metrics and passed frames
    metrics = {}
    passed = {}
    for k in qcmetrics_frame:
        metrics[k], passed[k] = qcmetrics_frame[k]
    return metrics, passed


# SINGLE METRICS
# ---------------------------------------------------------------------------- #
def check_stimOn_goCue_delays(data):
    """ StimOn and GoCue and should be within a 10 ms of each other on 99% of trials
    Variable name: stimOn_goCue_delays
    Metric: stimOn_times - goCue_times
    Criteria: 0 < M < 10 ms for 99% of trials
    """
    metric = data["goCue_times"] - data["stimOn_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 0.01) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_response_feedback_delays(data):
    """ response_time and feedback_time
    Variable name: response_feedback_delays
    Metric: Feedback_time - response_time
    Criterion: 0 < M < 10 ms for 99% of trials
    """
    metric = data["feedback_times"] - data["response_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = ((metric[~nans] < 0.01) & (metric[~nans] > 0)).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_response_stimFreeze_delays(data):
    """ Stim freeze and response time
    Variable name: response_stimFreeze_delays
    Metric: stim_freeze - response_time
    Criterion: 0 < M < 100 ms for 99% of trials
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
    return metric, passed


def check_stimOff_itiIn_delays(data):
    """ Start of iti_in should be within a very small tolerance of the stim off
    Variable name: stimOff_itiIn_delays
    Metric: iti_in - stim_off
    Criterion: 0 < M < 10 ms for 99% of trials
    """
    metric = data["itiIn_times"] - data["stimOff_times"]
    passed = valid = ~np.isnan(metric)
    passed[valid] = ((metric[valid] < 0.01) & (metric[valid] >= 0)).astype(np.float)
    # Remove no_go trials (stimOff triggered differently in no_go trials)
    metric[data["choice"] == 0] = np.nan
    passed[data["choice"] == 0] = np.nan
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_wheel_freeze_during_quiescence(data):
    """ Wheel should not move more than 2 ticks each direction for at least 0.2 + 0.2-0.6
    amount of time (quiescent period; exact value in bpod['quiescence']) before go cue
    Variable name: wheel_freeze_during_quiescence
    Metric: abs(min(W - w_t0), max(W - w_t0)) where W is wheel pos over interval
    np.max(Metric) to get highest displaceente in any direction
    interval = [goCueTrigger_time-quiescent_duration,goCueTrigger_time]
    Criterion: <2 degrees for 99% of trials
    """
    assert np.all(np.diff(data["wheel_timestamps"]) > 0)
    assert data["quiescence"].size == data["stimOnTrigger_times"].size
    # Get tuple of wheel times and positions over each trial's quiescence period
    qevt_start_times = data["stimOnTrigger_times"] - data["quiescence"]
    traces = traces_by_trial(
        data["wheel_timestamps"],
        data["wheel_position"],
        start=qevt_start_times,
        end=data["stimOnTrigger_times"]
    )

    metric = np.zeros((len(data["quiescence"]), 2))  # (n_trials, n_directions)
    for i, trial in enumerate(traces):
        t, pos = trial
        # Get the last position before the period began
        if pos.size > 0:
            # Find the position of the preceding sample and subtract it
            idx = np.abs(data["wheel_timestamps"] - t[0]).argmin() - 1
            origin = data["wheel_position"][idx if idx != -1 else 0]
            # Find the absolute min and max relative to the last sample
            metric[i, :] = np.abs([np.min(pos - origin), np.max(pos - origin)])
    # Reduce to the largest displacement found in any direction
    metric = np.max(metric, axis=1)
    metric = 180 * metric / np.pi  # convert to degrees from radians
    criterion = 2  # Position shouldn't change more than 2 in either direction
    passed = (metric < criterion).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_wheel_move_before_feedback(data):
    """ Wheel should move within 100ms of feedback
    Variable name: wheel_move_before_feedback
    Metric: (w_t - 0.05) - (w_t + 0.05) where t = feedback_time
    Criterion: != 0 for 99% of non-NoGo trials
    """
    # Get tuple of wheel times and positions within 100ms of feedback
    traces = traces_by_trial(
        data["wheel_timestamps"],
        data["wheel_position"],
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
    return metric, passed


def check_wheel_move_during_closed_loop(data, wheel_gain):
    """ Wheel should move a sufficient amount during the closed-loop period
    Variable name: wheel_move_during_closed_loop
    Metric: abs(w_resp - w_t0) - threshold_displacement, where w_resp = position at response
        time, w_t0 = position at go cue time, threshold_displacement = displacement required to
        move 35 visual degrees
    Criterion: displacement < 1 visual degree for 99% of non-NoGo trials
    """
    if wheel_gain is None:
        log.warning("No wheel_gain input in function call, returning None")
        return None

    # Get tuple of wheel times and positions over each trial's closed-loop period
    traces = traces_by_trial(
        data["wheel_timestamps"],
        data["wheel_position"],
        start=data["goCueTrigger_times"],
        end=data["response_times"],
    )

    metric = np.zeros_like(data["feedback_times"])
    # For each trial find the absolute displacement
    for i, trial in enumerate(traces):
        t, pos = trial
        if pos.size != 0:
            # Find the position of the preceding sample and subtract it
            idx = np.abs(data["wheel_timestamps"] - t[0]).argmin() - 1
            origin = data["wheel_position"][idx]
            metric[i] = np.abs(pos - origin).max()

    # Load wheel_gain and thresholds for each trial
    wheel_gain = np.array([wheel_gain] * len(data["position"]))
    thresh = data["position"]
    # abs displacement, s, in mm required to move 35 visual degrees
    s_mm = np.abs(thresh / wheel_gain)  # don't care about direction
    criterion = cm_to_rad(s_mm * 1e-1)  # convert abs displacement to radians (wheel pos is in rad)
    metric = metric - criterion  # difference should be close to 0
    rad_per_deg = cm_to_rad(1 / wheel_gain * 1e-1)
    passed = (np.abs(metric) < rad_per_deg).astype(np.float)  # less than 1 visual degree off
    metric[data["choice"] == 0] = np.nan  # except no-go trials
    passed[data["choice"] == 0] = np.nan  # except no-go trials
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_positive_feedback_stimOff_delays(data):
    """ Delay between valve and stim off should be 1s
    Variable name: positive_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 1s)
    Criterion: M < 150 ms on 99% of correct trials
    """
    metric = np.abs(data["stimOff_times"] - data["feedback_times"] - 1)
    metric[~data["correct"]] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 0.15).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_negative_feedback_stimOff_delays(data):
    """ Delay between noise and stim off should be 2 second
    Variable name: negative_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 2s)
    Criterion: M < 150 ms on 99% of incorrect trials
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
    return metric, passed


# def load_0(trial_data, session_path=None):
#     """ Number of Bonsai command to change screen should match
#     Number of state change of frame2ttl
#     Variable name: syncSquare
#     Metric: (count of bonsai screen updates) - (count of frame2ttl)
#     Criterion: 0 on 99% of trials
#     """
#     pass


def check_valve_pre_trial(data):
    """ No valve outputs between trialstart_time and gocue_time-20 ms
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
    return metric, passed


# Sequence of events:
def check_error_trial_event_sequence(data):
    """ on incorrect / miss trials : 2 audio events, 2 Bpod events (trial start, ITI)
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
    return metric, passed


def check_correct_trial_event_sequence(data):
    """ On correct trials : 1 audio events, 3 Bpod events (valve open, trial start, ITI)
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
    return metric, passed


def check_trial_length(data):
    """ Time between goCue and feedback <= 60s
    Variable name: trial_length
    Metric: (feedback_time - gocue_time)
    Criteria: M < 60.1 s AND M > 0 s both (true on 99% of trials)
    """
    metric = data["feedback_times"] - data["goCue_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 60.1) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


# def load_1(trial_data, session_path=None):
#     """ Between go tone and feedback, frame2ttl should be changing at ~60Hz
#     if wheel moves (exact frequency depending on velocity)
#     Variable name:
#     Metric:
#     Criterion:
#     """
#     pass


# Trigger response checks
def check_goCue_delays(data):
    """ Trigger response difference
    Variable name: goCue_delays
    Metric: goCue_times - goCueTrigger_times
    Criterion: 0 < M <= 1ms for 99% of trials
    """
    metric = data["goCue_times"] - data["goCueTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.0015) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_errorCue_delays(data):
    """ Trigger response difference
    Variable name: errorCue_delays
    Metric: errorCue_times - errorCueTrigger_times
    Criterion: 0 < M <= 1ms for 99% of trials
    """
    metric = data["errorCue_times"] - data["errorCueTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.0015) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_stimOn_delays(data):
    """ Trigger response difference
    Variable name: stimOn_delays
    Metric: stimOn_times - stiomOnTrigger_times
    Criterion: 0 < M < 150ms for 99% of trials
    """
    metric = data["stimOn_times"] - data["stimOnTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_stimOff_delays(data):
    """ Trigger response difference
    Variable name: stimOff_delays
    Metric: stimOff_times - stimOffTrigger_times
    Criterion: 0 < M < 150ms for 99% of trials
    """
    metric = data["stimOff_times"] - data["stimOffTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_stimFreeze_delays(data):
    """ Trigger response difference
    Variable name: stimFreeze_delays
    Metric: stimFreeze_times - stimFreezeTrigger_times
    Criterion: 0 < M < 150ms for 99% of trials
    """
    metric = data["stimFreeze_times"] - data["stimFreezeTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_reward_volumes(data):
    """ Reward volume tests
    Variable name: rewardVolume
    Metric: len(set(rewardVolume)) <= 2 & np.all(rewardVolume <= 3)
    Criterion: 100%
    """
    metric = data["rewardVolume"]
    val = np.min(np.unique(np.nonzero(metric)))
    vals = np.ones(len(metric)) * val
    passed = ((metric >= 1.5) & (metric == vals) & (metric <= 3)).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_stimulus_move_before_goCue(data, BNC1=None):
    """ No stimulus movements between trialstart_time and gocue_time-20 ms
    Variable name: stimulus_move_before_goCue
    Metric: count of any stimulus change events between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    if BNC1 is None:
        log.warning("No BNC1 input in function call, returning None")
        return None
    s = BNC1["times"]
    metric = np.array([])
    for i, c in zip(data["intervals_0"], data["goCue_times"]):
        metric = np.append(metric, np.count_nonzero(s[s > i] < (c - 0.02)))

    passed = (metric == 0).astype(np.float)
    # Remove no go trials
    passed[data["choice"] == 0] = np.nan
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_audio_pre_trial(data, BNC2=None):
    """ No audio outputs between trialstart_time and gocue_time-20 ms
    Variable name: audio_pre_trial
    Metric: Check if audio events exist between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    if BNC2 is None:
        log.warning("No BNC2 input in function call, retruning None")
        return None
    s = BNC2["times"]
    metric = np.array([], dtype=np.bool)
    for i, c in zip(data["intervals_0"], data["goCue_times"]):
        metric = np.append(metric, np.any(s[s > i] < (c - 0.02)))
    passed = (~metric).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_wheel_integrity(data, re_encoding='X1', enc_res=None):
    """
    Variable name: wheel_integrity
    Metric: (absolute difference of the positions - encoder resolution) + 1 if difference of
    timestamps <= 0
    Criterion: Close to zero for > 99% of samples
    :param data: dict of wheel data with keys ('wheel_timestamps', 'wheel_position')
    :param re_encoding: the encoding of the wheel data, X1, X2 or X4
    :param enc_res: the rotary encoder resolution
    """
    if isinstance(re_encoding, str):
        re_encoding = int(re_encoding[-1])
    if enc_res is None:
        enc_res = WHEEL_TICKS / re_encoding
    # The expected difference between samples in the extracted units
    resolution = (2 * np.pi / enc_res) * re_encoding * WHEEL_RADIUS_CM
    # We expect the difference of neighbouring positions to be close to the resolution
    pos_check = np.abs(np.diff(data['wheel_position'])) - resolution
    # Timestamps should be strictly increasing
    ts_check = np.diff(data['wheel_timestamps']) <= 0.
    metric = pos_check + ts_check.astype(float)  # all values should be close to zero
    passed = np.isclose(metric, np.zeros_like(metric))
    return metric, passed

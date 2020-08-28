"""Behaviour QC
This module runs a list of quality control metrics on the behaviour data.

Examples:
    # Running on a rig computer and updating QC fields in Alyx:
    from ibllib.qc.task_metrics import TaskQC
    TaskQC('path/to/session').run(update=True)

    # Downloading the required data and inspecting the QC on a different computer:
    from ibllib.qc.task_metrics import TaskQC
    qc = TaskQC(eid, download_data=True)
    outcome, results = qc.run()

    # Inspecting individual test outcomes
    from ibllib.qc.task_metrics import TaskQC
    qc = TaskQC(eid, download_data=True)
    outcome, results, outcomes = qc.compute().compute_session_status()

    # Running bpod QC on ephys session
    from ibllib.qc.task_metrics import TaskQC
    from ibllib.qc.task_extractors import TaskQCExtractor
    qc = TaskQC(eid, download_data=True)
    bpod = TaskQCExtractor(eid, lazy=True)
    bpod.data = bpod.extract_data(bpod_only=True)  # Extract without FPGA
    qc.extractor = bpod
    bpod_qc = qc.run()

"""
import logging

import numpy as np

from brainbox.behavior.wheel import cm_to_rad, traces_by_trial
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.io.extractors.training_wheel import WHEEL_RADIUS_CM
from ibllib.io.extractors.ephys_fpga import WHEEL_TICKS
from . import base

log = logging.getLogger('ibllib')


class TaskQC(base.QC):
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
        self.extractor = TaskQCExtractor(
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
        "_task_goCue_delays": check_goCue_delays(data),
        "_task_errorCue_delays": check_errorCue_delays(data),
        "_task_stimOn_delays": check_stimOn_delays(data),
        "_task_stimOff_delays": check_stimOff_delays(data),
        "_task_stimFreeze_delays": check_stimFreeze_delays(data),
        "_task_stimOn_goCue_delays": check_stimOn_goCue_delays(data),
        "_task_response_feedback_delays": check_response_feedback_delays(data),
        "_task_response_stimFreeze_delays": check_response_stimFreeze_delays(data),
        "_task_stimOff_itiIn_delays": check_stimOff_itiIn_delays(data),
        "_task_positive_feedback_stimOff_delays": check_positive_feedback_stimOff_delays(data),
        "_task_negative_feedback_stimOff_delays": check_negative_feedback_stimOff_delays(data),
        "_task_valve_pre_trial": check_valve_pre_trial(data),
        "_task_error_trial_event_sequence": check_error_trial_event_sequence(data),
        "_task_correct_trial_event_sequence": check_correct_trial_event_sequence(data),
        "_task_trial_length": check_trial_length(data),
        # Wheel trial_data loading
        "_task_wheel_integrity": check_wheel_integrity(data, re_encoding),
        "_task_wheel_freeze_during_quiescence": check_wheel_freeze_during_quiescence(data),
        "_task_wheel_move_before_feedback": check_wheel_move_before_feedback(data),
        "_task_wheel_move_during_closed_loop": check_wheel_move_during_closed_loop(
            data, wheel_gain
        ),
        # Bpod fronts loading
        "_task_stimulus_move_before_goCue": check_stimulus_move_before_goCue(data, BNC1=BNC1),
        "_task_audio_pre_trial": check_audio_pre_trial(data, BNC2=BNC2),
    }
    # Split metrics and passed frames
    metrics = {}
    passed = {}
    for k in qcmetrics_frame:
        metrics[k], passed[k] = qcmetrics_frame[k]
    return metrics, passed


# SINGLE METRICS
# ---------------------------------------------------------------------------- #

# === Delays between events checks ===

def check_stimOn_goCue_delays(data):
    """ Checks that the time difference between the onset of the visual stimulus
    and the onset of the go cue tone is positive and less than 10ms.
    Variable name: stimOn_goCue_delays
    Metric: M = stimOn_times - goCue_times
    Criteria: 0 < M < 0.010 s
    Units: seconds [s]
    """
    metric = data["goCue_times"] - data["stimOn_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 0.01) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_response_feedback_delays(data):
    """ Checks that the time difference between the response and the feedback onset
    (error sound or valve) is positive and less than 10ms.
    Variable name: response_feedback_delays
    Metric: M = Feedback_time - response_time
    Criterion: 0 < M < 0.010 s
    Units: seconds [s]
    """
    metric = data["feedback_times"] - data["response_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = ((metric[~nans] < 0.01) & (metric[~nans] > 0)).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_response_stimFreeze_delays(data):
    """ Checks that the time difference between the visual stimulus freezing and the
    response is positive and less than 100ms.
    Variable name: response_stimFreeze_delays
    Metric: M = stimFreeze_times - response_times
    Criterion: 0 < M < 0.100 s
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
    """ Check that the start of the trial interval is within 10ms of the visual stimulus turning off.
    Variable name: stimOff_itiIn_delays
    Metric: M = itiIn_times - stimOff_times
    Criterion: 0 < M < 0.010 s
    Units: seconds [s]
    """
    metric = data["itiIn_times"] - data["stimOff_times"]
    passed = valid = ~np.isnan(metric)
    passed[valid] = ((metric[valid] < 0.01) & (metric[valid] >= 0)).astype(np.float)
    # Remove no_go trials (stimOff triggered differently in no_go trials)
    metric[data["choice"] == 0] = np.nan
    passed[data["choice"] == 0] = np.nan
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_positive_feedback_stimOff_delays(data):
    """ Check the the time difference between the valve onset and the visual stimulus turning off
    is 1 ± 0.150 seconds.
    Variable name: positive_feedback_stimOff_delays
    Metric: M = abs((stimOff_times - feedback_times) - 1s)
    Criterion: M < 0.150 s
    Units: seconds [s] (absolute value TODO Change to get diff value with sign?)
    """
    metric = np.abs(data["stimOff_times"] - data["feedback_times"] - 1)
    metric[~data["correct"]] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 0.15).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_negative_feedback_stimOff_delays(data):
    """ Check the the time difference between the error sound and the visual stimulus
    turning off is 2 ± 0.150 seconds.
    Variable name: negative_feedback_stimOff_delays
    Metric: abs((stimOff_times - errorCue_times) - 2s)
    Criterion: M < 0.150 s
    Units: seconds [s] (absolute value TODO Change to get diff value with sign?)
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

# === Wheel movement during trial checks ===

def check_wheel_move_before_feedback(data):
    """ Check that the wheel does not move within 100ms of the feedback onset (error sound or valve).
    Variable name: wheel_move_before_feedback
    Metric: (w_t - 0.05) - (w_t + 0.05) where t = feedback_times
    Criterion: M != 0
    Units: TODO
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
    """ Check that the wheel moves by at least 35 degrees during the closed-loop period
    on trials where a feedback (error sound or valve) is delivered.
    Variable name: wheel_move_during_closed_loop
    Metric: M = abs(w_resp - w_t0) - threshold_displacement, where w_resp = position at response
        time, w_t0 = position at go cue time, threshold_displacement = displacement required to
        move 35 visual degrees
    Criterion: displacement < 1 visual degree
    Units: degrees
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


def check_wheel_freeze_during_quiescence(data):
    """ Check that the wheel does not move more than 2 ticks each direction for at least 0.2 + 0.2-0.6
    amount of time (quiescent period; exact value in bpod['quiescence']) before the go cue tone.
    Variable name: wheel_freeze_during_quiescence
    Metric: M = abs(min(W - w_t0), max(W - w_t0)) where W is wheel pos over interval
    Do  np.max(M)  to get the highest displacement in any direction
    interval = [goCueTrigger_time-quiescent_duration,goCueTrigger_time]
    Criterion: M < 2 degrees
    Units: degrees
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

# def load_0(trial_data, session_path=None):
#     """ Number of Bonsai command to change screen should match
#     Number of state change of frame2ttl
#     Variable name: syncSquare
#     Metric: (count of bonsai screen updates) - (count of frame2ttl)
#     Criterion: 0 on 99% of trials
#     """
#     pass

# === Sequence of events checks ===

def check_error_trial_event_sequence(data):
    """ Check that on incorrect / miss trials, there are exactly:
    2 audio events (go cue sound and error sound) and 2 Bpod events (ITI)
    TODO : This test does not seem to check for the above?
    And that the sequence of event is as expected:
    Bpod (trial start) > audio (go cue) > audio (error) > Bpod (ITI)
    Variable name: error_trial_event_sequence
    Metric: Bpod (trial start) > audio (go cue) > audio (error) > Bpod (ITI)
    Criterion: All three boolean comparisons true
    TODO: figure out single metric to use ; output unclear
    Units: boolean
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
    """ Check that on correct trials, there are exactly :
    1 audio events, 3 Bpod events (valve open, trial start, ITI)
    TODO : This test does not seem to check for the above?
    TODO explain comment: (ITI task version dependent on ephys)
    Variable name: correct_trial_event_sequence
    Metric: Bpod (trial start) > audio (go cue) > Bpod (valve) > Bpod (ITI)
    Criterion: All three boolean comparisons true
    TODO: figure out single metric to use ; output unclear
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
    """ Check that the time difference between the onset of the go cue sound
    and the feedback (error sound or valve) is positive and smaller than 60.1 s.
    Variable name: trial_length
    Metric: M = feedback_times - goCue_times
    Criteria: 0 < M < 60.1 s
    Units: seconds [s]
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


# === Trigger-response delay checks ===

def check_goCue_delays(data):
    """ Check that the time difference between the go cue sound being triggered and
    effectively played is smaller than 1ms.
    Variable name: goCue_delays
    Metric: goCue_times - goCueTrigger_times
    Criterion: 0 < M <= 0.001 s
    Units: seconds [s]
    """
    metric = data["goCue_times"] - data["goCueTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.0015) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_errorCue_delays(data):
    """ Check that the time difference between the error sound being triggered and
    effectively played is smaller than 1ms.
    Variable name: errorCue_delays
    Metric: errorCue_times - errorCueTrigger_times
    Criterion: 0 < M <= 0.001 s
    Units: seconds [s]
    """
    metric = data["errorCue_times"] - data["errorCueTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.0015) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_stimOn_delays(data):
    """ Check that the time difference between the visual stimulus onset-command being triggered
    and the stimulus effectively appearing on the screen is smaller than 150 ms.
    Variable name: stimOn_delays
    Metric: stimOn_times - stimOnTrigger_times
    Criterion: 0 < M < 0.150 s
    Units: seconds [s]
    """
    metric = data["stimOn_times"] - data["stimOnTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_stimOff_delays(data):
    """ Check that the time difference between the visual stimulus offset-command
    being triggered and the visual stimulus effectively turning off on the screen
    is smaller than 150 ms.
    Variable name: stimOff_delays
    Metric: stimOff_times - stimOffTrigger_times
    Criterion: 0 < M < 0.150 s
    Units: seconds [s]
    """
    metric = data["stimOff_times"] - data["stimOffTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_stimFreeze_delays(data):
    """ Check that the time difference between the visual stimulus freeze-command
    being triggered and the visual stimulus effectively freezing on the screen
    is smaller than 150 ms.
    Variable name: stimFreeze_delays
    Metric: stimFreeze_times - stimFreezeTrigger_times
    Criterion: 0 < M < 0.150 s
    Units: seconds [s]
    """
    metric = data["stimFreeze_times"] - data["stimFreezeTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


# === Data integrity checks ===

def check_reward_volumes(data):
    """ TODO review definition - Check that the reward volume is between 1.5 and 3 uL.
    TODO based on what is coded, is this a test at the session level then ?
    Variable name: rewardVolume
    TODO review definition - Metric: len(set(rewardVolume)) <= 2 & np.all(rewardVolume <= 3)
    Criterion: 100% TODO why not doing at trial level?
    """
    metric = data["rewardVolume"]
    val = np.min(np.unique(np.nonzero(metric)))
    vals = np.ones(len(metric)) * val
    passed = ((metric >= 1.5) & (metric == vals) & (metric <= 3)).astype(np.float)
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_wheel_integrity(data, re_encoding='X1', enc_res=None):
    """
    TODO definition
    Variable name: wheel_integrity
    Metric: M = (absolute difference of the positions - encoder resolution) + 1 if difference of
    timestamps <= 0
    TODO define accurately: Criterion: Close to zero for > 99% of samples
    :param data: dict of wheel data with keys ('wheel_timestamps', 'wheel_position')
    :param re_encoding: the encoding of the wheel data, X1, X2 or X4
    :param enc_res: the rotary encoder resolution
    TODO Units
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

# === Pre-trial checks ===

def check_valve_pre_trial(data):
    """ Check that there is no valve onset(s) between the start of the trial and
    the go cue sound onset - 20 ms.
    Variable name: valve_pre_trial
    Metric: M = number of valve event between trialstart_times and (goCue_times-20ms)
    Criterion: M = 0
    Units: -none-, integer
    """
    metric = data["valveOpen_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    # Apply criteria
    passed[~nans] = ~(metric[~nans] < (data["goCue_times"][~nans] - 0.02))
    assert len(data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def check_stimulus_move_before_goCue(data, BNC1=None):
    """ Check that there are no visual stimulus change(s) between the start of the trial and the
    go cue sound onset -20 ms.
    Variable name: stimulus_move_before_goCue
    Metric: M = number of visual stimulus change events between trialstart_time and (goCue_times-20ms)
    Criterion: M = 0
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
    """ Check that there are no audio outputs between the start of the trial and the
    go cue sound onset -20 ms.
    Metric: Check if audio events exist between trialstart_time and (goCue_times-20ms)
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

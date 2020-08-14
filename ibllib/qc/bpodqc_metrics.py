import logging
from pathlib import Path

import numpy as np
import pandas as pd

from brainbox.behavior.wheel import cm_to_rad, traces_by_trial
from ibllib.io.extractors.ephys_fpga import WHEEL_TICKS
from ibllib.io.extractors.training_wheel import WHEEL_RADIUS_CM
from ibllib.qc.base import QC
from ibllib.qc.bpodqc_extractors import BpodQCExtractor

log = logging.getLogger("ibllib")


class BpodQC(QC):
    def __init__(self, session_path_or_eid, one=None, ensure_data=False, lazy=False):
        super().__init__(session_path_or_eid, one, log=log)
        self.ensure_data = ensure_data
        self.lazy = lazy
        if self.ensure_data:
            self._ensure_required_data()

        # Data
        self.extractor = None
        # Utils
        self.wheel_gain = None
        self.bpod_ntrials = None
        self.wheel_trial_idxs = None

        # Metrics and passed trials
        self.metrics = None
        self.passed = None

        if not self.lazy:
            self.load_data()
            self.compute()

    def _ensure_required_data(self):
        dstypes = [
            "_iblrig_taskData.raw",
            "_iblrig_taskSettings.raw",
            "_iblrig_encoderPositions.raw",
            "_iblrig_encoderEvents.raw",
            "_iblrig_stimPositionScreen.raw",
            "_iblrig_syncSquareUpdate.raw",
            "_iblrig_encoderTrialInfo.raw",
            "_iblrig_ambientSensorData.raw",
        ]
        if (self.session_path is None) or (not Path(self.session_path).exists()):
            self.log.info(f"Downloading data for session {self.eid}")
            self.one.load(self.eid, dataset_types=dstypes, download_only=True)
            self.session_path = self.one.path_from_eid(self.eid)
            if self.session_path is None:
                self.lazy = True
                self.log.error("Data not found on server, can't calculate QC.")
        else:
            glob_sp = list(x.name for x in Path(self.session_path).rglob("*.raw.*") if x.is_file())
            if not all([x in glob_sp for x in dstypes]):
                self.log.warning(
                    f"Missing some datasets for session {self.eid} in path {self.session_path}"
                )
                self.log.info("Attempting download...")
                self.one.load(self.eid, dataset_types=dstypes, download_only=True)

    def load_data(self, lazy=False):
        self.extractor = BpodQCExtractor(self.session_path, lazy=lazy)
        self.wheel_gain = self.extractor.details["STIM_GAIN"]
        self.bpod_ntrials = len(self.extractor.raw_data)
        self.wheel_trial_idxs = BpodQC.hack_ts(
            self.extractor.wheel_data["re_ts"],
            self.extractor.trial_data["intervals_0"],
            self.extractor.trial_data["intervals_1"],
            idx=True,
        )
        return

    def compute(self):
        if self.extractor is None:
            self.load_data()
        self.log.info(f"Session {self.session_path}: Running QC on Bpod data...")
        self.metrics, self.passed = get_bpodqc_metrics_frame(
            self.extractor.trial_data,
            self.extractor.wheel_data,
            self.extractor.details["STIM_GAIN"],
            self.wheel_trial_idxs,
            self.extractor.BNC1,
            self.extractor.BNC2,
        )
        return

    @staticmethod
    def hack_ts(ts_array, intervals_0, intervals_1, idx=False):
        hacked_arr = []
        hacked_arr_idxs = []
        for start, end in zip(intervals_0, intervals_1):
            trial = ts_array[(ts_array >= start) & (ts_array < end)]
            trial_idx = np.where((ts_array >= start) & (ts_array < end))
            hacked_arr.append(trial)
            hacked_arr_idxs.extend(trial_idx)
        if np.max(hacked_arr_idxs[-1]) == len(ts_array):
            hacked_arr_idxs[-1] = np.setdiff1d(hacked_arr_idxs[-1], np.max(hacked_arr_idxs[-1]))
        return hacked_arr_idxs if idx else hacked_arr

    @property
    def metrics_df(self):
        if not self.metrics:
            log.error("Metrics frame not computed yet")
            return
        return BpodQC.frame_to_df(self.metrics)

    @property
    def passed_df(self):
        if not self.passed:
            log.error("Passed frame not computed yet")
            return
        return BpodQC.frame_to_df(self.passed)

    @staticmethod
    def frame_to_df(d: dict) -> pd.DataFrame:
        dd = d.copy()
        dd.pop("_bpod_wheel_integrity")
        out_df = pd.DataFrame.from_dict(dd)
        return out_df


def get_bpodqc_metrics_frame(trial_data, wheel_data, wheel_gain, wheel_trial_idxs, BNC1, BNC2):
    """Plottable metrics based on timings"""

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
        "_bpod_positive_feedback_stimOff_delays": load_positive_feedback_stimOff_delays(
            trial_data
        ),
        "_bpod_negative_feedback_stimOff_delays": load_negative_feedback_stimOff_delays(
            trial_data
        ),
        "_bpod_valve_pre_trial": load_valve_pre_trial(trial_data),
        "_bpod_error_trial_event_sequence": load_error_trial_event_sequence(trial_data),
        "_bpod_correct_trial_event_sequence": load_correct_trial_event_sequence(trial_data),
        "_bpod_trial_length": load_trial_length(trial_data),
        # Wheel trial_data loading
        "_bpod_wheel_integrity": load_wheel_integrity(wheel_data, trial_idxs=wheel_trial_idxs),
        "_bpod_wheel_freeze_during_quiescence": load_wheel_freeze_during_quiescence(
            trial_data, wheel_data
        ),
        "_bpod_wheel_move_before_feedback": load_wheel_move_before_feedback(
            trial_data, wheel_data
        ),
        "_bpod_wheel_move_during_closed_loop": load_wheel_move_during_closed_loop(
            trial_data, wheel_data, wheel_gain
        ),
        # Bpod fronts loading
        "_bpod_stimulus_move_before_goCue": load_stimulus_move_before_goCue(trial_data, BNC1=BNC1),
        "_bpod_audio_pre_trial": load_audio_pre_trial(trial_data, BNC2=BNC2),
    }
    # Split metrics and passed frames
    metrics = {}
    passed = {}
    for k in qcmetrics_frame:
        metrics[k], passed[k] = qcmetrics_frame[k]
    return metrics, passed


# SINGLE METRICS
# ---------------------------------------------------------------------------- #
def load_stimOn_goCue_delays(trial_data):
    """ StimOn and GoCue and should be within a 10 ms of each other on 99% of trials
    Variable name: stimOn_goCue_delays
    Metric: stimOn_times - goCue_times
    Criteria: 0 < M < 10 ms for 99% of trials
    """
    metric = trial_data["goCue_times"] - trial_data["stimOn_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 0.01) & (metric[~nans] > 0)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_response_feedback_delays(trial_data):
    """ response_time and feedback_time
    Variable name: response_feedback_delays
    Metric: Feedback_time - response_time
    Criterion: 0 < M < 10 ms for 99% of trials
    """
    metric = trial_data["feedback_times"] - trial_data["response_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = ((metric[~nans] < 0.01) & (metric[~nans] > 0)).astype(np.float)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_response_stimFreeze_delays(trial_data):
    """ Stim freeze and response time
    Variable name: response_stimFreeze_delays
    Metric: stim_freeze - response_time
    Criterion: 0 < M < 100 ms for 99% of trials
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
    """ Start of iti_in should be within a very small tolerance of the stim off
    Variable name: stimOff_itiIn_delays
    Metric: iti_in - stim_off
    Criterion: 0 < M < 10 ms for 99% of trials
    """
    metric = trial_data["itiIn_times"] - trial_data["stimOff_times"]
    passed = ((metric < 0.01) & (metric >= 0)).astype(np.float)
    # Remove no_go trials (stimOff triggered differently in no_go trials)
    metric[trial_data["choice"] == 0] = np.nan
    passed[trial_data["choice"] == 0] = np.nan
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_wheel_freeze_during_quiescence(trial_data, wheel_data):
    """ Wheel should not move more than 2 ticks each direction for at least 0.2 + 0.2-0.6
    amount of time (quiescent period; exact value in bpod['quiescence']) before go cue
    Variable name: wheel_freeze_during_quiescence
    Metric: abs(min(W - w_t0), max(W - w_t0)) where W is wheel pos over interval
    np.max(Metric) to get highest displaceente in any direction
    interval = [goCueTrigger_time-quiescent_duration,goCueTrigger_time]
    Criterion: <2 degrees for 99% of trials
    """
    assert np.all(np.diff(wheel_data["re_ts"]) > 0)
    assert trial_data["quiescence"].size == trial_data["stimOnTrigger_times"].size
    # Get tuple of wheel times and positions over each trial's quiescence period
    qevt_start_times = trial_data["stimOnTrigger_times"] - trial_data["quiescence"]
    traces = traces_by_trial(
        wheel_data["re_ts"],
        wheel_data["re_pos"],
        start=qevt_start_times,
        end=trial_data["stimOnTrigger_times"],
    )

    metric = np.zeros((len(trial_data["quiescence"]), 2))  # (n_trials, n_directions)
    for i, trial in enumerate(traces):
        t, pos = trial
        # Get the last position before the period began
        if pos.size > 0:
            # Find the position of the preceding sample and subtract it
            idx = np.abs(wheel_data["re_ts"] - t[0]).argmin() - 1
            origin = wheel_data["re_pos"][idx if idx != -1 else 0]
            # Find the absolute min and max relative to the last sample
            metric[i, :] = np.abs([np.min(pos - origin), np.max(pos - origin)])
    # Reduce to the largest displacement found in any direction
    metric = np.max(metric, axis=1)
    metric = 180 * metric / np.pi  # convert to degrees from radians
    criterion = 2  # Position shouldn't change more than 2 in either direction
    passed = (metric < criterion).astype(np.float)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_wheel_move_before_feedback(trial_data, wheel_data):
    """ Wheel should move within 100ms of feedback
    Variable name: wheel_move_before_feedback
    Metric: (w_t - 0.05) - (w_t + 0.05) where t = feedback_time
    Criterion: != 0 for 99% of non-NoGo trials
    """
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


def load_wheel_move_during_closed_loop(trial_data, wheel_data, wheel_gain):
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
        wheel_data["re_ts"],
        wheel_data["re_pos"],
        start=trial_data["goCueTrigger_times"],
        end=trial_data["response_times"],
    )

    metric = np.zeros_like(trial_data["feedback_times"])
    # For each trial find the absolute displacement
    for i, trial in enumerate(traces):
        t, pos = trial
        if pos.size != 0:
            # Find the position of the preceding sample and subtract it
            idx = np.abs(wheel_data["re_ts"] - t[0]).argmin() - 1
            origin = wheel_data["re_pos"][idx]
            metric[i] = np.abs(pos - origin).max()

    # Load wheel_gain and thresholds for each trial
    wheel_gain = np.array([wheel_gain] * len(trial_data["position"]))
    thresh = trial_data["position"]
    # abs displacement, s, in mm required to move 35 visual degrees
    s_mm = np.abs(thresh / wheel_gain)  # don't care about direction
    criterion = cm_to_rad(s_mm * 1e-1)  # convert abs displacement to radians (wheel pos is in rad)
    metric = metric - criterion  # difference should be close to 0
    rad_per_deg = cm_to_rad(1 / wheel_gain * 1e-1)
    passed = (np.abs(metric) < rad_per_deg).astype(np.float)  # less than 1 visual degree off
    metric[trial_data["choice"] == 0] = np.nan  # except no-go trials
    passed[trial_data["choice"] == 0] = np.nan  # except no-go trials
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_positive_feedback_stimOff_delays(trial_data):
    """ Delay between valve and stim off should be 1s
    Variable name: positive_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 1s)
    Criterion: M < 150 ms on 99% of correct trials
    """
    metric = np.abs(trial_data["stimOff_times"] - trial_data["feedback_times"] - 1)
    metric[~trial_data["correct"]] = np.nan
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] < 0.15).astype(np.float)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_negative_feedback_stimOff_delays(trial_data):
    """ Delay between noise and stim off should be 2 second
    Variable name: negative_feedback_stimOff_delays
    Metric: abs((stimoff_time - feedback_time) - 2s)
    Criterion: M < 150 ms on 99% of incorrect trials
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


def load_valve_pre_trial(trial_data):
    """ No valve outputs between trialstart_time and gocue_time-20 ms
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
    """ on incorrect / miss trials : 2 audio events, 2 Bpod events (trial start, ITI)
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
    """ On correct trials : 1 audio events, 3 Bpod events (valve open, trial start, ITI)
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
    """ Time between goCue and feedback <= 60s
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


# Trigger response checks
def load_goCue_delays(trial_data):
    """ Trigger response difference
    Variable name: goCue_delays
    Metric: goCue_times - goCueTrigger_times
    Criterion: 0 < M <= 1ms for 99% of trials
    """
    metric = trial_data["goCue_times"] - trial_data["goCueTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.0015) & (metric[~nans] > 0)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_errorCue_delays(trial_data):
    """ Trigger response difference
    Variable name: errorCue_delays
    Metric: errorCue_times - errorCueTrigger_times
    Criterion: 0 < M <= 1ms for 99% of trials
    """
    metric = trial_data["errorCue_times"] - trial_data["errorCueTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.0015) & (metric[~nans] > 0)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_stimOn_delays(trial_data):
    """ Trigger response difference
    Variable name: stimOn_delays
    Metric: stimOn_times - stiomOnTrigger_times
    Criterion: 0 < M < 150ms for 99% of trials
    """
    metric = trial_data["stimOn_times"] - trial_data["stimOnTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_stimOff_delays(trial_data):
    """ Trigger response difference
    Variable name: stimOff_delays
    Metric: stimOff_times - stimOffTrigger_times
    Criterion: 0 < M < 150ms for 99% of trials
    """
    metric = trial_data["stimOff_times"] - trial_data["stimOffTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_stimFreeze_delays(trial_data):
    """ Trigger response difference
    Variable name: stimFreeze_delays
    Metric: stimFreeze_times - stimFreezeTrigger_times
    Criterion: 0 < M < 150ms for 99% of trials
    """
    metric = trial_data["stimFreeze_times"] - trial_data["stimFreezeTrigger_times"]
    nans = np.isnan(metric)
    passed = np.zeros_like(metric) * np.nan
    passed[~nans] = (metric[~nans] <= 0.15) & (metric[~nans] > 0)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_reward_volumes(trial_data):
    """ Reward volume tests
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
    """ No stimulus movements between trialstart_time and gocue_time-20 ms
    Variable name: stimulus_move_before_goCue
    Metric: count of any stimulus change events between trialstart_time and (gocue_time-20ms)
    Criterion: 0 on 99% of trials
    """
    # FIXME: quiescence sync causes stim ove?
    if BNC1 is None:
        log.warning("No BNC1 input in function call, returning None")
        return None
    s = BNC1["times"]
    metric = np.array([])
    for i, c in zip(trial_data["intervals_0"], trial_data["goCue_times"]):
        metric = np.append(metric, np.count_nonzero(s[s >= i] < (c - 0.02)))

    passed = (metric == 0).astype(np.float)
    # Remove no go trials
    passed[trial_data["choice"] == 0] = np.nan
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_audio_pre_trial(trial_data, BNC2=None):
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
    for i, c in zip(trial_data["intervals_0"], trial_data["goCue_times"]):
        metric = np.append(metric, np.any(s[s > i] < (c - 0.02)))
    passed = (~metric).astype(np.float)
    assert len(trial_data["intervals_0"]) == len(metric) == len(passed)
    return metric, passed


def load_wheel_integrity(wheel_data, re_encoding="X1", enc_res=None, trial_idxs=None):
    """
    Variable name: wheel_integrity
    Metric: (absolute difference of the positions - encoder resolution) + 1 if difference of
    timestamps <= 0
    Criterion: Close to zero for > 99% of samples
    :param wheel_data: dict of wheel data with keys ('re_ts', 're_pos')
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
    # XXX: not necessarily, are we sure the only change allowed is of one tick?
    # what happens for "very fast" inputs? but should always be a multiple of it
    pos_check = np.abs(np.diff(wheel_data["re_pos"])) - resolution
    # Timestamps should be strictly increasing
    # XXX: Why not an assert?
    ts_check = np.diff(wheel_data["re_ts"]) <= 0.0
    # XXX: adding a bool to a metric is weird, metric here looks like the passed
    # Metric should be absolute diff of position, the rest is a criterion.
    metric = pos_check + ts_check.astype(float)  # all values should be close to zero
    passed = np.isclose(metric, np.zeros_like(metric))
    if trial_idxs is None:
        return metric, passed

    # hack metric and passed
    trial_metric = []
    trial_passed = []
    if np.max(trial_idxs[-1]) == len(metric):
        trial_idxs[-1] = np.setdiff1d(trial_idxs[-1], np.max(trial_idxs[-1]))

    for tr in trial_idxs:
        # one value per trial
        trial_metric.append(np.nanmean(metric[tr]))
        trial_passed.append(np.nanmean(passed[tr]))

    return trial_metric, trial_passed


# np.isclose(np.array([1,2,3]), np.array([1.1,2.1,3.1]),  )
# np.isclose()

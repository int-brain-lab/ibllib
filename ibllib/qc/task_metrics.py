"""Behaviour QC
This module runs a list of quality control metrics on the behaviour data.

Examples:
    # Running on a rig computer and updating QC fields in Alyx:
    from ibllib.qc.task_metrics import TaskQC
    TaskQC('path/to/session').run(update=True)

    # Downloading the required data and inspecting the QC on a different computer:
    from ibllib.qc.task_metrics import TaskQC
    qc = TaskQC(eid)
    outcome, results = qc.run()

    # Inspecting individual test outcomes
    from ibllib.qc.task_metrics import TaskQC
    qc = TaskQC(eid)
    outcome, results, outcomes = qc.compute().compute_session_status()

    # Running bpod QC on ephys session
    from ibllib.qc.task_metrics import TaskQC
    qc = TaskQC(eid)
    qc.load_data(bpod_only=True)  # Extract without FPGA
    bpod_qc = qc.run()

    # Running bpod QC only, from training rig PC
    from ibllib.qc.task_metrics import TaskQC
    from ibllib.qc.qcplots import plot_results
    session_path = r'/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_023/2020-02-14/001'
    qc = TaskQC(session_path)
    qc.load_data(bpod_only=True, download_data=False)  # Extract without FPGA
    qc.run()
    plot_results(qc, save_path=session_path)

    # Running ephys QC, from local server PC (after ephys + bpod data have been copied to a same
    folder)
    from ibllib.qc.task_metrics import TaskQC
    from ibllib.qc.qcplots import plot_results
    session_path = r'/home/nico/Downloads/FlatIron/mrsicflogellab/Subjects/SWC_023/2020-02-14/001'
    qc = TaskQC(session_path)
    qc.run()
    plot_results(qc, save_path=session_path)
"""
import logging
import sys
from datetime import datetime, timedelta
from inspect import getmembers, isfunction
from functools import reduce
from collections.abc import Sized

import numpy as np
from scipy.stats import chisquare

from brainbox.behavior.wheel import cm_to_rad, traces_by_trial
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.io.extractors.training_wheel import WHEEL_RADIUS_CM
from ibllib.io.extractors.ephys_fpga import WHEEL_TICKS
from alf.io import is_session_path
from . import base

_log = logging.getLogger('ibllib')


class TaskQC(base.QC):
    """A class for computing task QC metrics"""
    criteria = {"PASS": 0.99,
                "WARNING": 0.95,
                "FAIL": 0}

    def __init__(self, session_path_or_eid, **kwargs):
        """
        :param session_path_or_eid: A session eid or path
        :param log: A logging.Logger instance, if None the 'ibllib' logger is used
        :param one: An ONE instance for fetching and setting the QC on Alyx
        """
        # When an eid is provided, we will download the required data by default (if necessary)
        self.download_data = not is_session_path(session_path_or_eid)
        super().__init__(session_path_or_eid, **kwargs)

        # Data
        self.extractor = None

        # Metrics and passed trials
        self.metrics = None
        self.passed = None

    def load_data(self, bpod_only=False, download_data=True):
        """Extract the data from raw data files
        Extracts all the required task data from the raw data files.

        :param bpod_only: if True no data is extracted from the FPGA for ephys sessions
        :param download_data: if True, any missing raw data is downloaded via ONE.
        """
        self.extractor = TaskQCExtractor(
            self.session_path, one=self.one, download_data=download_data, bpod_only=bpod_only)

    def compute(self, **kwargs):
        """Compute and store the QC metrics
        Runs the QC on the session and stores a map of the metrics for each datapoint for each
        test, and a map of which datapoints passed for each test
        :param bpod_only: if True no data is extracted from the FPGA for ephys sessions
        :param download_data: if True, any missing raw data is downloaded via ONE.  By default
        data are not downloaded if a session path was provided to the constructor.
        :return:
        """
        if self.extractor is None:
            kwargs['download_data'] = kwargs.pop('download_data', self.download_data)
            self.load_data(**kwargs)
        self.log.info(f"Session {self.session_path}: Running QC on behavior data...")
        self.metrics, self.passed = get_bpodqc_metrics_frame(
            self.extractor.data,
            wheel_gain=self.extractor.settings["STIM_GAIN"],  # The wheel gain
            photodiode=self.extractor.frame_ttls,
            audio=self.extractor.audio_ttls,
            re_encoding=self.extractor.wheel_encoding or 'X1',
            min_qt=self.extractor.settings.get('QUIESCENT_PERIOD') or 0.2
        )
        return

    def run(self, update=False, **kwargs):
        """
        :param update: if True, updates the session QC fields on Alyx
        :param bpod_only: if True no data is extracted from the FPGA for ephys sessions
        :param download_data: if True, any missing raw data is downloaded via ONE.  By default
        data are not downloaded if a session path was provided to the constructor.
        :return: session outcome (str), a dict for extended QC
        """
        if self.metrics is None:
            self.compute(**kwargs)
        self.outcome, results, _ = self.compute_session_status()
        if update:
            self.update_extended_qc(results)
            self.update(self.outcome, 'task')
        return self.outcome, results

    def compute_session_status(self):
        """
        :return: Overall session QC outcome as a string
        :return: A map of QC tests and the proportion of data points that passed them
        :return: A map of QC tests and their outcomes
        """
        if self.passed is None:
            raise AttributeError('passed is None; compute QC first')
        MAX_BOUND, MIN_BOUND = (1, 0)
        # Get mean passed of each check, or None if passed is None or all NaN
        results = {k: None if v is None or np.isnan(v).all() else np.nanmean(v)
                   for k, v in self.passed.items()}

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

        def key_map(x):
            return 'NOT_SET' if x < 0 else list(criteria.keys())[x]
        # Criteria map is in order of severity so the max index is our overall QC outcome
        session_outcome = key_map(max(indices))
        outcomes = dict(zip(results.keys(), map(key_map, indices)))

        return session_outcome, results, outcomes


class HabituationQC(TaskQC):

    def compute(self, download_data=None):
        """Compute and store the QC metrics
        Runs the QC on the session and stores a map of the metrics for each datapoint for each
        test, and a map of which datapoints passed for each test
        :return:
        """
        if self.extractor is None:
            # If download_data is None, decide based on whether eid or session path was provided
            ensure_data = self.download_data if download_data is None else download_data
            self.load_data(download_data=ensure_data)
        self.log.info(f"Session {self.session_path}: Running QC on habituation data...")

        # Initialize checks
        prefix = '_task_'
        data = self.extractor.data
        metrics = {}
        passed = {}

        # Check all reward volumes == 3.0ul
        check = prefix + 'reward_volumes'
        metrics[check] = data['rewardVolume']
        passed[check] = metrics[check] == 3.0

        # Check session durations are increasing in steps >= 12 minutes
        check = prefix + 'habituation_time'
        if not self.one or not self.session_path:
            self.log.warning('unable to determine session trials without ONE')
            metrics[check] = passed[check] = None
        else:
            subject, session_date = self.session_path.parts[-3:-1]
            # compute from the date specified
            date_minus_week = (
                datetime.strptime(session_date, '%Y-%m-%d') - timedelta(days=7)
            ).strftime('%Y-%m-%d')
            sessions = self.one.alyx.rest('sessions', 'list', subject=subject,
                                          date_range=[date_minus_week, session_date],
                                          task_protocol='habituation')
            # Remove the current session if already registered
            if sessions[0]['start_time'].startswith(session_date):
                sessions = sessions[1:]
            metric = ([0, data['intervals'][-1, 1] - data['intervals'][0, 0]] +
                      [(datetime.fromisoformat(x['end_time']) -
                        datetime.fromisoformat(x['start_time'])).total_seconds() / 60
                       for x in [self.one.alyx.get(s['url']) for s in sessions]])

            # The duration from raw trial data
            # duration = map(float, self.extractor.raw_data[-1]['elapsed_time'].split(':'))
            # duration = timedelta(**dict(zip(('hours', 'minutes', 'seconds'),
            #                                 duration))).total_seconds() / 60
            metrics[check] = np.array(metric)
            passed[check] = np.diff(metric) >= 12

        # Check event orders: trial_start < stim on < stim center < feedback < stim off
        check = prefix + 'trial_event_sequence'
        nans = (
                np.isnan(data["intervals"][:, 0])  |  # noqa
                np.isnan(data["stimOn_times"])     |  # noqa
                np.isnan(data["stimCenter_times"]) |
                np.isnan(data["valveOpen_times"])  |  # noqa
                np.isnan(data["stimOff_times"])
        )
        a = np.less(data["intervals"][:, 0], data["stimOn_times"], where=~nans)
        b = np.less(data["stimOn_times"], data["stimCenter_times"], where=~nans)
        c = np.less(data["stimCenter_times"], data["valveOpen_times"], where=~nans)
        d = np.less(data["valveOpen_times"], data["stimOff_times"], where=~nans)

        metrics[check] = a & b & c & d & ~nans
        passed[check] = metrics[check].astype(np.float)

        # Check that the time difference between the visual stimulus center-command being
        # triggered and the stimulus effectively appearing in the center is smaller than 150 ms.
        check = prefix + 'stimCenter_delays'
        metric = np.nan_to_num(data["stimCenter_times"] - data["stimCenterTrigger_times"],
                               nan=np.inf)
        passed[check] = (metric <= 0.15) & (metric > 0)
        metrics[check] = metric

        # Phase check
        check = prefix + 'phase'
        metric = data['phase']
        passed[check] = (metric <= 2 * np.pi) & (metric >= 0)
        metrics[check] = metric

        check = prefix + 'phase_distribution'
        metric, _ = np.histogram(data['phase'])
        _, p = chisquare(metric)
        passed[check] = p < 0.05
        metrics[check] = metric

        # Checks common to training QC
        checks = [check_goCue_delays, check_stimOn_goCue_delays,
                  check_stimOn_delays, check_stimOff_delays]
        for fcn in checks:
            check = prefix + fcn.__name__[6:]
            metrics[check], passed[check] = fcn(data)

        self.metrics, self.passed = (metrics, passed)


def get_bpodqc_metrics_frame(data, **kwargs):
    """
    Evaluates all the QC metric functions in this module (those starting with 'check') and
    returns the results.  The optional kwargs listed below are passed to each QC metric function.
    :param data: dict of extracted task data
    :param re_encoding: the encoding of the wheel data, X1, X2 or X4
    :param enc_res: the rotary encoder resolution
    :param wheel_gain: the STIM_GAIN task parameter
    :param photodiode: the fronts from Bpod's BNC1 input or FPGA frame2ttl channel
    :param audio: the fronts from Bpod's BNC2 input FPGA audio sync channel
    :param min_qt: the QUIESCENT_PERIOD task parameter
    :return metrics: dict of checks and their QC metrics
    :return passed: dict of checks and a float array of which samples passed
    """
    def is_metric(x):
        return isfunction(x) and x.__name__.startswith('check_')
    checks = getmembers(sys.modules[__name__], is_metric)
    prefix = '_task_'
    qc_metrics_map = {prefix + k[6:]: fn(data, **kwargs) for k, fn in checks}

    # Split metrics and passed frames
    metrics = {}
    passed = {}
    for k in qc_metrics_map:
        metrics[k], passed[k] = qc_metrics_map[k]

    # Add a check for trial level pass: did a given trial pass all checks?
    n_trials = data['intervals'].shape[0]
    trial_level_passed = [m for m in passed.values()
                          if isinstance(m, Sized) and len(m) == n_trials]
    name = prefix + 'passed_trial_checks'
    metrics[name] = reduce(np.logical_and, trial_level_passed or (None, None))
    passed[name] = metrics[name].astype(np.float) if trial_level_passed else None

    return metrics, passed


# SINGLE METRICS
# ---------------------------------------------------------------------------- #

# === Delays between events checks ===

def check_stimOn_goCue_delays(data, **_):
    """ Checks that the time difference between the onset of the visual stimulus
    and the onset of the go cue tone is positive and less than 10ms.

    Metric: M = stimOn_times - goCue_times
    Criteria: 0 < M < 0.010 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('goCue_times', 'stimOn_times', 'intervals')
    """
    metric = np.nan_to_num(data["goCue_times"] - data["stimOn_times"], nan=np.inf)
    passed = (metric < 0.01) & (metric > 0)
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_response_feedback_delays(data, **_):
    """ Checks that the time difference between the response and the feedback onset
    (error sound or valve) is positive and less than 10ms.

    Metric: M = Feedback_time - response_time
    Criterion: 0 < M < 0.010 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('feedback_times', 'response_times', 'intervals')
    """
    metric = np.nan_to_num(data["feedback_times"] - data["response_times"], nan=np.inf)
    passed = (metric < 0.01) & (metric > 0)
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_response_stimFreeze_delays(data, **_):
    """ Checks that the time difference between the visual stimulus freezing and the
    response is positive and less than 100ms.

    Metric: M = (stimFreeze_times - response_times)
    Criterion: 0 < M < 0.100 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('stimFreeze_times', 'response_times', 'intervals',
    'choice')
    """
    metric = np.nan_to_num(data["stimFreeze_times"] - data["response_times"], nan=np.inf)
    # Test for valid values
    passed = ((metric < 0.1) & (metric > 0)).astype(np.float)
    # Finally remove no_go trials (stimFreeze triggered differently in no_go trials)
    # should account for all the nans
    passed[data["choice"] == 0] = np.nan
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_stimOff_itiIn_delays(data, **_):
    """ Check that the start of the trial interval is within 10ms of the visual stimulus turning off.

    Metric: M = itiIn_times - stimOff_times
    Criterion: 0 < M < 0.010 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('stimOff_times', 'itiIn_times', 'intervals',
    'choice')
    """
    metric = np.nan_to_num(data["itiIn_times"] - data["stimOff_times"], nan=np.inf)
    passed = ((metric < 0.01) & (metric >= 0)).astype(np.float)
    # Remove no_go trials (stimOff triggered differently in no_go trials)
    metric[data["choice"] == 0] = passed[data["choice"] == 0] = np.nan
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_iti_delays(data, **_):
    """ Check that the period of gray screen between stim off and the start of the next trial is
    0.5s +/- 200%.

    Metric: M = stimOff (n) - trialStart (n+1) - 0.5
    Criterion: |M| < 1
    Units: seconds [s]

    :param data: dict of trial data with keys ('stimOff_times', 'intervals')
    """
    # Initialize array the length of completed trials
    metric = np.full(data["intervals"].shape[0], np.nan)
    passed = metric.copy()
    # Get the difference between stim off and the start of the next trial
    # Missing data are set to Inf, except for the last trial which is a NaN
    metric[:-1] = \
        np.nan_to_num(data["intervals"][1:, 0] - data["stimOff_times"][:-1] - 0.5, nan=np.inf)
    passed[:-1] = np.abs(metric[:-1]) < .5  # Last trial is not counted
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_positive_feedback_stimOff_delays(data, **_):
    """ Check that the time difference between the valve onset and the visual stimulus turning off
    is 1 ± 0.150 seconds.

    Metric: M = stimOff_times - feedback_times - 1s
    Criterion: |M| < 0.150 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('stimOff_times', 'feedback_times', 'intervals',
    'correct')
    """
    metric = np.nan_to_num(data["stimOff_times"] - data["feedback_times"] - 1, nan=np.inf)
    passed = (np.abs(metric) < 0.15).astype(np.float)
    metric[~data["correct"]] = passed[~data["correct"]] = np.nan
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_negative_feedback_stimOff_delays(data, **_):
    """ Check that the time difference between the error sound and the visual stimulus
    turning off is 2 ± 0.150 seconds.

    Metric: M = stimOff_times - errorCue_times - 2s
    Criterion: |M| < 0.150 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('stimOff_times', 'errorCue_times', 'intervals')
    """
    metric = np.nan_to_num(data["stimOff_times"] - data["errorCue_times"] - 2, nan=np.inf)
    # Apply criteria
    passed = (np.abs(metric) < 0.15).astype(np.float)
    # Remove none negative feedback trials
    metric[data["correct"]] = passed[data["correct"]] = np.nan
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


# === Wheel movement during trial checks ===

def check_wheel_move_before_feedback(data, **_):
    """ Check that the wheel does move within 100ms of the feedback onset (error sound or valve).

    Metric: M = (w_t - 0.05) - (w_t + 0.05), where t = feedback_times
    Criterion: M != 0
    Units: radians

    :param data: dict of trial data with keys ('wheel_timestamps', 'wheel_position', 'choice',
    'intervals', 'feedback_times')
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
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def _wheel_move_during_closed_loop(re_ts, re_pos, data, wheel_gain=None, tol=1, **_):
    """ Check that the wheel moves by approximately 35 degrees during the closed-loop period
    on trials where a feedback (error sound or valve) is delivered.

    Metric: M = abs(w_resp - w_t0) - threshold_displacement, where w_resp = position at response
        time, w_t0 = position at go cue time, threshold_displacement = displacement required to
        move 35 visual degrees
    Criterion: displacement < tol visual degree
    Units: degrees angle of wheel turn

    :param re_ts: extarcted wheel timestamps in seconds
    :param re_pos: extracted wheel positions in radians
    :param data: a dict with the keys (goCueTrigger_times, response_times, feedback_times,
    position, choice, intervals)
    :param wheel_gain: the 'STIM_GAIN' task setting
    :param tol: the criterion in visual degrees
    """
    if wheel_gain is None:
        _log.warning("No wheel_gain input in function call, returning None")
        return None, None

    # Get tuple of wheel times and positions over each trial's closed-loop period
    traces = traces_by_trial(re_ts, re_pos,
                             start=data["goCueTrigger_times"],
                             end=data["response_times"])

    metric = np.zeros_like(data["feedback_times"])
    # For each trial find the absolute displacement
    for i, trial in enumerate(traces):
        t, pos = trial
        if pos.size != 0:
            # Find the position of the preceding sample and subtract it
            idx = np.abs(re_ts - t[0]).argmin() - 1
            origin = re_pos[idx]
            metric[i] = np.abs(pos - origin).max()

    # Load wheel_gain and thresholds for each trial
    wheel_gain = np.array([wheel_gain] * len(data["position"]))
    thresh = data["position"]
    # abs displacement, s, in mm required to move 35 visual degrees
    s_mm = np.abs(thresh / wheel_gain)  # don't care about direction
    criterion = cm_to_rad(s_mm * 1e-1)  # convert abs displacement to radians (wheel pos is in rad)
    metric = metric - criterion  # difference should be close to 0
    rad_per_deg = cm_to_rad(1 / wheel_gain * 1e-1)
    passed = (np.abs(metric) < rad_per_deg * tol).astype(np.float)  # less than 1 visual degree off
    metric[data["choice"] == 0] = passed[data["choice"] == 0] = np.nan  # except no-go trials
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_wheel_move_during_closed_loop(data, wheel_gain=None, **_):
    """ Check that the wheel moves by approximately 35 degrees during the closed-loop period
    on trials where a feedback (error sound or valve) is delivered.

    Metric: M = abs(w_resp - w_t0) - threshold_displacement, where w_resp = position at response
        time, w_t0 = position at go cue time, threshold_displacement = displacement required to
        move 35 visual degrees
    Criterion: displacement < 3 visual degrees
    Units: degrees angle of wheel turn

    :param data: dict of trial data with keys ('wheel_timestamps', 'wheel_position', 'choice',
    'intervals', 'goCueTrigger_times', 'response_times', 'feedback_times', 'position')
    :param wheel_gain: the 'STIM_GAIN' task setting
    """
    # Get the Bpod extracted wheel data
    timestamps = data['wheel_timestamps']
    position = data['wheel_position']

    return _wheel_move_during_closed_loop(timestamps, position, data, wheel_gain, tol=3)


def check_wheel_move_during_closed_loop_bpod(data, wheel_gain=None, **_):
    """ Check that the wheel moves by approximately 35 degrees during the closed-loop period
    on trials where a feedback (error sound or valve) is delivered.  This check uses the Bpod
    wheel data (measured at a lower resolution) with a stricter tolerance (1 visual degree).

    Metric: M = abs(w_resp - w_t0) - threshold_displacement, where w_resp = position at response
        time, w_t0 = position at go cue time, threshold_displacement = displacement required to
        move 35 visual degrees
    Criterion: displacement < 1 visual degree
    Units: degrees angle of wheel turn

    :param data: dict of trial data with keys ('wheel_timestamps(_bpod)', 'wheel_position(_bpod)',
    'choice', 'intervals', 'goCueTrigger_times', 'response_times', 'feedback_times', 'position')
    :param wheel_gain: the 'STIM_GAIN' task setting
    """
    # Get the Bpod extracted wheel data
    timestamps = data.get('wheel_timestamps_bpod', data['wheel_timestamps'])
    position = data.get('wheel_position_bpod', data['wheel_position'])

    return _wheel_move_during_closed_loop(timestamps, position, data, wheel_gain, tol=1)


def check_wheel_freeze_during_quiescence(data, **_):
    """ Check that the wheel does not move more than 2 degrees in each direction during the
    quiescence interval before the stimulus appears.

    Metric: M = |max(W) - min(W)| where W is wheel pos over quiescence interval
    interval = [stimOnTrigger_times - quiescent_duration, stimOnTrigger_times]
    Criterion: M < 2 degrees
    Units: degrees angle of wheel turn

    :param data: dict of trial data with keys ('wheel_timestamps', 'wheel_position', 'quiescence',
    'intervals', 'stimOnTrigger_times')
    """
    assert np.all(np.diff(data["wheel_timestamps"]) >= 0)
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
    passed = metric < criterion
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_detected_wheel_moves(data, min_qt=0, **_):
    """ Check that the detected first movement times are reasonable.

    Metric: M = firstMovement times
    Criterion: (goCue trigger time - min quiescent period) < M < response time
    Units: Seconds [s]

    :param data: dict of trial data with keys ('firstMovement_times', 'goCueTrigger_times',
    'response_times', 'choice', 'intervals')
    :param min_qt: the minimum possible quiescent period (the QUIESCENT_PERIOD task parameter)
    """
    # Depending on task version this may be a single value or an array of quiescent periods
    min_qt = np.array(min_qt)
    if min_qt.size > data["intervals"].shape[0]:
        min_qt = min_qt[:data["intervals"].shape[0]]

    metric = data['firstMovement_times']
    qevt_start = data['goCueTrigger_times'] - np.array(min_qt)
    response = data['response_times']
    passed = np.array([a < m < b for m, a, b in zip(metric, qevt_start, response)], dtype=np.float)
    nogo = data['choice'] == 0
    passed[nogo] = np.nan  # No go trial may have no movement times and that's fine
    return metric, passed


# === Sequence of events checks ===

def check_error_trial_event_sequence(data, **_):
    """ Check that on incorrect / miss trials, there are exactly:
    2 audio events (go cue sound and error sound) and 2 Bpod events (trial start, ITI), occurring
    in the correct order

    Metric: M = Bpod (trial start) > audio (go cue) > audio (error) > Bpod (ITI) > Bpod (trial end)
    Criterion: M == True
    Units: -none-

    :param data: dict of trial data with keys ('errorCue_times', 'goCue_times', 'intervals',
    'itiIn_times', 'correct')
    """
    nans = (
        np.isnan(data["intervals"][:, 0]) |
        np.isnan(data["goCue_times"])     |  # noqa
        np.isnan(data["errorCue_times"])  |  # noqa
        np.isnan(data["itiIn_times"])     |  # noqa
        np.isnan(data["intervals"][:, 1])
    )

    a = np.less(data["intervals"][:, 0], data["goCue_times"], where=~nans)
    b = np.less(data["goCue_times"], data["errorCue_times"], where=~nans)
    c = np.less(data["errorCue_times"], data["itiIn_times"], where=~nans)
    d = np.less(data["itiIn_times"], data["intervals"][:, 1], where=~nans)

    metric = a & b & c & d & ~nans

    passed = metric.astype(np.float)
    passed[data["correct"]] = np.nan  # Look only at incorrect trials
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_correct_trial_event_sequence(data, **_):
    """ Check that on correct trials, there are exactly:
    1 audio events and 3 Bpod events (valve open, trial start, ITI), occurring in the correct order

    Metric: M = Bpod (trial start) > audio (go cue) > Bpod (valve) > Bpod (ITI) > Bpod (trial end)
    Criterion: M == True
    Units: -none-

    :param data: dict of trial data with keys ('valveOpen_times', 'goCue_times', 'intervals',
    'itiIn_times', 'correct')
    """
    nans = (
        np.isnan(data["intervals"][:, 0]) |
        np.isnan(data["goCue_times"])     |  # noqa
        np.isnan(data["valveOpen_times"]) |
        np.isnan(data["itiIn_times"])     |  # noqa
        np.isnan(data["intervals"][:, 1])
    )

    a = np.less(data["intervals"][:, 0], data["goCue_times"], where=~nans)
    b = np.less(data["goCue_times"], data["valveOpen_times"], where=~nans)
    c = np.less(data["valveOpen_times"], data["itiIn_times"], where=~nans)
    d = np.less(data["itiIn_times"], data["intervals"][:, 1], where=~nans)
    metric = a & b & c & d & ~nans

    passed = metric.astype(np.float)
    passed[~data["correct"]] = np.nan  # Look only at correct trials
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_n_trial_events(data, **_):
    """ Check that the number events per trial is correct
    Within every trial interval there should be one of each trial event, except for
    goCueTrigger_times which should only be defined for incorrect trials

    Metric: M = all(start < event < end) for all event times except errorCueTrigger_times where
                start < error_trigger < end if not correct trial, else error_trigger == NaN
    Criterion: M == True
    Units: -none-, boolean

    :param data: dict of trial data with keys ('intervals', 'stimOnTrigger_times',
                 'stimOffTrigger_times', 'stimOn_times', 'stimOff_times', 'stimFreeze_times',
                 'stimFreezeTrigger_times', 'errorCueTrigger_times', 'itiIn_times',
                 'goCueTrigger_times', 'goCue_times', 'response_times', 'feedback_times')
    """

    intervals = data['intervals']
    correct = data['correct']
    err_trig = data['errorCueTrigger_times']

    # Exclude these fields; valve and errorCue times are the same as feedback_times and we must
    # test errorCueTrigger_times separately
    exclude = ['camera_timestamps', 'errorCueTrigger_times', 'errorCue_times',
               'firstMovement_times', 'peakVelocity_times', 'valveOpen_times',
               'wheel_moves_peak_amplitude', 'wheel_moves_intervals', 'wheel_timestamps',
               'wheel_intervals']
    events = [k for k in data.keys() if k.endswith('_times') and k not in exclude]
    metric = np.zeros(data["intervals"].shape[0], dtype=bool)

    # For each trial interval check that one of each trial event occurred.  For incorrect trials,
    # check the error cue trigger occurred within the interval, otherwise check it is nan.
    for i, (start, end) in enumerate(intervals):
        metric[i] = (all([start < data[k][i] < end for k in events]) and
                     (np.isnan(err_trig[i]) if correct[i] else start < err_trig[i] < end))
    passed = metric.astype(np.bool)
    assert intervals.shape[0] == len(metric) == len(passed)
    return metric, passed


def check_trial_length(data, **_):
    """ Check that the time difference between the onset of the go cue sound
    and the feedback (error sound or valve) is positive and smaller than 60.1 s.

    Metric: M = feedback_times - goCue_times
    Criteria: 0 < M < 60.1 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('feedback_times', 'goCue_times', 'intervals')
    """
    metric = np.nan_to_num(data["feedback_times"] - data["goCue_times"], nan=np.inf)
    passed = (metric < 60.1) & (metric > 0)
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


# === Trigger-response delay checks ===

def check_goCue_delays(data, **_):
    """ Check that the time difference between the go cue sound being triggered and
    effectively played is smaller than 1ms.

    Metric: M = goCue_times - goCueTrigger_times
    Criterion: 0 < M <= 0.001 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('goCue_times', 'goCueTrigger_times', 'intervals')
    """
    metric = np.nan_to_num(data["goCue_times"] - data["goCueTrigger_times"], nan=np.inf)
    passed = (metric <= 0.0015) & (metric > 0)
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_errorCue_delays(data, **_):
    """ Check that the time difference between the error sound being triggered and
    effectively played is smaller than 1ms.
    Metric: M = errorCue_times - errorCueTrigger_times
    Criterion: 0 < M <= 0.001 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('errorCue_times', 'errorCueTrigger_times',
    'intervals', 'correct')
    """
    metric = np.nan_to_num(data["errorCue_times"] - data["errorCueTrigger_times"], nan=np.inf)
    passed = ((metric <= 0.0015) & (metric > 0)).astype(np.float)
    passed[data["correct"]] = metric[data["correct"]] = np.nan
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_stimOn_delays(data, **_):
    """ Check that the time difference between the visual stimulus onset-command being triggered
    and the stimulus effectively appearing on the screen is smaller than 150 ms.

    Metric: M = stimOn_times - stimOnTrigger_times
    Criterion: 0 < M < 0.150 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('stimOn_times', 'stimOnTrigger_times',
    'intervals')
    """
    metric = np.nan_to_num(data["stimOn_times"] - data["stimOnTrigger_times"], nan=np.inf)
    passed = (metric <= 0.15) & (metric > 0)
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_stimOff_delays(data, **_):
    """ Check that the time difference between the visual stimulus offset-command
    being triggered and the visual stimulus effectively turning off on the screen
    is smaller than 150 ms.

    Metric: M = stimOff_times - stimOffTrigger_times
    Criterion: 0 < M < 0.150 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('stimOff_times', 'stimOffTrigger_times',
    'intervals')
    """
    metric = np.nan_to_num(data["stimOff_times"] - data["stimOffTrigger_times"], nan=np.inf)
    passed = (metric <= 0.15) & (metric > 0)
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_stimFreeze_delays(data, **_):
    """ Check that the time difference between the visual stimulus freeze-command
    being triggered and the visual stimulus effectively freezing on the screen
    is smaller than 150 ms.

    Metric: M = stimFreeze_times - stimFreezeTrigger_times
    Criterion: 0 < M < 0.150 s
    Units: seconds [s]

    :param data: dict of trial data with keys ('stimFreeze_times', 'stimFreezeTrigger_times',
    'intervals')
    """
    metric = np.nan_to_num(data["stimFreeze_times"] - data["stimFreezeTrigger_times"], nan=np.inf)
    passed = (metric <= 0.15) & (metric > 0)
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


# === Data integrity checks ===

def check_reward_volumes(data, **_):
    """ Check that the reward volume is between 1.5 and 3 uL for correct trials, 0 for incorrect.

    Metric: M = reward volume
    Criterion: 1.5 <= M <= 3 if correct else M == 0
    Units: uL

    :param data: dict of trial data with keys ('rewardVolume', 'correct', 'intervals')
    """
    metric = data['rewardVolume']
    correct = data['correct']
    passed = np.zeros_like(metric, dtype=np.bool)
    # Check correct trials within correct range
    passed[correct] = (1.5 <= metric[correct]) & (metric[correct] <= 3.)
    # Check incorrect trials are 0
    passed[~correct] = metric[~correct] == 0
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_reward_volume_set(data, **_):
    """ Check that there is only two reward volumes within a session, one of which is 0.

    Metric: M = set(rewardVolume)
    Criterion: (0 < len(M) <= 2) and 0 in M

    :param data: dict of trial data with keys ('rewardVolume')
    """
    metric = data["rewardVolume"]
    passed = 0 < len(set(metric)) <= 2 and 0. in metric
    return metric, passed


def check_wheel_integrity(data, re_encoding='X1', enc_res=None, **_):
    """ Check that the difference between wheel position samples is close to the encoder resolution
    and that the wheel timestamps strictly increase.

    Note: At high velocities some samples are missed due to the scanning frequency of the DAQ.
    This checks for more than 1 missing sample in a row (i.e. the difference between samples >= 2)

    Metric: M = (absolute difference of the positions < 1.5 * encoder resolution)
                 + 1 if (difference of timestamps <= 0) else 0
    Criterion: M  ~= 0
    Units: arbitrary (radians, sometimes + 1)

    :param data: dict of wheel data with keys ('wheel_timestamps', 'wheel_position')
    :param re_encoding: the encoding of the wheel data, X1, X2 or X4
    :param enc_res: the rotary encoder resolution (default 1024 ticks per revolution)
    """
    if isinstance(re_encoding, str):
        re_encoding = int(re_encoding[-1])
    # The expected difference between samples in the extracted units
    resolution = 1 / (enc_res or WHEEL_TICKS) * np.pi * 2 * WHEEL_RADIUS_CM / re_encoding
    # We expect the difference of neighbouring positions to be close to the resolution
    pos_check = np.abs(np.diff(data['wheel_position']))
    # Timestamps should be strictly increasing
    ts_check = np.diff(data['wheel_timestamps']) <= 0.
    metric = pos_check + ts_check.astype(float)  # all values should be close to zero
    passed = metric < 1.5 * resolution
    return metric, passed


# === Pre-stimulus checks ===

def check_stimulus_move_before_goCue(data, photodiode=None, **_):
    """ Check that there are no visual stimulus change(s) between the start of the trial and the
    go cue sound onset - 20 ms.

    Metric: M = number of visual stimulus change events between trial start and goCue_times - 20ms
    Criterion: M == 0
    Units: -none-, integer

    :param data: dict of trial data with keys ('goCue_times', 'intervals', 'choice')
    :param photodiode: the fronts from Bpod's BNC1 input or FPGA frame2ttl channel
    """
    if photodiode is None:
        _log.warning("No photodiode TTL input in function call, returning None")
        return None
    s = photodiode["times"]
    s = s[~np.isnan(s)]  # Remove NaNs
    metric = np.array([])
    for i, c in zip(data["intervals"][:, 0], data["goCue_times"]):
        metric = np.append(metric, np.count_nonzero(s[s > i] < (c - 0.02)))

    passed = (metric == 0).astype(np.float)
    # Remove no go trials
    passed[data["choice"] == 0] = np.nan
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed


def check_audio_pre_trial(data, audio=None, **_):
    """ Check that there are no audio outputs between the start of the trial and the
    go cue sound onset - 20 ms.

    Metric: M = sum(start_times < audio TTL < (goCue_times - 20ms))
    Criterion: M == 0
    Units: -none-, integer

    :param data: dict of trial data with keys ('goCue_times', 'intervals')
    :param audio: the fronts from Bpod's BNC2 input FPGA audio sync channel
    """
    if audio is None:
        _log.warning("No BNC2 input in function call, retuning None")
        return None
    s = audio["times"][~np.isnan(audio["times"])]  # Audio TTLs with NaNs removed
    metric = np.array([], dtype=np.int8)
    for i, c in zip(data["intervals"][:, 0], data["goCue_times"]):
        metric = np.append(metric, sum(s[s > i] < (c - 0.02)))
    passed = metric == 0
    assert data["intervals"].shape[0] == len(metric) == len(passed)
    return metric, passed

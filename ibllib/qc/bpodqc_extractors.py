import logging

import numpy as np

from ibllib.io.extractors.training_trials import (
    StimOnOffFreezeTimes, Choice, FeedbackType, Intervals, StimOnTriggerTimes, StimOnTimes,
    StimOffTriggerTimes, StimFreezeTriggerTimes, GoCueTriggerTimes, GoCueTimes,
    ErrorCueTriggerTimes, RewardVolume, ResponseTimes, FeedbackTimes, ItiInTimes,
    run_extractor_classes
)
from ibllib.io.extractors.ephys_fpga import ProbaContrasts, _get_pregenerated_events
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.training_wheel import get_wheel_position

_logger = logging.getLogger("ibllib")


# --------------------------------------------------------------------------- #
def extract_bpod_trial_data(session_path, raw_bpod_trials=None, raw_settings=None):
    """Extracts and loads ephys sessions from bpod data"""
    _logger.info(f"Extracting session: {session_path}")
    raw_bpod_trials = raw_bpod_trials or raw.load_data(session_path)
    raw_settings = raw_settings or raw.load_settings(session_path)
    classes = (
        StimOnOffFreezeTimes, Choice, FeedbackType, Intervals, StimOnTriggerTimes, StimOnTimes,
        StimOffTriggerTimes, StimFreezeTriggerTimes, GoCueTriggerTimes, GoCueTimes,
        ErrorCueTriggerTimes, RewardVolume, ResponseTimes, FeedbackTimes, ItiInTimes,
        ProbaContrasts
    )
    out, _ = run_extractor_classes(classes, save=False, session_path=session_path,
                                   bpod_trials=raw_bpod_trials, settings=raw_settings)
    out.update(_get_pregenerated_events(raw_bpod_trials, raw_settings))
    # get valve_time and errorCue_times from feedback_times
    correct = np.sign(out["position"]) + np.sign(out["choice"]) == 0
    errorCue_times = out["feedback_times"].copy()
    valveOpen_times = out["feedback_times"].copy()
    errorCue_times[correct] = np.nan
    valveOpen_times[~correct] = np.nan
    out.update(
        {"errorCue_times": errorCue_times, "valveOpen_times": valveOpen_times, "correct": correct}
    )
    # split intervals
    out["intervals_0"] = out["intervals"][:, 0]
    out["intervals_1"] = out["intervals"][:, 1]
    _ = out.pop("intervals")
    out["outcome"] = out["feedbackType"].copy()
    out["outcome"][out["choice"] == 0] = 0
    return out


class BpodQCExtractor(object):
    def __init__(self, session_path, lazy=False):
        self.session_path = session_path
        self.load_raw_data()
        if not lazy:
            self.extract_trial_data()

    def load_raw_data(self):
        _logger.info(f"Loading raw data from {self.session_path}")
        self.raw_data = raw.load_data(self.session_path)
        self.details = raw.load_settings(self.session_path)
        self.BNC1, self.BNC2 = raw.load_bpod_fronts(self.session_path, data=self.raw_data)
        # NOTE: wheel_position is actually an extractor needs _iblrig_encoderPositions.raw
        # to be there but not as input... FIXME: we should have the extractor use the data
        # without assuming it's there
        ts, pos = get_wheel_position(self.session_path, bp_data=self.raw_data)
        self.wheel_data = {'re_ts': ts, 're_pos': pos}
        assert np.all(np.diff(self.wheel_data["re_ts"]) > 0)

    def extract_trial_data(self):
        _logger.info("Extracting trial data table...")
        self.trial_data = extract_bpod_trial_data(
            self.session_path, raw_bpod_trials=self.raw_data, raw_settings=self.details
        )

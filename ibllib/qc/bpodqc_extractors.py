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
from ibllib.pipes.training_preprocessing import extract_training

_logger = logging.getLogger("ibllib")


class BpodQCExtractor(object):
    def __init__(self, session_path, lazy=False):
        self.session_path = session_path
        self.load_raw_data()
        if not lazy:
            self.trial_data = self.extract_trial_data()

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
        """Extracts and loads ephys sessions from bpod data"""
        _logger.info(f"Extracting session: {self.session_path}")
        raw_bpod_trials = self.raw_data or raw.load_data(self.session_path)
        raw_settings = self.details or raw.load_settings(self.session_path)
        extractor_type = raw.get_session_extractor_type(self.session_path)
        classes = [
            StimOnOffFreezeTimes, Choice, FeedbackType, Intervals, StimOnTriggerTimes,
            StimOnTimes, StimOffTriggerTimes, StimFreezeTriggerTimes, GoCueTriggerTimes,
            GoCueTimes, ErrorCueTriggerTimes, RewardVolume, ResponseTimes, FeedbackTimes,
            ItiInTimes]
        if extractor_type == 'ephys':
            classes.append(ProbaContrasts)
        out, _ = run_extractor_classes(classes, save=False, session_path=self.session_path,
                                       bpod_trials=raw_bpod_trials, settings=raw_settings)
        if extractor_type == 'ephys':
            out.update(_get_pregenerated_events(raw_bpod_trials, raw_settings))
        else:
            out['quiescence'] = np.array([t['quiescent_period'] for t in raw_bpod_trials])
            out['position'] = np.array([t['position'] for t in raw_bpod_trials])
        # get valve_time and errorCue_times from feedback_times
        correct = out['feedbackType'] > 0
        errorCue_times = out["feedback_times"].copy()
        valveOpen_times = out["feedback_times"].copy()
        errorCue_times[correct] = np.nan
        valveOpen_times[~correct] = np.nan
        out.update(
            {"errorCue_times": errorCue_times, "valveOpen_times": valveOpen_times,
             "correct": correct}
        )
        # split intervals
        out["intervals_0"] = out["intervals"][:, 0]
        out["intervals_1"] = out["intervals"][:, 1]
        _ = out.pop("intervals")
        out["outcome"] = out["feedbackType"].copy()
        out["outcome"][out["choice"] == 0] = 0
        return out

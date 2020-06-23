from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
import numpy as np

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.training_trials import (  # noqa; noqa
    CameraTimestamps, Choice, FeedbackTimes, FeedbackType, GoCueTimes, GoCueTriggerTimes,
    IncludedTrials, Intervals, ItiDuration, ProbabilityLeft, ResponseTimes, RewardVolume,
    StimOnTimes, StimOnTriggerTimes, StimOnOffFreezeTimes)
from ibllib.misc import version


class ContrastLR(BaseBpodTrialsExtractor):
    """
    Get left and right contrasts from raw datafile.
    """
    save_names = ('_ibl_trials.contrastLeft.npy', '_ibl_trials.contrastRight.npy')
    var_names = ('contrastLeft', 'contrastRight')

    def _extract(self):
        contrastLeft = np.array([t['contrast'] if np.sign(
            t['position']) < 0 else np.nan for t in self.bpod_trials])
        contrastRight = np.array([t['contrast'] if np.sign(
            t['position']) > 0 else np.nan for t in self.bpod_trials])
        return contrastLeft, contrastRight


def extract_all(session_path, save=False, bpod_trials=False, settings=False):
    if not bpod_trials:
        bpod_trials = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}
    base = [FeedbackType, ContrastLR, ProbabilityLeft, Choice, RewardVolume,
            FeedbackTimes, StimOnTimes, Intervals, ResponseTimes, GoCueTriggerTimes,
            GoCueTimes, CameraTimestamps]
    # Version specific extractions
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        base.extend([StimOnTriggerTimes, IncludedTrials])
    else:
        base.append(ItiDuration)

    out, fil = run_extractor_classes(
        base, save=save, session_path=session_path, bpod_trials=bpod_trials, settings=settings)
    return out, fil

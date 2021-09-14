import numpy as np

from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.training_trials import (
    Choice, FeedbackTimes, FeedbackType, GoCueTimes, GoCueTriggerTimes,
    IncludedTrials, Intervals, ItiDuration, ProbabilityLeft, ResponseTimes, RewardVolume,
    StimOnTimes_deprecated, StimOnTriggerTimes, StimOnOffFreezeTimes, ItiInTimes,
    StimOffTriggerTimes, StimFreezeTriggerTimes, ErrorCueTriggerTimes)
from ibllib.misc import version


class ContrastLR(BaseBpodTrialsExtractor):
    """
    Get left and right contrasts from raw datafile.
    """
    save_names = ('_ibl_trials.contrastLeft.npy', '_ibl_trials.contrastRight.npy')
    var_names = ('contrastLeft', 'contrastRight')

    def _extract(self, **kwargs):
        contrastLeft = np.array([t['contrast'] if np.sign(
            t['position']) < 0 else np.nan for t in self.bpod_trials])
        contrastRight = np.array([t['contrast'] if np.sign(
            t['position']) > 0 else np.nan for t in self.bpod_trials])
        return contrastLeft, contrastRight


def extract_all(session_path, save=False, bpod_trials=False, settings=False, extra_classes=None):
    """
    :param session_path:
    :param save:
    :param bpod_trials:
    :param settings:
    :param extra_classes: additional BaseBpodTrialsExtractor subclasses for custom extractions
    :return:
    """
    if not bpod_trials:
        bpod_trials = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}
    base = [FeedbackType, ContrastLR, ProbabilityLeft, Choice, RewardVolume,
            FeedbackTimes, Intervals, ResponseTimes, GoCueTriggerTimes, GoCueTimes]

    # Version specific extractions
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        base.extend([StimOnTriggerTimes, IncludedTrials, StimOnOffFreezeTimes, ItiInTimes,
                     StimOffTriggerTimes, StimFreezeTriggerTimes, ErrorCueTriggerTimes])
    else:
        base.extend([ItiDuration, StimOnTimes_deprecated])

    if extra_classes:
        base.extend(extra_classes)

    out, fil = run_extractor_classes(
        base, save=save, session_path=session_path, bpod_trials=bpod_trials, settings=settings)
    return out, fil

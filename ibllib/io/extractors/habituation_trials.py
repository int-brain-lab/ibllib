import logging
import numpy as np

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.biased_trials import ContrastLR
from ibllib.io.extractors.training_trials import (FeedbackTimes, StimOnTriggerTimes, Intervals,
                                                  GoCueTimes)

_logger = logging.getLogger('ibllib')


class HabituationTrials(BaseBpodTrialsExtractor):
    var_names = ('feedbackType', 'rewardVolume', 'stimOff_times', 'contrastLeft', 'contrastRight',
                 'feedback_times', 'stimOn_times', 'stimOnTrigger_times', 'intervals',
                 'goCue_times', 'goCueTrigger_times', 'itiIn_times', 'stimOffTrigger_times',
                 'stimCenterTrigger_times', 'stimCenter_times')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exclude = ['itiIn_times', 'stimOffTrigger_times',
                   'stimCenter_times', 'stimCenterTrigger_times']
        self.save_names = tuple([f'_ibl_trials.{x}.npy' if x not in exclude else None
                                 for x in self.var_names])

    def _extract(self):
        # Extract all trials...

        # Get all stim_sync events detected
        ttls = [raw.get_port_events(tr, 'BNC1') for tr in self.bpod_trials]

        # Report missing events
        n_missing = sum(len(pulses) != 3 for pulses in ttls)
        # Check if all stim syncs have failed to be detected
        if n_missing == len(ttls):
            _logger.error(f'{self.session_path}: Missing ALL BNC1 TTLs ({n_missing} trials)')
        elif n_missing > 0:  # Check if any stim_sync has failed be detected for every trial
            _logger.warning(f'{self.session_path}: Missing BNC1 TTLs on {n_missing} trial(s)')

        # Extract datasets common to trainingChoiceWorld
        training = [ContrastLR, FeedbackTimes, Intervals, GoCueTimes, StimOnTriggerTimes]
        out, _ = run_extractor_classes(training, session_path=self.session_path, save=False,
                                       bpod_trials=self.bpod_trials, settings=self.settings)

        # GoCueTriggerTimes is the same event as StimOnTriggerTimes
        out['goCueTrigger_times'] = out['stimOnTrigger_times'].copy()

        # StimCenterTrigger times
        # Get the stim_on_state that triggers the onset of the stim
        stim_center_state = np.array([tr['behavior_data']['States timestamps']
                                      ['stim_center'][0] for tr in self.bpod_trials])
        out['stimCenterTrigger_times'] = stim_center_state[:, 0].T

        # StimCenter times
        stim_center_times = np.full(out['stimCenterTrigger_times'].shape, np.nan)
        for i, (sync, last) in enumerate(zip(ttls, out['stimCenterTrigger_times'])):
            """We expect there to be 3 pulses per trial; if this is the case, stim center will
            be the third pulse. If any pulses are missing, we can only be confident of the correct
            one if exactly one pulse occurs after the stim center trigger"""
            if len(sync) == 3 or (len(sync) > 0 and sum(pulse > last for pulse in sync) == 1):
                stim_center_times[i] = sync[-1]
        out['stimCenter_times'] = stim_center_times

        # StimOn times
        stimOn_times = np.full(out['stimOnTrigger_times'].shape, np.nan)
        for i, (sync, last) in enumerate(zip(ttls, out['stimCenterTrigger_times'])):
            """We expect there to be 3 pulses per trial; if this is the case, stim on will be the
            second pulse. If 1 pulse is missing, we can only be confident of the correct one if
            both pulses occur before the stim center trigger"""
            if len(sync) == 3 or (len(sync) == 2 and sum(pulse < last for pulse in sync) == 2):
                stimOn_times[i] = sync[1]
        out['stimOn_times'] = stimOn_times

        # RewardVolume
        trial_volume = [x['reward_amount'] for x in self.bpod_trials]
        out['rewardVolume'] = np.array(trial_volume).astype(np.float64)

        # StimOffTrigger times
        # StimOff occurs at trial start (ignore the first trial's state update)
        out['stimOffTrigger_times'] = np.array(
            [tr["behavior_data"]["States timestamps"]
             ["trial_start"][0][0] for tr in self.bpod_trials[1:]]
        )

        # StimOff times
        """
        There should be exactly three TTLs per trial.  stimOff_times should be the first TTL pulse.
        If 1 or more pulses are missing, we can not be confident of assigning the correct one.
        """
        trigg = out['stimOffTrigger_times']
        out['stimOff_times'] = np.array([sync[0] if len(sync) == 3 else np.nan
                                         for sync, off in zip(ttls[1:], trigg)])

        # FeedbackType is always positive
        out['feedbackType'] = np.ones(len(out['feedback_times']), dtype=np.int8)

        # ItiIn times
        out['itiIn_times'] = np.array(
            [tr["behavior_data"]["States timestamps"]
             ["iti"][0][0] for tr in self.bpod_trials]
        )

        # NB: We lose the last trial because the stim off event occurs at trial_num + 1
        n_trials = out['stimOff_times'].size
        return [out[k][:n_trials] for k in self.var_names]


def extract_all(session_path, save=False, bpod_trials=False, settings=False):
    """Extract all datasets from habituationChoiceWorld
    Note: only the datasets from the HabituationTrials extractor will be saved to disc.

    :param session_path: The session path where the raw data are saved
    :param save: If True, the datasets that are considered standard are saved to the session path
    :param bpod_trials: The raw Bpod trial data
    :param settings: The raw Bpod sessions
    :returns: a dict of datasets and a corresponding list of file names
    """
    if not bpod_trials:
        bpod_trials = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)

    # Standard datasets that may be saved as ALFs
    params = dict(session_path=session_path, bpod_trials=bpod_trials, settings=settings)
    out, fil = run_extractor_classes(HabituationTrials, save=save, **params)
    return out, fil

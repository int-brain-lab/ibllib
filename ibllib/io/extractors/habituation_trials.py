import logging
import numpy as np

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.biased_trials import ContrastLR
from ibllib.io.extractors.training_trials import (FeedbackTimes, StimOnTriggerTimes, Intervals,
                                                  GoCueTimes)

_logger = logging.getLogger('ibllib')


class StimCenterTriggerTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimCenterTrigger_times.npy'
    var_names = 'stimCenterTrigger_times'

    def _extract(self):
        # Get the stim_on_state that triggers the onset of the stim
        stim_center_state = np.array([tr['behavior_data']['States timestamps']
                                      ['stim_center'][0] for tr in self.bpod_trials])
        return stim_center_state[:, 0].T


class StimCenterTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimCenter_times.npy'
    var_names = 'stimCenter_times'

    def _extract(self):
        """
        Find the stim sync pulses of each trial.  There should be exactly three TTLs per trial.
        stimCenter_times should be the third TTL pulse.
        (Stim updates are in BNC1High and BNC1Low - frame2TTL device)
        """
        # Get all stim_sync events detected
        ttls = [raw.get_port_events(tr, 'BNC1') for tr in self.bpod_trials]
        stim_center_triggers, _ = StimCenterTriggerTimes(
            self.session_path).extract(self.bpod_trials, self.settings)

        # StimCenter times
        stim_center_times = np.full(stim_center_triggers.shape, np.nan)
        for i, (sync, last) in enumerate(zip(ttls, stim_center_triggers)):
            """We expect there to be 3 pulses per trial; if this is the case, stim center will
            be the third pulse. If any pulses are missing, we can only be confident of the correct
            one if exactly one pulse occurs after the stim center trigger"""
            if len(sync) == 3 or (len(sync) > 0 and sum(pulse > last for pulse in sync) == 1):
                stim_center_times[i] = sync[-1]

        return stim_center_times


class StimOffTriggerTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimOffTrigger_times.npy'
    var_names = 'stimOffTrigger_times'

    def _extract(self):
        # StimOff occurs at trial start (ignore the first trial's state update)
        stimOffTrigger_times = np.array(
            [tr["behavior_data"]["States timestamps"]
             ["trial_start"][0][0] for tr in self.bpod_trials[1:]]
        )

        return stimOffTrigger_times


class ItiInTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.itiIn_times.npy'
    var_names = 'itiIn_times'

    def _extract(self):
        iti_in = np.array(
            [tr["behavior_data"]["States timestamps"]
             ["iti"][0][0] for tr in self.bpod_trials]
        )
        return iti_in


class HabituationTrials(BaseBpodTrialsExtractor):
    var_names = ('feedbackType', 'rewardVolume', 'stimOff_times', 'contrastLeft', 'contrastRight',
                 'feedback_times', 'stimOn_times', 'stimOnTrigger_times', 'intervals',
                 'goCue_times', 'goCueTrigger_times')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_names = tuple([f'_ibl_trials.{x}.npy' for x in self.var_names])

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

        # StimOn times
        stimOn_times = np.full(out['stimOnTrigger_times'].shape, np.nan)
        stim_center_triggers, _ = (StimCenterTriggerTimes(self.session_path)
                                   .extract(self.bpod_trials, self.settings))
        for i, (sync, last) in enumerate(zip(ttls, stim_center_triggers)):
            """We expect there to be 3 pulses per trial; if this is the case, stim on will be the
            second pulse. If 1 pulse is missing, we can only be confident of the correct one if
            both pulses occur before the stim center trigger"""
            if len(sync) == 3 or (len(sync) == 2 and sum(pulse < last for pulse in sync) == 2):
                stimOn_times[i] = sync[1]
        out['stimOn_times'] = stimOn_times

        # RewardVolume
        trial_volume = [x['reward_amount'] for x in self.bpod_trials]
        out['rewardVolume'] = np.array(trial_volume).astype(np.float64)

        # StimOffTrigger times (not saved)
        stimOffTriggers, _ = (StimOffTriggerTimes(self.session_path)
                              .extract(self.bpod_trials, self.settings))

        # StimOff times
        """
        There should be exactly three TTLs per trial.  stimOff_times should be the first TTL pulse.
        If 1 or more pulses are missing, we can not be confident of assigning the correct one.
        """
        out['stimOff_times'] = np.array([sync[0] if len(sync) == 3 else np.nan
                                         for sync, off in zip(ttls[1:], stimOffTriggers)])

        # FeedbackType is always positive
        out['feedbackType'] = np.ones(len(out['feedback_times']), dtype=np.int8)

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
    # The extra datasets
    non_standard = [ItiInTimes, StimOffTriggerTimes, StimCenterTriggerTimes, StimCenterTimes]
    data, _ = run_extractor_classes(non_standard, save=False, **params)

    # Merge the extracted data
    out.update(data)
    fil.extend([None for _ in data.keys()])
    return out, fil

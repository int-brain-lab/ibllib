import logging
import numpy as np

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.biased_trials import ContrastLR
from ibllib.io.extractors.training_trials import (FeedbackTimes, StimOnTriggerTimes, Intervals,
                                                  StimOnTimes, GoCueTimes)
from ibllib.misc import version

_logger = logging.getLogger('ibllib')


class RewardVolume(BaseBpodTrialsExtractor):
    """
    Load reward volume delivered for each trial.
    **Optional:** saves _ibl_trials.rewardVolume.npy

    Uses reward_current to accumulate the amount of
    """
    save_names = '_ibl_trials.rewardVolume.npy'
    var_names = 'rewardVolume'

    def _extract(self):
        trial_volume = [x['reward_amount'] for x in self.bpod_trials]
        reward_volume = np.array(trial_volume).astype(np.float64)
        assert len(reward_volume) == len(self.bpod_trials)
        return reward_volume


class GoCueTriggerTimes(StimOnTriggerTimes):
    """
    Get trigger times of goCue from state machine.

    Current software solution for triggering sounds uses PyBpod soft codes.
    Delays can be in the order of 10's of ms. This is the time when the command
    to play the sound was executed. To measure accurate time, either getting the
    sound onset from xonar soundcard sync pulse (latencies may vary).

    NB: The goCue is triggered at the same time as stim on
    """
    save_names = '_ibl_trials.goCueTrigger_times.npy'
    var_names = 'goCueTrigger_times'


class StimCenterTriggerTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimCenterTrigger_times.npy'
    var_names = 'stimCenterTrigger_times'

    def _extract(self):
        # Get the stim_on_state that triggers the onset of the stim
        stim_on_state = np.array([tr['behavior_data']['States timestamps']
                                  ['stim_center'][0] for tr in self.bpod_trials])
        return stim_on_state[:, 0].T


class FeedbackType(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.feedbackType.npy'
    var_names = 'feedbackType'

    def _extract(self):
        # FeedbackType is always positive
        feedback_type = np.ones(len(self.bpod_trials), dtype=np.int8)
        return feedback_type


class StimCenterTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimCenter_times.npy'
    var_names = 'stimCenter_times'

    def _extract(self):
        """
        Find the stim_sync pulses of each trial.  There should be exactly three TTLs per trial.
        stimCenter_times should be the second TTL pulse.
        (Stim updates are in BNC1High and BNC1Low - frame2TTL device)
        """
        # Get all stim_sync events detected
        stim_sync_all = [raw.get_port_events(tr, 'BNC1') for tr in self.bpod_trials]
        stim_off_trigg, _ = StimOffTriggerTimes(self.session_path).extract(
            save=False, bpod_trials=self.bpod_trials, settings=self.settings)

        stimCenter_times = np.full(stim_off_trigg.shape, np.nan)
        for i, (sync, off) in enumerate(zip(stim_sync_all, stim_off_trigg)):
            if len(sync) == 3:
                """We expect there to be 3 pulses per trial; if this is the case, stim center will 
                be the second pulse"""
                stimCenter_times[i] = sync[1]
            elif len(sync) == 2:
                """If 1 pulse is missing, we can only be confident of the correct one if both 
                pulses occur before the stim off trigger"""
                if all(pulse < off for pulse in sync) == 2:
                    stimCenter_times[i] = sync[1]
            else:
                """If there are less than 2 pulses (or more than 3) we cannot reliably determine 
                which pulse is the stim center"""
                pass

        n_missing = np.count_nonzero(np.isnan(stimCenter_times))
        # Check if all stim_syncs have failed to be detected
        if n_missing == stimCenter_times.size:
            _logger.error(f'{self.session_path}: Missing ALL BNC1 TTLs ({n_missing} trials)')
        elif n_missing > 0:  # Check if any stim_sync has failed be detected for every trial
            _logger.warning(f'{self.session_path}: Missing BNC1 TTLs on {n_missing} trials')

        return stimCenter_times


class StimOffTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimOff_times.npy'
    var_names = 'stimOff_times'

    def _extract(self):
        """
        Find the stim_sync pulses of each trial.  There should be exactly three TTLs per trial.
        stimOff_times should be the third TTL pulse.
        (Stim updates are in BNC1High and BNC1Low - frame2TTL device)
        """
        # Get all stim_sync events detected
        stim_sync_all = [raw.get_port_events(tr, 'BNC1') for tr in self.bpod_trials]
        stim_off_trigg, _ = StimOffTriggerTimes(self.session_path).extract(
            save=False, bpod_trials=self.bpod_trials, settings=self.settings)

        stimOff_times = np.full(stim_off_trigg.shape, np.nan)
        for i, (sync, off) in enumerate(zip(stim_sync_all, stim_off_trigg)):
            if len(sync) == 3:
                """We expect there to be 3 pulses per trial; if this is the case, stim center will 
                be the second pulse"""
                stimOff_times[i] = sync[-1]
            else:
                """If 1 or more pulses are missing, we can only be confident of the correct one if 
                exactly 1 pulse occurs after the stim off trigger"""
                pulse = [x for x in sync if x > off]
                if len(pulse) == 1:
                    stimOff_times[i] = pulse

        n_missing = np.count_nonzero(np.isnan(stimOff_times))
        # Check if all stim_syncs have failed to be detected
        if n_missing == stimOff_times.size:
            _logger.error(f'{self.session_path}: Missing ALL BNC1 TTLs ({n_missing} trials)')
        elif n_missing > 0:  # Check if any stim_sync has failed be detected for every trial
            _logger.warning(f'{self.session_path}: Missing BNC1 TTLs on {n_missing} trials')

        return stimOff_times


class StimOffTriggerTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimOffTrigger_times.npy'
    var_names = 'stimOffTrigger_times'

    def _extract(self):
        # StimOff occurs at the end of the so-colled iti period
        stimOffTrigger_times = np.array(
            [tr["behavior_data"]["States timestamps"]
             ["iti"][0][1] for tr in self.bpod_trials]
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
                 'feedback_times', 'stimOn_times', 'intervals', 'goCue_times',
                 'goCueTrigger_times')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_names = tuple([f'_ibl_trials.{x}.npy' for x in self.var_names])

    def _extract(self):
        # Extract all trials
        data = []

        # FeedbackType is always positive
        feedback_type = np.ones(len(self.bpod_trials), dtype=np.int8)
        data.append(feedback_type)

        # RewardVolume
        trial_volume = [x['reward_amount'] for x in self.bpod_trials]
        reward_volume = np.array(trial_volume).astype(np.float64)
        data.append(reward_volume)

        # StimOffTimes
        """
        Find the stim_sync pulses of each trial.  There should be exactly three TTLs per trial.
        stimOff_times should be the third TTL pulse.
        (Stim updates are in BNC1High and BNC1Low - frame2TTL device)
        """
        # Get all stim_sync events detected
        stim_sync_all = [raw.get_port_events(tr, 'BNC1') for tr in self.bpod_trials]
        stim_off_trigg, _ = StimOffTriggerTimes(self.session_path).extract(
            save=False, bpod_trials=self.bpod_trials, settings=self.settings)

        stimOff_times = np.full(stim_off_trigg.shape, np.nan)
        for i, (sync, off) in enumerate(zip(stim_sync_all, stim_off_trigg)):
            if len(sync) == 3:
                """We expect there to be 3 pulses per trial; if this is the case, stim center will 
                be the second pulse"""
                stimOff_times[i] = sync[-1]
            else:
                """If 1 or more pulses are missing, we can only be confident of the correct one if 
                exactly 1 pulse occurs after the stim off trigger"""
                pulse = [x for x in sync if x > off]
                if len(pulse) == 1:
                    stimOff_times[i] = pulse

        n_missing = np.count_nonzero(np.isnan(stimOff_times))
        # Check if all stim_syncs have failed to be detected
        if n_missing == stimOff_times.size:
            _logger.error(f'{self.session_path}: Missing ALL BNC1 TTLs ({n_missing} trials)')
        elif n_missing > 0:  # Check if any stim_sync has failed be detected for every trial
            _logger.warning(f'{self.session_path}: Missing BNC1 TTLs on {n_missing} trials')

        data.append(stimOff_times)

        # Extract the rest from training
        # StimOnTriggerTimes is the same event as GoCueTriggerTimes
        training = [ContrastLR, FeedbackTimes, StimOnTimes, Intervals, GoCueTimes,
                    StimOnTriggerTimes]
        out, _ = run_extractor_classes(training, session_path=self.session_path, save=False,
                                         bpod_trials=self.bpod_trials, settings=self.settings)
        data.extend(out.values())  # FIXME Wrong order

        return data


def extract_all(session_path, save=False, bpod_trials=False, settings=False):
    if not bpod_trials:
        bpod_trials = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    base = [ContrastLR, ItiInTimes, StimOffTriggerTimes, RewardVolume, FeedbackType,
            FeedbackTimes, StimOnTimes, Intervals, GoCueTriggerTimes, GoCueTimes]
    # Version check
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        base.extend([StimOnTriggerTimes])

    out, fil = run_extractor_classes(
        base, save=save, session_path=session_path, bpod_trials=bpod_trials, settings=settings)
    return out, fil

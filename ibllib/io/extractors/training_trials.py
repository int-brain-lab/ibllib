import logging
import numpy as np
from pkg_resources import parse_version

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.misc import version


_logger = logging.getLogger('ibllib')


class FeedbackType(BaseBpodTrialsExtractor):
    """
    Get the feedback that was delivered to subject.
    **Optional:** saves _ibl_trials.feedbackType.npy

    Checks in raw datafile for error and reward state.
    Will raise an error if more than one of the mutually exclusive states have
    been triggered.

    Sets feedbackType to -1 if error state was trigered (applies to no-go)
    Sets feedbackType to +1 if reward state was triggered
    """
    save_names = '_ibl_trials.feedbackType.npy'
    var_names = 'feedbackType'

    def _extract(self):
        feedbackType = np.empty(len(self.bpod_trials))
        feedbackType.fill(np.nan)
        reward = []
        error = []
        no_go = []
        for t in self.bpod_trials:
            reward.append(~np.isnan(t['behavior_data']['States timestamps']['reward'][0][0]))
            error.append(~np.isnan(t['behavior_data']['States timestamps']['error'][0][0]))
            no_go.append(~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0]))

        if not all(np.sum([reward, error, no_go], axis=0) == np.ones(len(self.bpod_trials))):
            raise ValueError

        feedbackType[reward] = 1
        feedbackType[error] = -1
        feedbackType[no_go] = -1
        feedbackType = feedbackType.astype('int64')
        return feedbackType


class ContrastLR(BaseBpodTrialsExtractor):
    """
    Get left and right contrasts from raw datafile. Optionally, saves
    _ibl_trials.contrastLeft.npy and _ibl_trials.contrastRight.npy to alf folder.

    Uses signed_contrast to create left and right contrast vectors.
    """
    save_names = ('_ibl_trials.contrastLeft.npy', '_ibl_trials.contrastRight.npy')
    var_names = ('contrastLeft', 'contrastRight')

    def _extract(self):
        contrastLeft = np.array([t['contrast']['value'] if np.sign(
            t['position']) < 0 else np.nan for t in self.bpod_trials])
        contrastRight = np.array([t['contrast']['value'] if np.sign(
            t['position']) > 0 else np.nan for t in self.bpod_trials])

        return contrastLeft, contrastRight


class ProbabilityLeft(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.probabilityLeft.npy'
    var_names = 'probabilityLeft'

    def _extract(self, **kwargs):
        return np.array([t['stim_probability_left'] for t in self.bpod_trials])


class Choice(BaseBpodTrialsExtractor):
    """
    Get the subject's choice in every trial.
    **Optional:** saves _ibl_trials.choice.npy to alf folder.

    Uses signed_contrast and trial_correct.
    -1 is a CCW turn (towards the left)
    +1 is a CW turn (towards the right)
    0 is a no_go trial
    If a trial is correct the choice of the animal was the inverse of the sign
    of the position.

    >>> choice[t] = -np.sign(position[t]) if trial_correct[t]
    """
    save_names = '_ibl_trials.choice.npy'
    var_names = 'choice'

    def _extract(self):
        sitm_side = np.array([np.sign(t['position']) for t in self.bpod_trials])
        trial_correct = np.array([t['trial_correct'] for t in self.bpod_trials])
        trial_nogo = np.array(
            [~np.isnan(t['behavior_data']['States timestamps']['no_go'][0][0])
             for t in self.bpod_trials])
        choice = sitm_side.copy()
        choice[trial_correct] = -choice[trial_correct]
        choice[trial_nogo] = 0
        choice = choice.astype(int)
        return choice


class RepNum(BaseBpodTrialsExtractor):
    """
    Count the consecutive repeated trials.
    **Optional:** saves _ibl_trials.repNum.npy to alf folder.

    Creates trial_repeated from trial['contrast']['type'] == 'RepeatContrast'

    >>> trial_repeated = [0, 1, 1, 0, 1, 0, 1, 1, 1, 0]
    >>> repNum =         [0, 1, 2, 0, 1, 0, 1, 2, 3, 0]
    """
    save_names = '_ibl_trials.repNum.npy'
    var_names = 'repNum'

    def _extract(self):
        trial_repeated = np.array(
            [t['contrast']['type'] == 'RepeatContrast' for t in self.bpod_trials])
        trial_repeated = trial_repeated.astype(int)
        repNum = trial_repeated.copy()
        c = 0
        for i in range(len(trial_repeated)):
            if trial_repeated[i] == 0:
                c = 0
                repNum[i] = 0
                continue
            c += 1
            repNum[i] = c
        return repNum


class RewardVolume(BaseBpodTrialsExtractor):
    """
    Load reward volume delivered for each trial.
    **Optional:** saves _ibl_trials.rewardVolume.npy

    Uses reward_current to accumulate the amount of
    """
    save_names = '_ibl_trials.rewardVolume.npy'
    var_names = 'rewardVolume'

    def _extract(self):
        trial_volume = [x['reward_amount']
                        if x['trial_correct'] else 0 for x in self.bpod_trials]
        reward_volume = np.array(trial_volume).astype(np.float64)
        assert len(reward_volume) == len(self.bpod_trials)
        return reward_volume


class FeedbackTimes(BaseBpodTrialsExtractor):
    """
    Get the times the water or error tone was delivered to the animal.
    **Optional:** saves _ibl_trials.feedback_times.npy

    Gets reward  and error state init times vectors,
    checks if theintersection of nans is empty, then
    merges the 2 vectors.
    """
    save_names = '_ibl_trials.feedback_times.npy'
    var_names = 'feedback_times'

    @staticmethod
    def get_feedback_times_lt5(session_path, data=False):
        if not data:
            data = raw.load_data(session_path)
        rw_times = [tr['behavior_data']['States timestamps']['reward'][0][0]
                    for tr in data]
        err_times = [tr['behavior_data']['States timestamps']['error'][0][0]
                     for tr in data]
        nogo_times = [tr['behavior_data']['States timestamps']['no_go'][0][0]
                      for tr in data]
        assert sum(np.isnan(rw_times) &
                   np.isnan(err_times) & np.isnan(nogo_times)) == 0
        merge = np.array([np.array(times)[~np.isnan(times)] for times in
                          zip(rw_times, err_times, nogo_times)]).squeeze()

        return np.array(merge)

    @staticmethod
    def get_feedback_times_ge5(session_path, data=False):
        # ger err and no go trig times -- look for BNC2High of trial -- verify
        # only 2 onset times go tone and noise, select 2nd/-1 OR select the one
        # that is grater than the nogo or err trial onset time
        if not data:
            data = raw.load_data(session_path)
        missed_bnc2 = 0
        rw_times, err_sound_times, merge = [np.zeros([len(data), ]) for _ in range(3)]

        for ind, tr in enumerate(data):
            st = tr['behavior_data']['Events timestamps'].get('BNC2High', None)
            if not st:
                st = np.array([np.nan, np.nan])
                missed_bnc2 += 1
            # xonar soundcard duplicates events, remove consecutive events too close together
            st = np.delete(st, np.where(np.diff(st) < 0.020)[0] + 1)
            rw_times[ind] = tr['behavior_data']['States timestamps']['reward'][0][0]
            # get the error sound only if the reward is nan
            err_sound_times[ind] = st[-1] if st.size >= 2 and np.isnan(rw_times[ind]) else np.nan
        if missed_bnc2 == len(data):
            _logger.warning('No BNC2 for feedback times, filling error trials NaNs')
        merge *= np.nan
        merge[~np.isnan(rw_times)] = rw_times[~np.isnan(rw_times)]
        merge[~np.isnan(err_sound_times)] = err_sound_times[~np.isnan(err_sound_times)]

        return merge

    def _extract(self):
        # Version check
        if version.ge(self.settings['IBLRIG_VERSION_TAG'], '5.0.0'):
            merge = self.get_feedback_times_ge5(self.session_path, data=self.bpod_trials)
        else:
            merge = self.get_feedback_times_lt5(self.session_path, data=self.bpod_trials)
        return np.array(merge)


class Intervals(BaseBpodTrialsExtractor):
    """
    Trial start to trial end. Trial end includes 1 or 2 seconds after feedback,
    (depending on the feedback) and 0.5 seconds of iti.
    **Optional:** saves _ibl_trials.intervals.npy

    Uses the corrected Trial start and Trial end timpestamp values form PyBpod.
    """
    save_names = '_ibl_trials.intervals.npy'
    var_names = 'intervals'

    def _extract(self):
        starts = [t['behavior_data']['Trial start timestamp'] for t in self.bpod_trials]
        ends = [t['behavior_data']['Trial end timestamp'] for t in self.bpod_trials]
        return np.array([starts, ends]).T


class ResponseTimes(BaseBpodTrialsExtractor):
    """
    Time (in absolute seconds from session start) when a response was recorded.
    **Optional:** saves _ibl_trials.response_times.npy

    Uses the timestamp of the end of the closed_loop state.
    """
    save_names = '_ibl_trials.response_times.npy'
    var_names = 'response_times'

    def _extract(self):
        rt = np.array([tr['behavior_data']['States timestamps']['closed_loop'][0][1]
                       for tr in self.bpod_trials])
        return rt


class ItiDuration(BaseBpodTrialsExtractor):
    """
    Calculate duration of iti from state timestamps.
    **Optional:** saves _ibl_trials.iti_duration.npy

    Uses Trial end timestamp and get_response_times to calculate iti.
    """
    save_names = '_ibl_trials.itiDuration.npy'
    var_names = 'iti_dur'

    def _extract(self):
        rt, _ = ResponseTimes(self.session_path).extract(
            save=False, bpod_trials=self.bpod_trials, settings=self.settings)
        ends = np.array([t['behavior_data']['Trial end timestamp'] for t in self.bpod_trials])
        iti_dur = ends - rt
        return iti_dur


class GoCueTriggerTimes(BaseBpodTrialsExtractor):
    """
    Get trigger times of goCue from state machine.

    Current software solution for triggering sounds uses PyBpod soft codes.
    Delays can be in the order of 10's of ms. This is the time when the command
    to play the sound was executed. To measure accurate time, either getting the
    sound onset from xonar soundcard sync pulse (latencies may vary).
    """
    save_names = '_ibl_trials.goCueTrigger_times.npy'
    var_names = 'goCueTrigger_times'

    def _extract(self):
        if version.ge(self.settings['IBLRIG_VERSION_TAG'], '5.0.0'):
            goCue = np.array([tr['behavior_data']['States timestamps']
                              ['play_tone'][0][0] for tr in self.bpod_trials])
        else:
            goCue = np.array([tr['behavior_data']['States timestamps']
                             ['closed_loop'][0][0] for tr in self.bpod_trials])
        return goCue


class TrialType(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.type.npy'
    var_name = 'trial_type'

    def _extract(self):
        trial_type = []
        for tr in self.bpod_trials:
            if ~np.isnan(tr["behavior_data"]["States timestamps"]["reward"][0][0]):
                trial_type.append(1)
            elif ~np.isnan(tr["behavior_data"]["States timestamps"]["error"][0][0]):
                trial_type.append(-1)
            elif ~np.isnan(tr["behavior_data"]["States timestamps"]["no_go"][0][0]):
                trial_type.append(0)
            else:
                _logger.warning("Trial is not in set {-1, 0, 1}, appending NaN to trialType")
                trial_type.append(np.nan)
        return np.array(trial_type)


class GoCueTimes(BaseBpodTrialsExtractor):
    """
    Get trigger times of goCue from state machine.

    Current software solution for triggering sounds uses PyBpod soft codes.
    Delays can be in the order of 10-100s of ms. This is the time when the command
    to play the sound was executed. To measure accurate time, either getting the
    sound onset from the future microphone OR the new xonar soundcard and
    setup developed by Sanworks guarantees a set latency (in testing).
    """
    save_names = '_ibl_trials.goCue_times.npy'
    var_names = 'goCue_times'

    def _extract(self):
        go_cue_times = np.zeros([len(self.bpod_trials), ])
        for ind, tr in enumerate(self.bpod_trials):
            if raw.get_port_events(tr, 'BNC2'):
                bnchigh = tr['behavior_data']['Events timestamps'].get('BNC2High', None)
                if bnchigh:
                    go_cue_times[ind] = bnchigh[0]
                    continue
                bnclow = tr['behavior_data']['Events timestamps'].get('BNC2Low', None)
                if bnclow:
                    go_cue_times[ind] = bnclow[0] - 0.1
                    continue
                go_cue_times[ind] = np.nan
            else:
                go_cue_times[ind] = np.nan

        nmissing = np.sum(np.isnan(go_cue_times))
        # Check if all stim_syncs have failed to be detected
        if np.all(np.isnan(go_cue_times)):
            _logger.warning(
                f'{self.session_path}: Missing ALL !! BNC2 TTLs ({nmissing} trials)')
        # Check if any stim_sync has failed be detected for every trial
        elif np.any(np.isnan(go_cue_times)):
            _logger.warning(f'{self.session_path}: Missing BNC2 TTLs on {nmissing} trials')

        return go_cue_times


class IncludedTrials(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.included.npy'
    var_names = 'included'

    def _extract(self):
        if version.ge(self.settings['IBLRIG_VERSION_TAG'], '5.0.0'):
            trials_included = self.get_included_trials_ge5(
                data=self.bpod_trials, settings=self.settings)
        else:
            trials_included = self.get_included_trials_lt5(data=self.bpod_trials)
        return trials_included

    @staticmethod
    def get_included_trials_lt5(data=False):
        trials_included = np.array([True for t in data])
        return trials_included

    @staticmethod
    def get_included_trials_ge5(data=False, settings=False):
        trials_included = np.array([True for t in data])
        if ('SUBJECT_DISENGAGED_TRIGGERED' in settings.keys() and settings[
                'SUBJECT_DISENGAGED_TRIGGERED'] is not False):
            idx = settings['SUBJECT_DISENGAGED_TRIALNUM'] - 1
            trials_included[idx:] = False
        return trials_included


class ItiInTimes(BaseBpodTrialsExtractor):
    var_names = 'itiIn_times'

    def _extract(self):
        if parse_version(self.settings["IBLRIG_VERSION_TAG"]) < parse_version("5.0.0"):
            iti_in = np.ones(len(self.bpod_trials)) * np.nan
        else:
            iti_in = np.array(
                [tr["behavior_data"]["States timestamps"]
                 ["exit_state"][0][0] for tr in self.bpod_trials]
            )
        return iti_in


class ErrorCueTriggerTimes(BaseBpodTrialsExtractor):
    var_names = 'errorCueTrigger_times'

    def _extract(self):
        errorCueTrigger_times = np.zeros(len(self.bpod_trials)) * np.nan
        for i, tr in enumerate(self.bpod_trials):
            nogo = tr["behavior_data"]["States timestamps"]["no_go"][0][0]
            error = tr["behavior_data"]["States timestamps"]["error"][0][0]
            if np.all(~np.isnan(nogo)):
                errorCueTrigger_times[i] = nogo
            elif np.all(~np.isnan(error)):
                errorCueTrigger_times[i] = error
        return errorCueTrigger_times


class StimFreezeTriggerTimes(BaseBpodTrialsExtractor):
    var_names = 'stimFreezeTrigger_times'

    def _extract(self):
        if parse_version(self.settings["IBLRIG_VERSION_TAG"]) < parse_version("6.2.5"):
            return np.ones(len(self.bpod_trials)) * np.nan
        freeze_reward = np.array(
            [
                True
                if np.all(~np.isnan(tr["behavior_data"]["States timestamps"]["freeze_reward"][0]))
                else False
                for tr in self.bpod_trials
            ]
        )
        freeze_error = np.array(
            [
                True
                if np.all(~np.isnan(tr["behavior_data"]["States timestamps"]["freeze_error"][0]))
                else False
                for tr in self.bpod_trials
            ]
        )
        no_go = np.array(
            [
                True
                if np.all(~np.isnan(tr["behavior_data"]["States timestamps"]["no_go"][0]))
                else False
                for tr in self.bpod_trials
            ]
        )
        assert (np.sum(freeze_error) + np.sum(freeze_reward) +
                np.sum(no_go) == len(self.bpod_trials))
        stimFreezeTrigger = np.array([])
        for r, e, n, tr in zip(freeze_reward, freeze_error, no_go, self.bpod_trials):
            if n:
                stimFreezeTrigger = np.append(stimFreezeTrigger, np.nan)
                continue
            state = "freeze_reward" if r else "freeze_error"
            stimFreezeTrigger = np.append(
                stimFreezeTrigger, tr["behavior_data"]["States timestamps"][state][0][0]
            )
        return stimFreezeTrigger


class StimOffTriggerTimes(BaseBpodTrialsExtractor):
    var_names = 'stimOffTrigger_times'

    def _extract(self):
        if parse_version(self.settings["IBLRIG_VERSION_TAG"]) >= parse_version("6.2.5"):
            stim_off_trigger_state = "hide_stim"
        elif parse_version(self.settings["IBLRIG_VERSION_TAG"]) >= parse_version("5.0.0"):
            stim_off_trigger_state = "exit_state"
        else:
            stim_off_trigger_state = "trial_start"

        stimOffTrigger_times = np.array(
            [tr["behavior_data"]["States timestamps"][stim_off_trigger_state][0][0]
             for tr in self.bpod_trials]
        )
        # If pre version 5.0.0 no specific nogo Off trigger was given, just return trial_starts
        if stim_off_trigger_state == "trial_start":
            return stimOffTrigger_times

        no_goTrigger_times = np.array(
            [tr["behavior_data"]["States timestamps"]["no_go"][0][0] for tr in self.bpod_trials]
        )
        # Stim off trigs are either in their own state or in the no_go state if the
        # mouse did not move, if the stim_off_trigger_state always exist
        # (exit_state or trial_start)
        # no NaNs will happen, NaNs might happen in at last trial if
        # session was stopped after response
        # if stim_off_trigger_state == "hide_stim":
        #     assert all(~np.isnan(no_goTrigger_times) == np.isnan(stimOffTrigger_times))
        # Patch with the no_go states trig times
        stimOffTrigger_times[~np.isnan(no_goTrigger_times)] = no_goTrigger_times[
            ~np.isnan(no_goTrigger_times)
        ]
        return stimOffTrigger_times


class StimOnTriggerTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimOnTrigger_times.npy'
    var_names = 'stimOnTrigger_times'

    def _extract(self):
        # Get the stim_on_state that triggers the onset of the stim
        stim_on_state = np.array([tr['behavior_data']['States timestamps']
                                 ['stim_on'][0] for tr in self.bpod_trials])
        return stim_on_state[:, 0].T


class StimOnTimes(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.stimOn_times.npy'
    var_names = 'stimOn_times'

    def _extract(self):
        """
        Find the time of the state machine command to turn on the stim
        (state stim_on start or rotary_encoder_event2)
        Find the next frame change from the photodiode after that TS.
        Screen is not displaying anything until then.
        (Frame changes are in BNC1 High and BNC1 Low)
        """
        # Version check
        if version.ge(self.settings['IBLRIG_VERSION_TAG'], '5.0.0'):
            stimOn_times = self.get_stimOn_times_ge5(self.session_path, data=self.bpod_trials)
        else:
            stimOn_times = self.get_stimOn_times_lt5(self.session_path, data=self.bpod_trials)
        return np.array(stimOn_times)

    @staticmethod
    def get_stimOn_times_ge5(session_path, data=False):
        """
        Find first and last stim_sync pulse of the trial.
        stimOn_times should be the first after the stim_on state.
        (Stim updates are in BNC1High and BNC1Low - frame2TTL device)
        Check that all trials have frame changes.
        Find length of stim_on_state [start, stop].
        If either check fails the HW device failed to detect the stim_sync square change
        Substitute that trial's missing or incorrect value with a NaN.
        return stimOn_times
        """
        if not data:
            data = raw.load_data(session_path)
        # Get all stim_sync events detected
        stim_sync_all = [raw.get_port_events(tr, 'BNC1') for tr in data]
        stim_sync_all = [np.array(x) for x in stim_sync_all]
        # Get the stim_on_state that triggers the onset of the stim
        stim_on_state = np.array([tr['behavior_data']['States timestamps']
                                 ['stim_on'][0] for tr in data])

        stimOn_times = np.array([])
        for sync, on, off in zip(
                stim_sync_all, stim_on_state[:, 0], stim_on_state[:, 1]):
            pulse = sync[np.where(np.bitwise_and((sync > on), (sync <= off)))]
            if pulse.size == 0:
                stimOn_times = np.append(stimOn_times, np.nan)
            else:
                stimOn_times = np.append(stimOn_times, pulse)

        nmissing = np.sum(np.isnan(stimOn_times))
        # Check if all stim_syncs have failed to be detected
        if np.all(np.isnan(stimOn_times)):
            _logger.error(f'{session_path}: Missing ALL BNC1 TTLs ({nmissing} trials)')

        # Check if any stim_sync has failed be detected for every trial
        if np.any(np.isnan(stimOn_times)):
            _logger.warning(f'{session_path}: Missing BNC1 TTLs on {nmissing} trials')

        return stimOn_times

    @staticmethod
    def get_stimOn_times_lt5(session_path, data=False):
        """
        Find the time of the statemachine command to turn on hte stim
        (state stim_on start or rotary_encoder_event2)
        Find the next frame change from the photodiodeafter that TS.
        Screen is not displaying anything until then.
        (Frame changes are in BNC1High and BNC1Low)
        """
        if not data:
            data = raw.load_data(session_path)
        stim_on = []
        bnc_h = []
        bnc_l = []
        for tr in data:
            stim_on.append(tr['behavior_data']['States timestamps']['stim_on'][0][0])
            if 'BNC1High' in tr['behavior_data']['Events timestamps'].keys():
                bnc_h.append(np.array(tr['behavior_data']
                                      ['Events timestamps']['BNC1High']))
            else:
                bnc_h.append(np.array([np.NINF]))
            if 'BNC1Low' in tr['behavior_data']['Events timestamps'].keys():
                bnc_l.append(np.array(tr['behavior_data']
                                      ['Events timestamps']['BNC1Low']))
            else:
                bnc_l.append(np.array([np.NINF]))

        stim_on = np.array(stim_on)
        bnc_h = np.array(bnc_h, dtype=object)
        bnc_l = np.array(bnc_l, dtype=object)

        count_missing = 0
        stimOn_times = np.zeros_like(stim_on)
        for i in range(len(stim_on)):
            hl = np.sort(np.concatenate([bnc_h[i], bnc_l[i]]))
            stot = hl[hl > stim_on[i]]
            if np.size(stot) == 0:
                stot = np.array([np.nan])
                count_missing += 1
            stimOn_times[i] = stot[0]

        if np.all(np.isnan(stimOn_times)):
            _logger.error(f'{session_path}: Missing ALL BNC1 TTLs ({count_missing} trials)')

        if count_missing > 0:
            _logger.warning(f'{session_path}: Missing BNC1 TTLs on {count_missing} trials')

        return np.array(stimOn_times)


class StimOnOffFreezeTimes(BaseBpodTrialsExtractor):
    """
    Extracts stim on / off and freeze times from Bpod BNC1 detected fronts
    """
    save_names = ("_ibl_trials.stimOn_times.npy", None, None)
    var_names = ('stimOn_times', 'stimOff_times', 'stimFreeze_times')

    def _extract(self):
        choice = Choice(self.session_path).extract(
            bpod_trials=self.bpod_trials, settings=self.settings, save=False
        )[0]
        f2TTL = [raw.get_port_events(tr, name="BNC1") for tr in self.bpod_trials]

        stimOn_times = np.array([])
        stimOff_times = np.array([])
        stimFreeze_times = np.array([])
        for tr in f2TTL:
            if tr and len(tr) == 2:
                stimOn_times = np.append(stimOn_times, tr[0])
                stimOff_times = np.append(stimOff_times, tr[-1])
                stimFreeze_times = np.append(stimFreeze_times, np.nan)
            elif tr and len(tr) >= 3:
                stimOn_times = np.append(stimOn_times, tr[0])
                stimOff_times = np.append(stimOff_times, tr[-1])
                stimFreeze_times = np.append(stimFreeze_times, tr[-2])
            else:
                stimOn_times = np.append(stimOn_times, np.nan)
                stimOff_times = np.append(stimOff_times, np.nan)
                stimFreeze_times = np.append(stimFreeze_times, np.nan)

        # In no_go trials no stimFreeze happens jsut stim Off
        stimFreeze_times[choice == 0] = np.nan
        # Check for trigger times
        # 2nd order criteria:
        # stimOn -> Closest one to stimOnTrigger?
        # stimOff -> Closest one to stimOffTrigger?
        # stimFreeze -> Closest one to stimFreezeTrigger?

        return stimOn_times, stimOff_times, stimFreeze_times


def extract_all(session_path, save=False, bpod_trials=None, settings=None):
    if not bpod_trials:
        bpod_trials = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    base = [FeedbackType, ContrastLR, ProbabilityLeft, Choice, RepNum, RewardVolume,
            FeedbackTimes, Intervals, ResponseTimes, GoCueTriggerTimes, GoCueTimes]
    # Version check
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        base.extend([StimOnTriggerTimes, StimOnOffFreezeTimes, ItiInTimes,
                     StimOffTriggerTimes, StimFreezeTriggerTimes, ErrorCueTriggerTimes])
    else:
        base.extend([IncludedTrials, ItiDuration, StimOnTimes])

    out, fil = run_extractor_classes(
        base, save=save, session_path=session_path, bpod_trials=bpod_trials, settings=settings)
    return out, fil

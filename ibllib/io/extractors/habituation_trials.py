"""Habituation ChoiceWorld Bpod trials extraction."""

import logging
import numpy as np

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
from ibllib.io.extractors.biased_trials import ContrastLR
from ibllib.io.extractors.training_trials import FeedbackTimes, StimOnTriggerTimes, GoCueTimes

_logger = logging.getLogger(__name__)


class HabituationTrials(BaseBpodTrialsExtractor):
    var_names = (
        'feedbackType',
        'rewardVolume',
        'stimOff_times',
        'contrastLeft',
        'contrastRight',
        'feedback_times',
        'stimOn_times',
        'stimOnTrigger_times',
        'intervals',
        'goCue_times',
        'goCueTrigger_times',
        'itiIn_times',
        'stimOffTrigger_times',
        'stimCenterTrigger_times',
        'stimCenter_times',
        'position',
        'phase',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exclude = ['itiIn_times', 'stimCenter_times', 'stimCenterTrigger_times', 'position', 'phase']
        self.save_names = tuple(f'_ibl_trials.{x}.npy' if x not in exclude else None for x in self.var_names)

    def _extract(self) -> dict:
        """
        Extract the Bpod trial events.

        For iblrig versions < 8.17 the Bpod state machine for this task had extremely misleading names!
        The 'iti' state was actually the delay between valve close and trial end (the stimulus is
        still present during this period), and the 'trial_start' state is actually the ITI during
        which there is a 1s Bpod TTL and gray screen period.

        In version 8.17 and later, the 'iti' state was renamed to 'post_reward' and 'trial_start'
        was renamed to 'iti'. NB: Some development builds reporting earlier versions (e.g. 8.15.5)
        also use the new names, therefore the state names are used to determine which applies.

        In version 8.30 and later, the grey screen period was moved to the end of the trial: the
        trial starts at the stimulus onset trigger and ends with the 'hide_stim' and 'exit_state'
        states, as in trainingChoiceWorld (see `_extract_trial_events`).

        Returns
        -------
        dict
            A dictionary of Bpod trial events. The keys are defined in the `var_names` attribute.
        """
        # Extract all trials...

        # Get all detected TTLs. These are stored for QC purposes
        self.frame2ttl, self.audio = raw.load_bpod_fronts(self.session_path, data=self.bpod_trials)
        # These are the frame2TTL pulses as a list of lists, one per trial
        ttls = [raw.get_port_events(tr, 'BNC1') for tr in self.bpod_trials]

        # Report missing events
        n_missing = sum(len(pulses) != 3 for pulses in ttls)
        # Check if all stim syncs have failed to be detected
        if n_missing == len(ttls):
            _logger.error(f'{self.session_path}: Missing ALL BNC1 TTLs ({n_missing} trials)')
        elif n_missing > 0:  # Check if any stim_sync has failed be detected for every trial
            _logger.warning(f'{self.session_path}: Missing BNC1 TTLs on {n_missing} trial(s)')

        # Extract datasets common to trainingChoiceWorld
        training = [ContrastLR, FeedbackTimes, GoCueTimes, StimOnTriggerTimes]
        out, _ = run_extractor_classes(
            training,
            session_path=self.session_path,
            save=False,
            bpod_trials=self.bpod_trials,
            settings=self.settings,
            task_collection=self.task_collection,
        )

        # RewardVolume
        trial_volume = [x['reward_amount'] for x in self.bpod_trials]
        out['rewardVolume'] = np.array(trial_volume).astype(np.float64)

        # FeedbackType is always positive
        out['feedbackType'] = np.ones(len(out['feedback_times']), dtype=np.int8)

        # Phase and position
        out['position'] = np.array([t['position'] for t in self.bpod_trials])
        out['phase'] = np.array([t['stim_phase'] for t in self.bpod_trials])

        if self.has_exit_state(self.bpod_trials):
            out.update(self._extract_trial_events(ttls))
            return {k: out[k] for k in self.var_names}

        """
        The 'trial_start'/'iti' state is in fact the 1s grey screen period, therefore the first
        timestamp is really the end of the previous trial and also the stimOff trigger time. The
        second timestamp is the true trial start time. This state was renamed in version 8.17.
        """
        state_names = self.bpod_trials[0]['behavior_data']['States timestamps'].keys()
        key = 'iti' if 'post_reward' in state_names else 'trial_start'
        (_, *ends), starts = zip(*[t['behavior_data']['States timestamps'][key][-1] for t in self.bpod_trials])

        # StimOffTrigger times
        out['stimOffTrigger_times'] = np.array(ends)

        # StimOff times
        """
        There should be exactly three TTLs per trial.  stimOff_times should be the first TTL pulse.
        If 1 or more pulses are missing, we can not be confident of assigning the correct one.
        """
        out['stimOff_times'] = np.array([sync[0] if len(sync) == 3 else np.nan for sync in ttls[1:]])

        # Trial intervals
        """
        In terms of TTLs, the intervals are defined by the 'trial_start' state, however the stim
        off time often happens after the trial end TTL front, i.e. after the 'trial_start' start
        begins.  For these trials, we set the trial end time as the stim off time.
        """
        # NB: We lose the last trial because the stim off event occurs at trial_num + 1
        n_trials = out['stimOff_times'].size
        out['intervals'] = np.c_[starts, np.r_[ends, np.nan]][:n_trials, :]

        to_correct = ~np.isnan(out['stimOff_times']) & (out['stimOff_times'] > out['intervals'][:, 1])
        if np.any(to_correct):
            _logger.debug(
                '%i/%i stim off events occurring outside trial intervals; using stim off times as trial end',
                sum(to_correct),
                len(to_correct),
            )
            out['intervals'][to_correct, 1] = out['stimOff_times'][to_correct]

        # itiIn times
        out['itiIn_times'] = np.r_[ends, np.nan]

        # GoCueTriggerTimes is the same event as StimOnTriggerTimes
        out['goCueTrigger_times'] = out['stimOnTrigger_times'].copy()

        # StimCenterTrigger times
        # Get the stim_on_state that triggers the onset of the stim
        stim_center_state = np.array([tr['behavior_data']['States timestamps']['stim_center'][0] for tr in self.bpod_trials])
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

        # Double-check that the early and late trial events occur within the trial intervals
        idx = ~np.isnan(out['stimOn_times'][:n_trials])
        assert not np.any(out['stimOn_times'][:n_trials][idx] < out['intervals'][idx, 0]), (
            'Stim on events occurring outside trial intervals'
        )

        # Truncate arrays and return in correct order
        return {k: out[k][:n_trials] for k in self.var_names}

    @staticmethod
    def has_exit_state(bpod_trials) -> bool:
        """Return True if the trials end with the 'hide_stim' and 'exit_state' states.

        From iblrig v8.30.0 the habituationChoiceWorld state machine follows the same structure as
        trainingChoiceWorld: each trial begins with the stimulus onset ('stim_on') and ends with
        the stimulus offset ('hide_stim') followed by a 0.5 s 'exit_state', during which the Bpod
        TTL is high. The remainder of the ITI happens between state machines.

        Parameters
        ----------
        bpod_trials : list of dict
            The raw Bpod trial data.

        Returns
        -------
        bool
            True if the first trial contains both 'hide_stim' and 'exit_state' states.
        """
        state_names = bpod_trials[0]['behavior_data']['States timestamps'].keys()
        return {'hide_stim', 'exit_state'}.issubset(state_names)

    def _extract_trial_events(self, ttls) -> dict:
        """
        Extract the trial events for the iblrig >= v8.30.0 state machine.

        The states of each trial are 'stim_on' (Bpod TTL high until the stimulus is detected),
        'play_tone', 'stim_center', 'reward' (Bpod TTL high), 'post_reward', 'hide_stim', and
        'exit_state' (Bpod TTL high for 0.5 s). Unlike previous versions, each trial starts with
        the stimulus onset trigger and ends after the stimulus offset, so all trial events occur
        within the Bpod trial and no trials are dropped. Each stimulus event is the first BNC1
        (frame2ttl) front after its trigger state starts and before the next trigger state.

        Parameters
        ----------
        ttls : list of list of float
            The BNC1 (frame2ttl) event times for each trial.

        Returns
        -------
        dict
            The trial intervals, stimulus event and trigger times, go cue trigger and itiIn times.
        """

        def state_start(name):
            return np.array([tr['behavior_data']['States timestamps'][name][0][0] for tr in self.bpod_trials])

        out = {
            'intervals': np.array([
                [tr['behavior_data']['Trial start timestamp'], tr['behavior_data']['Trial end timestamp']]
                for tr in self.bpod_trials
            ]),
            'stimOnTrigger_times': state_start('stim_on'),
            'goCueTrigger_times': state_start('play_tone'),
            'stimCenterTrigger_times': state_start('stim_center'),
            'stimOffTrigger_times': state_start('hide_stim'),
            'itiIn_times': state_start('exit_state'),
        }

        def first_front(start, end, inclusive=False):
            """Return the first TTL of each trial occurring after start and before end."""
            times = np.full(len(ttls), np.nan)
            for i, (sync, t0, t1) in enumerate(zip(ttls, start, end)):
                sync = np.asarray(sync)
                (idx,) = np.where((sync > t0) & ((sync <= t1) if inclusive else (sync < t1)))
                if idx.size > 0:
                    times[i] = sync[idx[0]]
            return times

        out['stimOn_times'] = first_front(out['stimOnTrigger_times'], out['stimCenterTrigger_times'])
        out['stimCenter_times'] = first_front(out['stimCenterTrigger_times'], out['stimOffTrigger_times'])
        out['stimOff_times'] = first_front(out['stimOffTrigger_times'], out['intervals'][:, 1], inclusive=True)
        return out

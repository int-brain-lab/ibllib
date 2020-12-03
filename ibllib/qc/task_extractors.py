import logging

import numpy as np

from ibllib.io.extractors.training_trials import (
    StimOnOffFreezeTimes, Choice, FeedbackType, Intervals, StimOnTriggerTimes, StimOnTimes,
    StimOffTriggerTimes, StimFreezeTriggerTimes, GoCueTriggerTimes, GoCueTimes,
    ErrorCueTriggerTimes, RewardVolume, ResponseTimes, FeedbackTimes, ItiInTimes,
    ProbabilityLeft, run_extractor_classes  # ContrastLR
)
import ibllib.io.extractors.habituation_trials as habit
from ibllib.io.extractors.training_wheel import Wheel, get_wheel_position
from ibllib.io.extractors.ephys_fpga import (
    _get_pregenerated_events, _get_main_probe_sync, bpod_fpga_sync, FpgaTrials
)
from ibllib.io.extractors.base import get_session_extractor_type
import ibllib.io.raw_data_loaders as raw
from alf.io import is_session_path
from oneibl.one import ONE


class TaskQCExtractor(object):
    def __init__(self, session_path, lazy=False, one=None, download_data=False, bpod_only=False):
        """
        A class for extracting the task data required to perform task quality control
        :param session_path: a valid session path
        :param lazy: if True, the data are not extracted immediately
        :param one: an instance of ONE, used to download the raw data if download_data is True
        :param download_data: if True, any missing raw data is downloaded via ONE
        :param bpod_only: extract from from raw Bpod data only, even for FPGA sessions
        """
        if not is_session_path(session_path):
            raise ValueError('Invalid session path')
        self.session_path = session_path
        self.one = one
        self.log = logging.getLogger("ibllib")

        self.data = None
        self.settings = None
        self.raw_data = None
        self.frame_ttls = self.audio_ttls = self.bpod_ttls = None
        self.type = None
        self.wheel_encoding = None
        self.bpod_only = bpod_only

        if download_data:
            self.one = one or ONE()
            self._ensure_required_data()

        if not lazy:
            self.load_raw_data()
            self.extract_data()

    def _ensure_required_data(self):
        """
        Attempt to download any required raw data if missing, and raise exception if any data are
        missing.
        :return:
        """
        dstypes = [
            "_iblrig_taskData.raw",
            "_iblrig_taskSettings.raw",
            "_iblrig_encoderPositions.raw",
            "_iblrig_encoderEvents.raw",
            "_iblrig_stimPositionScreen.raw",
            "_iblrig_syncSquareUpdate.raw",
            "_iblrig_encoderTrialInfo.raw",
            "_iblrig_ambientSensorData.raw",
        ]
        eid = self.one.eid_from_path(self.session_path)
        # Ensure we have the settings
        settings = self.one.load(eid, ["_iblrig_taskSettings.raw"], download_only=True)
        if settings and get_session_extractor_type(self.session_path) == 'ephys':
            dstypes.extend(['_spikeglx_sync.channels',
                            '_spikeglx_sync.polarities',
                            '_spikeglx_sync.times',
                            'ephysData.raw.meta',
                            'ephysData.raw.wiring'])
        self.log.info(f"Downloading data for session {eid}")
        files = self.one.load(eid, dataset_types=dstypes, download_only=True)
        missing = [True for _ in dstypes] if not files else [x is None for x in files]
        if self.session_path is None or all(missing):
            self.lazy = True
            self.log.error("Data not found on server, can't calculate QC.")
        elif any(missing):
            self.log.warning(
                f"Missing some datasets for session {eid} in path {self.session_path}"
            )

    def load_raw_data(self):
        """
        Loads the TTLs, raw task data and task settings
        :return:
        """
        self.log.info(f"Loading raw data from {self.session_path}")
        self.type = self.type or get_session_extractor_type(self.session_path)
        self.settings, self.raw_data = raw.load_bpod(self.session_path)
        # Fetch the TTLs for the photodiode and audio
        if self.type != 'ephys' or self.bpod_only is True:  # Extract from Bpod
            self.frame_ttls, self.audio_ttls = raw.load_bpod_fronts(
                self.session_path, data=self.raw_data)
        else:  # Extract from FPGA
            sync, chmap = _get_main_probe_sync(self.session_path)

            def channel_events(name):
                """Fetches the polarities and times for a given channel"""
                keys = ('polarities', 'times')
                mask = sync['channels'] == chmap[name]
                return dict(zip(keys, (sync[k][mask] for k in keys)))

            ttls = [channel_events(ch) for ch in ('frame2ttl', 'audio', 'bpod')]
            self.frame_ttls, self.audio_ttls, self.bpod_ttls = ttls

    def extract_data(self, partial=False):
        """Extracts and loads behaviour data for QC
        NB: partial extraction when bpod_only sttricbute is False requires intervals and
        intervals_bpod to be assigned to the data attribute before calling this function.
        :param partial: If True, extracts only the required data that aren't usually saved to ALFs
        :return:
        """
        self.log.info(f"Extracting session: {self.session_path}")
        self.type = self.type or get_session_extractor_type(self.session_path)
        self.wheel_encoding = 'X4' if (self.type == 'ephys' and not self.bpod_only) else 'X1'

        # Partial extraction for FPGA sessions only worth it if intervals already extracted and
        # assigned to the data attribute
        data_assigned = self.data and {'intervals', 'intervals_bpod'}.issubset(self.data)
        if partial and self.type == 'ephys' and not self.bpod_only and not data_assigned:
            partial = False  # Requires intervals for converting to FPGA time

        if not self.raw_data:
            self.load_raw_data()

        # Signals and parameters not usually saved to file
        if self.type == 'habituation':
            extractors = [habit.StimCenterTimes, habit.StimCenterTriggerTimes,
                          habit.ItiInTimes, habit.StimOffTriggerTimes]
        else:
            extractors = [
                StimOnTriggerTimes, StimOffTriggerTimes, StimOnOffFreezeTimes,
                StimFreezeTriggerTimes, ErrorCueTriggerTimes, ItiInTimes]

        # Extract the data that are usually saved to file;
        # this must be after the Bpod extractors in the list
        if not partial:
            if self.type == 'ephys' and not self.bpod_only:
                extractors.append(FpgaTrials)
            elif self.type == 'habituation':
                extractors.append(habit.HabituationTrials)
            else:
                extractors.extend([
                    Choice, FeedbackType, Intervals, StimOnTimes, GoCueTriggerTimes, Wheel,
                    GoCueTimes, RewardVolume, ResponseTimes, FeedbackTimes, ProbabilityLeft])
                # if type == 'biased':
                #     # FIXME ContrastLR fails on old sessions (contrast is a float, not a dict)
                #     extractors.append(ContrastLR)

        # Run behaviour extractors
        kwargs = dict(save=False, bpod_trials=self.raw_data, settings=self.settings)
        data, _ = run_extractor_classes(extractors, session_path=self.session_path, **kwargs)

        n_trials = np.unique(list(map(lambda k: data[k].shape[0], data)))[0]

        # Extract some parameters
        if self.type == 'ephys':
            # For ephys sessions extract quiescence and phase from pre-generated file
            data.update(_get_pregenerated_events(self.raw_data, self.settings))

            if not self.bpod_only:
                # Get the extracted intervals for sync.  For partial ephys extraction attempt to
                # get intervals from data attribute.
                intervals, intervals_bpod = [data[key] if key in data else self.data[key]
                                             for key in ('intervals', 'intervals_bpod')]
                # We need to sync the extra extracted data to FPGA time
                # 0.5s iti already removed during extraction so we set duration to 0 here
                ibpod, _, bpod2fpga = bpod_fpga_sync(intervals_bpod, intervals, iti_duration=0)
                # These fields have to be re-synced
                sync_fields = ['stimOnTrigger_times', 'stimOffTrigger_times', 'stimFreeze_times',
                               'stimFreezeTrigger_times', 'errorCueTrigger_times', 'itiIn_times']
                bpod_fields = ['probabilityLeft', 'contrastLeft', 'contrastRight', 'position',
                               'contrast', 'quiescence', 'phase']
                if partial:
                    # Remove any extraneous fields, i.e. bpod stimOn, stimOff
                    data = {k: v for k, v in data.items() if k in sync_fields + bpod_fields}
                # Build trials output
                data.update({k: bpod2fpga(data[k][ibpod]) for k in sync_fields})
                data.update({k: data[k][ibpod] for k in bpod_fields})
                # Add Bpod wheel data
                re_ts, pos = get_wheel_position(self.session_path, self.raw_data)
                data['wheel_timestamps_bpod'] = bpod2fpga(re_ts)
                data['wheel_position_bpod'] = pos

        elif self.type == 'habituation':
            data['position'] = np.array([t['position'] for t in self.raw_data])
            data['phase'] = np.array([t['stim_phase'] for t in self.raw_data])
            # Nasty hack to trim last trial due to stim off events happening at trial num + 1
            data = {k: v[:n_trials] for k, v in data.items()}
        else:
            data['quiescence'] = \
                np.array([t['quiescent_period'] for t in self.raw_data[:n_trials]])
            data['position'] = np.array([t['position'] for t in self.raw_data[:n_trials]])
            # FIXME Check this is valid for biased choiceWorld
            data['phase'] = np.array([t['stim_phase'] for t in self.raw_data[:n_trials]])

        # Update the data attribute with extracted data
        if self.data:
            self.data.update(data)
            self.rename_data(self.data)
        else:
            self.data = data if partial else self.rename_data(data)

    @staticmethod
    def rename_data(data):
        """Rename the extracted data dict for use with TaskQC
        Splits 'feedback_times' to 'errorCue_times' and 'valveOpen_times'.
        NB: The data is not copied before making changes
        :param data: A dict of task data returned by the task extractors
        :return: the same dict after modifying the keys
        """
        # get valve_time and errorCue_times from feedback_times
        correct = data['feedbackType'] > 0
        errorCue_times = data["feedback_times"].copy()
        valveOpen_times = data["feedback_times"].copy()
        errorCue_times[correct] = np.nan
        valveOpen_times[~correct] = np.nan
        data.update(
            {"errorCue_times": errorCue_times,
             "valveOpen_times": valveOpen_times,
             "correct": correct}
        )
        return data

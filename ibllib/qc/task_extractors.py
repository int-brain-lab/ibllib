import logging

import numpy as np

from ibllib.io.extractors.training_trials import (
    StimOnOffFreezeTimes, Choice, FeedbackType, Intervals, StimOnTriggerTimes, StimOnTimes,
    StimOffTriggerTimes, StimFreezeTriggerTimes, GoCueTriggerTimes, GoCueTimes,
    ErrorCueTriggerTimes, RewardVolume, ResponseTimes, FeedbackTimes, ItiInTimes,
    ProbabilityLeft, ContrastLR, run_extractor_classes
)
from ibllib.io.extractors.training_wheel import Wheel
from oneibl.one import ONE
from ibllib.io.extractors.ephys_fpga import _get_pregenerated_events, bpod_fpga_sync, FpgaTrials
import ibllib.io.raw_data_loaders as raw


class TaskQCExtractor(object):
    def __init__(self, session_path, lazy=False, one=None, ensure_data=True):
        self.session_path = session_path
        self.one = one or ONE()
        self.log = logging.getLogger("ibllib")

        self.data = None
        self.wheel_data = None
        self.settings = None
        self.raw_data = None
        self.BNC1 = self.BNC2 = None
        self.type = None
        self.wheel_encoding = None

        if ensure_data:
            self._ensure_required_data()

        if not lazy:
            self.load_raw_data()
            self.data = self.extract_data()

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
        if settings and raw.get_session_extractor_type(self.session_path) == 'ephys':
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
        self.log.info(f"Loading raw data from {self.session_path}")
        self.settings, self.raw_data = raw.load_bpod(self.session_path)
        self.BNC1, self.BNC2 = raw.load_bpod_fronts(self.session_path, data=self.raw_data)

    def extract_data(self, partial=False, bpod_only=False):
        """Extracts and loads behaviour data for QC
        NB: partial extraction only allowed for bpod only extraction
        :param partial: If True, returns only the required data that aren't usually saved to ALFs
        :param bpod_only: If False, FPGA data are extracted where available
        :return: dict of data required for the behaviour QC
        """
        self.log.info(f"Extracting session: {self.session_path}")
        self.type = raw.get_session_extractor_type(self.session_path)
        self.wheel_encoding = 'X4' if (self.type == 'ephys' and not bpod_only) else 'X1'

        if partial and self.type == 'ephys' and not bpod_only:
            partial = False  # Requires intervals for converting to FPGA time

        if not self.raw_data:
            self.load_raw_data()

        # Signals and parameters not usually saved to file, available for all task types
        extractors = [
            StimOnTriggerTimes, StimOffTriggerTimes, StimOnOffFreezeTimes,
            StimFreezeTriggerTimes, ErrorCueTriggerTimes, ItiInTimes]

        # Extract the data that are usually saved to file
        if not partial:
            if self.type == 'ephys' and not bpod_only:
                extractors.append(FpgaTrials)
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

        # Extract some parameters
        if self.type == 'ephys':
            # For ephys sessions extract quiescence and phase from pre-generated file
            data.update(_get_pregenerated_events(self.raw_data, self.settings))

            if not bpod_only:
                # if partial:
                    # FIXME not worth it
                    # sync, chmap = _get_main_probe_sync(self.session_path, bin_exists=False)
                    # state = self.raw_data[-1]['behavior_data']['States timestamps']['exit_state']
                    # tmax = state[0][-1] + 60
                    # fpga_trials = extract_behaviour_sync(sync=sync, chmap=chmap, tmax=tmax)
                    # data['intervals'] = fpga_trials['intervals']
                    # data['intervals_bpod'] = Intervals(self.session_path).extract(**kwargs)[0]
                # We need to sync the extra extracted data to FPGA time
                # 0.5s iti already removed during extraction so we set duration to 0 here
                ibpod, _, bpod2fpga = bpod_fpga_sync(
                    data['intervals_bpod'], data['intervals'], iti_duration=0)
                # These fields have to be re-synced
                sync_fields = ['stimOnTrigger_times', 'stimOffTrigger_times', 'stimFreeze_times',
                               'stimFreezeTrigger_times', 'errorCueTrigger_times', 'itiIn_times']
                # build trials output
                data.update({k: bpod2fpga(data[k][ibpod]) for k in sync_fields})
        else:
            data['quiescence'] = np.array([t['quiescent_period'] for t in self.raw_data])
            data['position'] = np.array([t['position'] for t in self.raw_data])
            self.wheel_encoding = 'X1'

        return data if partial else self.rename_data(data)

    @staticmethod
    def rename_data(data):
        # get valve_time and errorCue_times from feedback_times
        correct = data['feedbackType'] > 0
        errorCue_times = data["feedback_times"].copy()
        valveOpen_times = data["feedback_times"].copy()
        errorCue_times[correct] = np.nan
        valveOpen_times[~correct] = np.nan
        data.update(
            {"errorCue_times": errorCue_times, "valveOpen_times": valveOpen_times,
             "correct": correct}
        )
        # split intervals
        data["intervals_0"] = data["intervals"][:, 0]
        data["intervals_1"] = data["intervals"][:, 1]
        _ = data.pop("intervals")
        data["outcome"] = data["feedbackType"].copy()
        data["outcome"][data["choice"] == 0] = 0
        return data

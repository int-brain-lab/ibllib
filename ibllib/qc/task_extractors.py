import logging

import numpy as np
from scipy.interpolate import interp1d

from ibllib.io.extractors import bpod_trials
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io.extractors.training_wheel import get_wheel_position
from ibllib.io.extractors import ephys_fpga
import ibllib.io.raw_data_loaders as raw
from one.alf.spec import is_session_path
import one.alf.io as alfio
from one.api import ONE


_logger = logging.getLogger("ibllib")

REQUIRED_FIELDS = ['choice', 'contrastLeft', 'contrastRight', 'correct',
                   'errorCueTrigger_times', 'errorCue_times', 'feedbackType', 'feedback_times',
                   'firstMovement_times', 'goCueTrigger_times', 'goCue_times', 'intervals',
                   'itiIn_times', 'phase', 'position', 'probabilityLeft', 'quiescence',
                   'response_times', 'rewardVolume', 'stimFreezeTrigger_times',
                   'stimFreeze_times', 'stimOffTrigger_times', 'stimOff_times',
                   'stimOnTrigger_times', 'stimOn_times', 'valveOpen_times',
                   'wheel_moves_intervals', 'wheel_moves_peak_amplitude',
                   'wheel_position', 'wheel_timestamps']


class TaskQCExtractor(object):
    def __init__(self, session_path, lazy=False, one=None, download_data=False, bpod_only=False,
                 sync_collection=None, sync_type=None, task_collection=None, save_path=None):
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
        self.log = _logger

        self.data = None
        self.settings = None
        self.raw_data = None
        self.frame_ttls = self.audio_ttls = self.bpod_ttls = None
        self.type = None
        self.wheel_encoding = None
        self.bpod_only = bpod_only
        self.sync_collection = sync_collection or 'raw_ephys_data'
        self.sync_type = sync_type
        self.task_collection = task_collection or 'raw_behavior_data'
        self.save_path = save_path

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
        eid = self.one.path2eid(self.session_path)
        self.log.info(f"Downloading data for session {eid}")
        # Ensure we have the settings
        settings, _ = self.one.load_datasets(eid, ["_iblrig_taskSettings.raw.json"],
                                             collections=[self.task_collection],
                                             download_only=True, assert_present=False)

        is_ephys = get_session_extractor_type(self.session_path, task_collection=self.task_collection) == 'ephys'
        self.sync_type = self.sync_type or 'nidq' if is_ephys else 'bpod'
        is_fpga = 'bpod' not in self.sync_type

        if settings and is_ephys:

            dstypes.extend(['_spikeglx_sync.channels',
                            '_spikeglx_sync.polarities',
                            '_spikeglx_sync.times',
                            'ephysData.raw.meta',
                            'ephysData.raw.wiring'])
        elif settings and is_fpga:

            dstypes.extend(['_spikeglx_sync.channels',
                            '_spikeglx_sync.polarities',
                            '_spikeglx_sync.times',
                            'DAQData.raw.meta',
                            'DAQData.wiring'])

        dataset = self.one.type2datasets(eid, dstypes, details=True)
        files = self.one._check_filesystem(dataset)

        missing = [True] * len(dstypes) if not files else [x is None for x in files]
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
        self.type = self.type or get_session_extractor_type(self.session_path, task_collection=self.task_collection)
        # Finds the sync type when it isn't explicitly set, if ephys we assume nidq otherwise bpod
        self.sync_type = self.sync_type or 'nidq' if self.type == 'ephys' else 'bpod'

        self.settings, self.raw_data = raw.load_bpod(self.session_path, task_collection=self.task_collection)
        # Fetch the TTLs for the photodiode and audio
        if self.sync_type == 'bpod' or self.bpod_only is True:  # Extract from Bpod
            self.frame_ttls, self.audio_ttls = raw.load_bpod_fronts(
                self.session_path, data=self.raw_data, task_collection=self.task_collection)
        else:  # Extract from FPGA
            sync, chmap = ephys_fpga.get_sync_and_chn_map(self.session_path, self.sync_collection)

            def channel_events(name):
                """Fetches the polarities and times for a given channel"""
                keys = ('polarities', 'times')
                mask = sync['channels'] == chmap[name]
                return dict(zip(keys, (sync[k][mask] for k in keys)))

            ttls = [ephys_fpga._clean_frame2ttl(channel_events('frame2ttl')),
                    ephys_fpga._clean_audio(channel_events('audio')),
                    channel_events('bpod')]
            self.frame_ttls, self.audio_ttls, self.bpod_ttls = ttls

    def extract_data(self):
        """Extracts and loads behaviour data for QC
        NB: partial extraction when bpod_only attribute is False requires intervals and
        intervals_bpod to be assigned to the data attribute before calling this function.
        :return:
        """
        self.log.info(f"Extracting session: {self.session_path}")
        self.type = self.type or get_session_extractor_type(self.session_path, task_collection=self.task_collection)
        # Finds the sync type when it isn't explicitly set, if ephys we assume nidq otherwise bpod
        self.sync_type = self.sync_type or 'nidq' if self.type == 'ephys' else 'bpod'

        self.wheel_encoding = 'X4' if (self.sync_type != 'bpod' and not self.bpod_only) else 'X1'

        if not self.raw_data:
            self.load_raw_data()
        # Run extractors
        if self.sync_type != 'bpod' and not self.bpod_only:
            data, _ = ephys_fpga.extract_all(self.session_path, task_collection=self.task_collection, save_path=self.save_path)
            bpod2fpga = interp1d(data['intervals_bpod'][:, 0], data['table']['intervals_0'],
                                 fill_value='extrapolate')
            # Add Bpod wheel data
            re_ts, pos = get_wheel_position(self.session_path, self.raw_data, task_collection=self.task_collection)
            data['wheel_timestamps_bpod'] = bpod2fpga(re_ts)
            data['wheel_position_bpod'] = pos
        else:
            kwargs = dict(save=False, bpod_trials=self.raw_data, settings=self.settings,
                          task_collection=self.task_collection, save_path=self.save_path)
            trials, wheel, _ = bpod_trials.extract_all(self.session_path, **kwargs)
            n_trials = np.unique(list(map(lambda k: trials[k].shape[0], trials)))[0]
            if self.type == 'habituation':
                data = trials
                data['position'] = np.array([t['position'] for t in self.raw_data])
                data['phase'] = np.array([t['stim_phase'] for t in self.raw_data])
                # Nasty hack to trim last trial due to stim off events happening at trial num + 1
                data = {k: v[:n_trials] for k, v in data.items()}
            else:
                data = {**trials, **wheel}
        # Update the data attribute with extracted data
        self.data = self.rename_data(data)

    @staticmethod
    def rename_data(data):
        """Rename the extracted data dict for use with TaskQC
        Splits 'feedback_times' to 'errorCue_times' and 'valveOpen_times'.
        NB: The data is not copied before making changes
        :param data: A dict of task data returned by the task extractors
        :return: the same dict after modifying the keys
        """
        # Expand trials dataframe into key value pairs
        trials_table = data.pop('table', None)
        if trials_table is not None:
            data = {**data, **alfio.AlfBunch.from_df(trials_table)}
        correct = data['feedbackType'] > 0
        # get valve_time and errorCue_times from feedback_times
        if 'errorCue_times' not in data:
            data['errorCue_times'] = data["feedback_times"].copy()
            data['errorCue_times'][correct] = np.nan
        if 'valveOpen_times' not in data:
            data['valveOpen_times'] = data["feedback_times"].copy()
            data['valveOpen_times'][~correct] = np.nan
        data['correct'] = correct
        diff_fields = list(set(REQUIRED_FIELDS).difference(set(data.keys())))
        for miss_field in diff_fields:
            data[miss_field] = data["feedback_times"] * np.nan
        if len(diff_fields):
            _logger.warning(f"QC extractor, missing fields filled with NaNs: {diff_fields}")
        return data

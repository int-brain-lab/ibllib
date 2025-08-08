import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Optional
import pickle

import ibldsp.utils
import ibllib.io.session_params
from ibllib.pipes import base_tasks
from iblutil.io import jsonable

from nptdms import TdmsFile

from abc import abstractmethod
from iblphotometry import io as fpio
from iblutil.spacer import Spacer

_logger = logging.getLogger('ibllib')
_logger.setLevel(logging.DEBUG)


def extract_timestamps_from_tdms_file(tdms_filepath: Path, save_path: Optional[Path] = None) -> dict:
    # extractor for tdms files as written by the daqami software, configured
    # for neurophotometrics experiments: Frameclock is in AI7, DI1-4 are the
    # bpod sync signals
    _logger.info(f'extracting timestamps from tdms file: {tdms_filepath}')

    tdms_file = TdmsFile.read(tdms_filepath)
    groups = tdms_file.groups()
    # this unfortunate hack is in here because there are a bunch of sessions where the frameclock is on DI0
    if len(groups) == 1:
        has_analog_group = False
        (digital_group,) = groups
    if len(groups) == 2:
        has_analog_group = True
        analog_group, digital_group = groups
    fs = digital_group.properties['ScanRate']  # this should be 10kHz
    df = tdms_file.as_dataframe()
    col = df.columns[-1]
    vals = df[col].values.astype('int32')
    columns = ['DI0', 'DI1', 'DI2', 'DI3']

    # ugly but basically just a binary decoder for the binary data
    # assumes 4 channels
    data = np.array([list(bin(v)[2:].zfill(4)[::-1]) for v in vals], dtype='int32')
    timestamps = {}
    for i, name in enumerate(columns):
        timestamps[name] = np.where(np.diff(data[:, i]) == 1)[0] / fs

    if has_analog_group:
        # frameclock data is recorded on an analog channel
        for channel in analog_group.channels():
            signal = (channel.data > 2.5).astype('int32')  # assumes 0-5V
            timestamps[channel.name] = np.where(np.diff(signal) == 1)[0] / fs

    if save_path is not None:
        _logger.info(f'saving extracted timestamps to: {save_path}')
        with open(save_path, 'wb') as fH:
            pickle.dump(timestamps, fH)

    return timestamps


def extract_timestamps_from_tdms_file_fast(tdms_filepath: Path, save_path: Optional[Path] = None, chunk_size=10000) -> dict:
    # extractor for tdms files as written by the daqami software, configured
    # for neurophotometrics experiments: Frameclock is in AI7, DI1-4 are the
    # bpod sync signals
    _logger.info(f'extracting timestamps from tdms file: {tdms_filepath}')

    # this should be 10kHz
    tdms_file = TdmsFile.read(tdms_filepath)
    groups = tdms_file.groups()

    # this unfortunate hack is in here because there are a bunch of sessions
    # where the frameclock is on DI0
    if len(groups) == 1:
        has_analog_group = False
        (digital_group,) = groups
    if len(groups) == 2:
        has_analog_group = True
        analog_group, digital_group = groups
    fs = digital_group.properties['ScanRate']  # this should be 10kHz
    df = tdms_file.as_dataframe()

    # inferring digital col name
    (digital_col,) = [col for col in df.columns if 'Digital' in col]
    vals = df[digital_col].values.astype('int8')
    digital_channel_names = ['DI0', 'DI1', 'DI2', 'DI3']

    # ini
    timestamps = {}
    for ch in digital_channel_names:
        timestamps[ch] = []

    # chunked loop
    n_chunks = df.shape[0] // chunk_size
    for i in range(n_chunks):
        vals_ = vals[i * chunk_size: (i + 1) * chunk_size]
        data = np.array([list(f'{v:04b}'[::-1]) for v in vals_], dtype='int8')

        for j, name in enumerate(digital_channel_names):
            ix = np.where(np.diff(data[:, j]) == 1)[0] + (chunk_size * i)
            timestamps[name].append(ix / fs)

    for ch in digital_channel_names:
        timestamps[ch] = np.concatenate(timestamps[ch])

    if has_analog_group:
        # frameclock data is recorded on an analog channel
        for channel in analog_group.channels():
            signal = (channel.data > 2.5).astype('int32')  # assumes 0-5V
            timestamps[channel.name] = np.where(np.diff(signal) == 1)[0] / fs

    if save_path is not None:
        _logger.info(f'saving extracted timestamps to: {save_path}')
        with open(save_path, 'wb') as fH:
            pickle.dump(timestamps, fH)

    return timestamps


class FibrePhotometryBaseSync(base_tasks.DynamicTask):
    # base clas for syncing fibre photometry
    # derived classes are: FibrePhotometryBpodSync and FibrePhotometryDAQSync
    priority = 90
    job_size = 'small'

    def __init__(self, session_path, one, **kwargs):
        super().__init__(session_path, one=one, **kwargs)
        self.photometry_collection = kwargs.get('collection', 'raw_photometry_data')  # raw_photometry_data
        self.kwargs = kwargs

        # we will work with the first protocol here
        for task in self.session_params['tasks']:
            self.task_protocol = next(k for k in task)
            self.task_collection = ibllib.io.session_params.get_task_collection(self.session_params, self.task_protocol)
            break

    def _get_bpod_timestamps(self) -> Tuple[np.ndarray, list]:
        # the timestamps for syncing, in the time of the bpod
        if 'habituation' in self.task_protocol:
            sync_states_names = ['iti', 'reward']
        else:
            sync_states_names = ['trial_start', 'reward', 'exit_state']

        # read in the raw behaviour data for syncing
        file_jsonable = self.session_path.joinpath(self.task_collection, '_iblrig_taskData.raw.jsonable')
        _, bpod_data = jsonable.load_task_jsonable(file_jsonable)

        # we get the timestamps of the states from the bpod data
        timestamps_bpod = []
        for sync_name in sync_states_names:
            timestamps_bpod.append(
                np.array(
                    [
                        data['States timestamps'][sync_name][0][0] + data['Trial start timestamp']
                        for data in bpod_data
                        if sync_name in data['States timestamps']
                    ]
                )
            )
        timestamps_bpod = np.sort(np.concatenate(timestamps_bpod))
        timestamps_bpod = timestamps_bpod[~np.isnan(timestamps_bpod)]
        return timestamps_bpod, bpod_data

    @abstractmethod
    def _get_neurophotometrics_timestamps(self) -> np.ndarray:
        # this function needs to be implemented in the derived classes:
        # for bpod based syncing, the timestamps are in the digial inputs file
        # for daq based syncing, the timestamps are extracted from the tdms file
        ...

    def _get_sync_function(self, spacer_detection_mode='fast') -> Tuple[callable, list]:
        # returns the synchronization function
        # get the timestamps
        timestamps_bpod, bpod_data = self._get_bpod_timestamps()
        timestamps_nph = self._get_neurophotometrics_timestamps()

        # verify presence of sync timestamps
        for source, timestamps in zip(['bpod', 'neurophotometrics'], [timestamps_bpod, timestamps_nph]):
            assert len(timestamps) > 0, f'{source} sync timestamps are empty'

        # split into segments if multiple spacers are found
        # attempt to sync for each segment (only one will work)
        spacer = Spacer()

        # the fast way
        match spacer_detection_mode:
            case 'fast':
                spacer_ix = spacer.find_spacers_from_timestamps(timestamps_nph, atol=1e-5)
            case 'safe':
                spacer_ix, spacer_times = spacer.find_spacers_from_positive_fronts(timestamps_nph, fs=1000)

        # the indices that mark the boundaries of segments
        segment_ix = np.concatenate([spacer_ix, [timestamps_nph.shape[0]]])
        segments = []
        for i in range(segment_ix.shape[0] - 1):
            start_ix = segment_ix[i] + (spacer.n_pulses * 2) - 1
            stop_ix = segment_ix[i + 1]
            segments.append(timestamps_nph[start_ix:stop_ix])

        for i, timestamps_segment in enumerate(segments):
            # sync the behaviour events to the photometry timestamps
            try:
                sync_nph_to_bpod_fcn, drift_ppm, ix_nph, ix_bpod = ibldsp.utils.sync_timestamps(
                    timestamps_segment, timestamps_bpod, return_indices=True, linear=True
                )
            except ValueError:
                # this gets raised when there are no timestamps (multiple session restart)
                continue

            # then we check the alignment, should be less than the camera sampling rate
            tcheck = sync_nph_to_bpod_fcn(timestamps_segment[ix_nph]) - timestamps_bpod[ix_bpod]
            _logger.info(
                f'sync: n trials {len(bpod_data)}'
                f'n bpod sync {len(timestamps_bpod)}'
                f'n photometry {len(timestamps_segment)}, n match {len(ix_nph)}'
            )
            if len(ix_nph) / len(timestamps_bpod) < 0.95:
                # wrong segment
                print('wrong segment')
                continue
            # TODO the framerate here is hardcoded, infer it instead!
            assert np.all(np.abs(tcheck) < 1 / 60), 'Sync issue detected, residual above 1/60s'

        valid_bounds = [bpod_data[0]['Trial start timestamp'] - 2, bpod_data[-1]['Trial end timestamp'] + 2]

        return sync_nph_to_bpod_fcn, valid_bounds

    def load_data(self) -> pd.DataFrame:
        # loads the raw photometry data
        raw_photometry_folder = self.session_path / self.photometry_collection
        raw_neurophotometrics_df = pd.read_parquet(raw_photometry_folder / '_neurophotometrics_fpData.raw.pqt')
        return raw_neurophotometrics_df

    def _run(self, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # 1) load photometry data

        # note: when loading daq based syncing, the SystemTimestamp column
        # will be overridden with the timestamps from the tdms file
        # the idea behind this is that the rest of the sync is then the same
        # and handled by this base class
        raw_df = self.load_data()

        # 2) get the synchronization function
        spacer_detection_mode = kwargs.get('spacer_detection_mode', 'fast')
        sync_nph_to_bpod_fcn, valid_bounds = self._get_sync_function(spacer_detection_mode=spacer_detection_mode)

        # 3) convert to ibl_df
        ibl_df = fpio.from_raw_neurophotometrics_df_to_ibl_df(raw_df, rois=self.kwargs['fibers'], drop_first=False)

        # 3) apply synchronization
        ibl_df['times'] = sync_nph_to_bpod_fcn(raw_df['SystemTimestamp'])
        ibl_df['valid'] = np.logical_and(ibl_df['times'] >= valid_bounds[0], ibl_df['times'] <= valid_bounds[1])

        # 4) write to disk
        output_folder = self.session_path.joinpath('alf', 'photometry')
        output_folder.mkdir(parents=True, exist_ok=True)

        # writing the synced photometry signal
        ibl_df_outpath = output_folder / 'photometry.signal.pqt'
        ibl_df.to_parquet(ibl_df_outpath)

        # writing the locations
        rois = []
        for k, v in self.kwargs['fibers'].items():
            rois.append({'ROI': k, 'fiber': f'fiber_{v["location"]}', 'brain_region': v['location']})
        locations_df = pd.DataFrame(rois).set_index('ROI')
        locations_df_outpath = output_folder / 'photometryROI.locations.pqt'
        locations_df.to_parquet(locations_df_outpath)
        return ibl_df_outpath, locations_df_outpath


class FibrePhotometryBpodSync(FibrePhotometryBaseSync):
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [
                ('_neurophotometrics_fpData.raw.pqt', self.photometry_collection, True, True),
                ('_iblrig_taskData.raw.jsonable', self.task_collection, True, True),
                ('_neurophotometrics_fpData.channels.csv', self.photometry_collection, True, True),
                ('_neurophotometrics_fpData.digitalIntputs.pqt', self.photometry_collection, True),
            ],
            'output_files': [
                ('photometry.signal.pqt', 'alf/photometry', True),
                ('photometryROI.locations.pqt', 'alf/photometry', True),
            ],
        }
        return signature

    def _get_neurophotometrics_timestamps(self) -> np.ndarray:
        # for bpod based syncing, the timestamps for syncing are in the digital inputs file
        raw_photometry_folder = self.session_path / self.photometry_collection
        digital_inputs_df = pd.read_parquet(raw_photometry_folder / '_neurophotometrics_fpData.digitalIntputs.pqt')
        timestamps_nph = digital_inputs_df['SystemTimestamp'].values[digital_inputs_df['Channel'] == self.kwargs['sync_channel']]

        # TODO replace this rudimentary spacer removal
        # to implement: detect spacer / remove spacer methods
        # timestamps_nph = timestamps_nph[15:]
        return timestamps_nph


class FibrePhotometryDAQSync(FibrePhotometryBaseSync):
    priority = 90
    job_size = 'small'

    def __init__(self, *args, load_timestamps: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync_kwargs = kwargs.get('sync_metadata', self.session_params['sync'])
        self.sync_channel = kwargs.get('sync_channel', self.session_params['devices']['neurophotometrics']['sync_channel'])
        self.load_timestamps = load_timestamps

    @property
    def signature(self):
        signature = {
            'input_files': [
                ('_neurophotometrics_fpData.raw.pqt', self.photometry_collection, True, True),
                ('_iblrig_taskData.raw.jsonable', self.task_collection, True, True),
                ('_neurophotometrics_fpData.channels.csv', self.photometry_collection, True, True),
                ('_mcc_DAQdata.raw.tdms', self.photometry_collection, True, True),
            ],
            'output_files': [
                ('photometry.signal.pqt', 'alf/photometry', True),
                ('photometryROI.locations.pqt', 'alf/photometry', True),
            ],
        }
        return signature

    def load_data(self) -> pd.DataFrame:
        # the point of this functions is to overwrite the SystemTimestamp column
        # in the ibl_df with the values from the DAQ clock
        # then syncing will work the same as for the bpod based syncing
        raw_df = super().load_data()

        # get daqami timestamps
        # attempt to load
        timestamps_filepath = self.session_path / self.photometry_collection / '_mcc_DAQdata.pkl'
        if self.load_timestamps and timestamps_filepath.exists():
            with open(timestamps_filepath, 'rb') as fH:
                self.timestamps = pickle.load(fH)
        else:  # extract timestamps:
            tdms_filepath = self.session_path / self.photometry_collection / '_mcc_DAQdata.raw.tdms'
            self.timestamps = extract_timestamps_from_tdms_file_fast(tdms_filepath, save_path=timestamps_filepath)

        # downward compatibility - frameclock moved around, now is back on the AI7
        if self.sync_kwargs['frameclock_channel'] in ['0', 0]:
            sync_channel_name = f'DI{self.sync_kwargs["frameclock_channel"]}'
        elif self.sync_kwargs['frameclock_channel'] in ['7', 7]:
            sync_channel_name = f'AI{self.sync_kwargs["frameclock_channel"]}'
        else:
            sync_channel_name = self.sync_kwargs['frameclock_channel']
        frame_timestamps = self.timestamps[sync_channel_name]

        # compare number of frame timestamps
        # and put them in the raw_df SystemTimestamp column
        # based on the different scenarios

        # they are the same, all is well
        if raw_df.shape[0] == frame_timestamps.shape[0]:
            raw_df['SystemTimestamp'] = frame_timestamps
            _logger.debug(f'timestamps are of equal size {raw_df.shape[0]}')

        # there is one more timestamp recorded by the daq
        # (probably bonsai drops the last incomplete frame)
        elif raw_df.shape[0] + 1 == frame_timestamps.shape[0]:
            raw_df['SystemTimestamp'] = frame_timestamps[:-1]
            _logger.debug('one more timestamp in daq than frames by bonsai')

        # there is one more frame by bonsai that doesn't have
        # a timestamp (strange case)
        elif raw_df.shape[0] == frame_timestamps.shape[0] + 1:
            raw_df = raw_df.iloc[:-1]  # dropping the last frame
            raw_df['SystemTimestamp'] = frame_timestamps
            _logger.debug('one frame in bonsai than timestamps recorded by daq')

        # there are many more frames recorded by bonsai than
        # timestamps recorded by daqami
        elif raw_df.shape[0] > frame_timestamps.shape[0]:
            # the daqami was stopped / closed before bonsai
            # we discard all frames that can not be mapped
            _logger.warning(f'#frames bonsai: {raw_df.shape[0]} > #frames daqami {frame_timestamps.shape[0]}, dropping excess')
            raw_df = raw_df.iloc[: frame_timestamps.shape[0]]

        # there are more timestamps recorded by daqami than
        # frames recorded by bonsai
        elif raw_df.shape[0] + 1 < frame_timestamps.shape[0]:
            # this should not be possible / indicates a serious issue / bonsai crash')
            raise ValueError('more timestamps for frames recorded by the daqami than frames were recorded by bonsai.')
        return raw_df

    def _get_neurophotometrics_timestamps(self) -> np.ndarray:
        # get the sync channel and the corresponding timestamps
        timestamps_nph = self.timestamps[f'DI{self.sync_channel}']

        # TODO replace this rudimentary spacer removal
        # to implement: detect spacer / remove spacer methods
        # timestamps_nph = timestamps_nph[15: ]
        return timestamps_nph

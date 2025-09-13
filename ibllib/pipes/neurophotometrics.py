import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List
import pickle

import ibldsp.utils
import ibllib.io.session_params
from ibllib.pipes import base_tasks
from iblutil.io import jsonable

from nptdms import TdmsFile

from abc import abstractmethod
from iblphotometry import fpio
from iblutil.spacer import Spacer

_logger = logging.getLogger('ibllib')


def _int2digital_channels(values: np.ndarray) -> np.ndarray:
    """decoder for the digital channel values from the tdms file into a channel
    based array (rows are temporal samples, columns are channels).

    essentially does:

    0 -> 0000
    1 -> 1000
    2 -> 0100
    3 -> 1100
    4 -> 0010
    5 -> 1010
    6 -> 0110
    ...

    the order from binary representation is reversed so
    columns index represents channel index

    Parameters
    ----------
    values : np.ndarray
        the input values from the tdms digital channel

    Returns
    -------
    np.ndarray
        a (n x 4) array
    """
    return np.array([list(f'{v:04b}'[::-1]) for v in values], dtype='int8')


def extract_timestamps_from_tdms_file(
    tdms_filepath: Path,
    save_path: Optional[Path] = None,
    chunk_size=10000,
) -> dict:
    """extractor for tdms files as written by the daqami software, configured for neurophotometrics
    experiments: Frameclock is in AI7, DI1-4 are the bpod sync signals

    Parameters
    ----------
    tdms_filepath : Path
        path to TDMS file
    save_path : Optional[Path], optional
        if a path, save extracted timestamps from tdms file to this location, by default None
    chunk_size : int, optional
        if not None, process tdms data in chunks for decreased memory usage, by default 10000

    Returns
    -------
    dict
        a dict with the tdms channel names as keys and the timestamps of the rising fronts
    """
    #
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

    # chunked loop for memory efficiency
    if chunk_size is not None:
        n_chunks = df.shape[0] // chunk_size
        for i in range(n_chunks):
            vals_ = vals[i * chunk_size : (i + 1) * chunk_size]
            # data = np.array([list(f'{v:04b}'[::-1]) for v in vals_], dtype='int8')
            data = _int2digital_channels(vals_)

            for j, name in enumerate(digital_channel_names):
                ix = np.where(np.diff(data[:, j]) == 1)[0] + (chunk_size * i)
                timestamps[name].append(ix / fs)

        for ch in digital_channel_names:
            timestamps[ch] = np.concatenate(timestamps[ch])
    else:
        data = _int2digital_channels(vals)
        for j, name in enumerate(digital_channel_names):
            ix = np.where(np.diff(data[:, j]) == 1)[0]
            timestamps[name].append(ix / fs)

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


def extract_timestamps_from_bpod_jsonable(file_jsonable: str | Path, sync_states_names: List[str]):
    _, bpod_data = jsonable.load_task_jsonable(file_jsonable)
    timestamps = []
    for sync_name in sync_states_names:
        timestamps.append(
            np.array(
                [
                    data['States timestamps'][sync_name][0][0] + data['Trial start timestamp'] - data['Bpod start timestamp']
                    for data in bpod_data
                    if sync_name in data['States timestamps']
                ]
            )
        )
    timestamps = np.sort(np.concatenate(timestamps))
    timestamps = timestamps[~np.isnan(timestamps)]
    return timestamps


class FibrePhotometryBaseSync(base_tasks.DynamicTask):
    # base clas for syncing fibre photometry
    # derived classes are: FibrePhotometryBpodSync and FibrePhotometryDAQSync
    priority = 90
    job_size = 'small'

    def __init__(self, session_path, one, task_protocol=None, task_collection=None, **kwargs):
        super().__init__(session_path, one=one, **kwargs)
        self.photometry_collection = kwargs.get('collection', 'raw_photometry_data')  # raw_photometry_data
        self.kwargs = kwargs
        self.task_protocol = task_protocol
        self.task_collection = task_collection

        if self.task_protocol is None:
            # we will work with the first protocol here
            for task in self.session_params['tasks']:
                self.task_protocol = next(k for k in task)
                break

        if self.task_collection is None:
            # if not provided, infer
            self.task_collection = ibllib.io.session_params.get_task_collection(self.session_params, self.task_protocol)

    def _get_bpod_timestamps(self) -> Tuple[np.ndarray, list]:
        # the timestamps for syncing, in the time of the bpod
        if 'habituation' in self.task_protocol:
            sync_states_names = ['iti', 'reward']
        else:
            sync_states_names = ['trial_start', 'reward', 'exit_state']

        file_jsonable = self.session_path.joinpath(self.task_collection, '_iblrig_taskData.raw.jsonable')
        timestamps_bpod = extract_timestamps_from_bpod_jsonable(file_jsonable, sync_states_names)
        return timestamps_bpod

    def _get_valid_bounds(self):
        file_jsonable = self.session_path.joinpath(self.task_collection, '_iblrig_taskData.raw.jsonable')
        _, bpod_data = jsonable.load_task_jsonable(file_jsonable)
        return [bpod_data[0]['Trial start timestamp'] - 2, bpod_data[-1]['Trial end timestamp'] + 2]

    @abstractmethod
    def _get_neurophotometrics_timestamps(self) -> np.ndarray:
        # this function needs to be implemented in the derived classes:
        # for bpod based syncing, the timestamps are in the digial inputs file
        # for daq based syncing, the timestamps are extracted from the tdms file
        ...

    def _get_sync_function(self) -> Tuple[callable, list]:
        # returns the synchronization function
        # get the timestamps
        timestamps_bpod = self._get_bpod_timestamps()
        timestamps_nph = self._get_neurophotometrics_timestamps()

        # verify presence of sync timestamps
        for source, timestamps in zip(['bpod', 'neurophotometrics'], [timestamps_bpod, timestamps_nph]):
            assert len(timestamps) > 0, f'{source} sync timestamps are empty'

        sync_nph_to_bpod_fcn, drift_ppm, ix_nph, ix_bpod = ibldsp.utils.sync_timestamps(
            timestamps_nph, timestamps_bpod, return_indices=True, linear=True
        )
        _logger.info(f'synced with drift: {drift_ppm}')
        # TODO - assertion needed. 95% of timestamps in bpod need to be in timestamps of nph (but not the other way around)

        valid_bounds = self._get_valid_bounds()
        return sync_nph_to_bpod_fcn, valid_bounds

    def load_data(self) -> pd.DataFrame:
        # loads the raw photometry data
        raw_photometry_folder = self.session_path / self.photometry_collection
        photometry_df = fpio.from_neurophotometrics_file_to_photometry_df(
            raw_photometry_folder / '_neurophotometrics_fpData.raw.pqt',
            # data_columns=self.kwargs['fibers'],
            drop_first=False,
        )
        return photometry_df

    def _run(self, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # 1) load photometry data

        # note: when loading daq based syncing, the SystemTimestamp column
        # will be overridden with the timestamps from the tdms file
        # the idea behind this is that the rest of the sync is then the same
        # and handled by this base class
        photometry_df = self.load_data()

        # 2) get the synchronization function
        sync_nph_to_bpod_fcn, valid_bounds = self._get_sync_function()

        # 3) apply synchronization
        photometry_df['times'] = sync_nph_to_bpod_fcn(photometry_df['times'])
        photometry_df['valid'] = np.logical_and(
            photometry_df['times'] >= valid_bounds[0], photometry_df['times'] <= valid_bounds[1]
        )

        # 4) write to disk
        output_folder = self.session_path.joinpath('alf', 'photometry')
        output_folder.mkdir(parents=True, exist_ok=True)

        # writing the synced photometry signal
        photometry_df_outpath = output_folder / 'photometry.signal.pqt'
        photometry_df.to_parquet(photometry_df_outpath)

        # writing the locations
        rois = []
        for k, v in self.kwargs['fibers'].items():
            rois.append({'ROI': k, 'fiber': f'fiber_{v["location"]}', 'brain_region': v['location']})
        locations_df = pd.DataFrame(rois).set_index('ROI')
        locations_df_outpath = output_folder / 'photometryROI.locations.pqt'
        locations_df.to_parquet(locations_df_outpath)
        return photometry_df_outpath, locations_df_outpath


class FibrePhotometryBpodSync(FibrePhotometryBaseSync):
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [
                ('_neurophotometrics_fpData.raw.pqt', self.photometry_collection, True, True),
                ('_iblrig_taskData.raw.jsonable', self.task_collection, True, True),
                # ('_neurophotometrics_fpData.channels.csv', self.photometry_collection, True, True),
                ('_neurophotometrics_fpData.digitalInputs.pqt', self.photometry_collection, True),
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
        digital_inputs_filepath = raw_photometry_folder / '_neurophotometrics_fpData.digitalInputs.pqt'
        digital_inputs_df = fpio.read_digital_inputs_file(digital_inputs_filepath)
        # timestamps_nph = digital_inputs_df['times'].values[digital_inputs_df['channel'] == self.kwargs['sync_channel']]
        timestamps_nph = digital_inputs_df.groupby('channel').get_group(self.kwargs['sync_channel'])['times'].values

        # TODO replace this rudimentary spacer removal
        # to implement: detect spacer / remove spacer methods
        # timestamps_nph = timestamps_nph[15:]
        return timestamps_nph


class FibrePhotometryDAQSync(FibrePhotometryBaseSync):
    priority = 90
    job_size = 'small'

    def __init__(self, *args, load_timestamps: bool = True, **kwargs):
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
                # ('_neurophotometrics_fpData.channels.csv', self.photometry_collection, True, True),
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
        photometry_df = super().load_data()

        # get daqami timestamps
        # attempt to load
        timestamps_filepath = self.session_path / self.photometry_collection / '_mcc_DAQdata.pkl'
        if self.load_timestamps and timestamps_filepath.exists():
            with open(timestamps_filepath, 'rb') as fH:
                self.timestamps = pickle.load(fH)
        else:  # extract timestamps:
            tdms_filepath = self.session_path / self.photometry_collection / '_mcc_DAQdata.raw.tdms'
            self.timestamps = extract_timestamps_from_tdms_file(tdms_filepath, save_path=timestamps_filepath)

        # downward compatibility - frameclock moved around, now is back on the AI7
        if self.sync_kwargs['frameclock_channel'] in ['0', 0]:
            sync_channel_name = f'DI{self.sync_kwargs["frameclock_channel"]}'
        elif self.sync_kwargs['frameclock_channel'] in ['7', 7]:
            sync_channel_name = f'AI{self.sync_kwargs["frameclock_channel"]}'
        else:
            sync_channel_name = self.sync_kwargs['frameclock_channel']
        frame_timestamps = self.timestamps[sync_channel_name]

        # compare number of frame timestamps
        # and put them in the photometry_df SystemTimestamp column
        # based on the different scenarios
        frame_times_adjusted = False  # for debugging reasons

        # they are the same, all is well
        if photometry_df.shape[0] == frame_timestamps.shape[0]:
            photometry_df['times'] = frame_timestamps
            _logger.info(f'timestamps are of equal size {photometry_df.shape[0]}')
            frame_times_adjusted = True

        # there are more timestamps recorded by DAQ than
        # frames recorded by bonsai
        elif photometry_df.shape[0] < frame_timestamps.shape[0]:
            _logger.info(f'# bonsai frames: {photometry_df.shape[0]}, # daq timestamps: {frame_timestamps.shape[0]}')
            # there is exactly one more timestamp recorded by the daq
            # (probably bonsai drops the last incomplete frame)
            if photometry_df.shape[0] == frame_timestamps.shape[0] - 1:
                photometry_df['times'] = frame_timestamps[:-1]
            # there are two more frames recorded by the DAQ than by
            # bonsai - this is observed. TODO understand when this happens
            elif photometry_df.shape[0] == frame_timestamps.shape[0] - 2:
                photometry_df['times'] = frame_timestamps[:-2]
            # there are more frames recorded by the DAQ than that
            # this indicates and issue -
            elif photometry_df.shape[0] < frame_timestamps.shape[0] - 2:
                raise ValueError('more timestamps for frames recorded by the daqami than frames were recorded by bonsai.')
            frame_times_adjusted = True

        # there are more frames recorded by bonsai than by the DAQ
        # this happens when the user stops the daqami recording before stopping the bonsai
        # or when daqami crashes
        elif photometry_df.shape[0] > frame_timestamps.shape[0]:
            # we drop all excess frames
            _logger.warning(
                f'#frames bonsai: {photometry_df.shape[0]} > #frames daqami {frame_timestamps.shape[0]}, dropping excess'
            )
            n_frames_daqami = frame_timestamps.shape[0]
            photometry_df = photometry_df.iloc[:n_frames_daqami]
            photometry_df.loc[:, 'SystemTimestamp'] = frame_timestamps
            frame_times_adjusted = True

        if not frame_times_adjusted:
            raise ValueError('timestamp issue that hasnt been caught')

        return photometry_df

    def _get_neurophotometrics_timestamps(self) -> np.ndarray:
        # get the sync channel and the corresponding timestamps
        timestamps_nph = self.timestamps[f'DI{self.sync_channel}']

        # TODO replace this rudimentary spacer removal
        # to implement: detect spacer / remove spacer methods
        # timestamps_nph = timestamps_nph[15: ]
        return timestamps_nph

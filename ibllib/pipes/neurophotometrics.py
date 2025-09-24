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
from iblrig_tasks import _iblrig_tasks_passiveChoiceWorld

from one.api import ONE
import json
from scipy.optimize import minimize

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
    extract_durations: bool = False,
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
            vals_ = vals[i * chunk_size: (i + 1) * chunk_size]
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


def extract_ttl_durations_from_tdms_file(
    tdms_filepath: Path,
    save_path: Optional[Path] = None,
    chunk_size=10000,
) -> dict:
    _logger.info(f'extracting ttl_durations from tdms file: {tdms_filepath}')

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
        timestamps[ch] = dict(positive=[], negative=[])

    # chunked loop for memory efficiency
    if chunk_size is not None:
        n_chunks = df.shape[0] // chunk_size
        for i in range(n_chunks):
            vals_ = vals[i * chunk_size: (i + 1) * chunk_size]
            # data = np.array([list(f'{v:04b}'[::-1]) for v in vals_], dtype='int8')
            data = _int2digital_channels(vals_)

            for j, name in enumerate(digital_channel_names):
                ix = np.where(np.diff(data[:, j]) == 1)[0] + (chunk_size * i)
                timestamps[name]['positive'].append(ix / fs)
                ix = np.where(np.diff(data[:, j]) == -1)[0] + (chunk_size * i)
                timestamps[name]['negative'].append(ix / fs)

        for ch in digital_channel_names:
            timestamps[ch]['positive'] = np.concatenate(timestamps[ch]['positive'])
            timestamps[ch]['negative'] = np.concatenate(timestamps[ch]['negative'])
    else:
        data = _int2digital_channels(vals)
        for j, name in enumerate(digital_channel_names):
            ix = np.where(np.diff(data[:, j]) == 1)[0]
            timestamps[name]['positive'].append(ix / fs)
            ix = np.where(np.diff(data[:, j]) == -1)[0]
            timestamps[name]['negative'].append(ix / fs)

    if has_analog_group:
        # frameclock data is recorded on an analog channel
        for channel in analog_group.channels():
            timestamps[channel.name] = {}
            signal = (channel.data > 2.5).astype('int32')  # assumes 0-5V
            timestamps[channel.name]['positive'] = np.where(np.diff(signal) == 1)[0] / fs
            timestamps[channel.name]['negative'] = np.where(np.diff(signal) == -1)[0] / fs

    # the actual diff
    durations = {}
    for channel in timestamps.keys():
        durations[channel] = timestamps[channel]['negative'] - timestamps[channel]['positive']

    if save_path is not None:
        _logger.info(f'saving extracted ttl durations to: {save_path}')
        with open(save_path, 'wb') as fH:
            pickle.dump(durations, fH)

    return durations


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

    def __init__(
        self,
        session_path: str | Path,
        one: ONE,
        task_protocol: str | None = None,
        task_collection: str | None = None,
        **kwargs,
    ):
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

    def _get_bpod_timestamps(self) -> np.ndarray:
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
        if np.absolute(drift_ppm) > 20:
            _logger.warning(f'sync with excessive drift: {drift_ppm}')
        else:
            _logger.info(f'synced with drift: {drift_ppm}')

        # assertion: 95% of timestamps in bpod need to be in timestamps of nph (but not the other way around)
        assert timestamps_bpod.shape[0] * 0.95 < ix_bpod.shape[0], 'less than 95% of bpod timestamps matched'

        valid_bounds = self._get_valid_bounds()
        return sync_nph_to_bpod_fcn, valid_bounds

    def load_data(self) -> pd.DataFrame:
        # loads the raw photometry data
        raw_photometry_folder = self.session_path / self.photometry_collection
        photometry_df = fpio.from_neurophotometrics_file_to_photometry_df(
            raw_photometry_folder / '_neurophotometrics_fpData.raw.pqt',
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
        photometry_filepath = self.session_path / 'alf' / 'photometry' / 'photometry.signal.pqt'
        photometry_filepath.parent.mkdir(parents=True, exist_ok=True)
        photometry_df.to_parquet(photometry_filepath)

        # writing the locations
        rois = []
        for k, v in self.session_params['devices']['neurophotometrics']['fibers'].items():
            rois.append({'ROI': k, 'fiber': f'fiber_{v["location"]}', 'brain_region': v['location']})
        locations_df = pd.DataFrame(rois).set_index('ROI')
        locations_filepath = self.session_path / 'alf' / 'photometry' / 'photometryROI.locations.pqt'
        locations_filepath.parent.mkdir(parents=True, exist_ok=True)
        locations_df.to_parquet(locations_filepath)
        return photometry_filepath, locations_filepath


class FibrePhotometryBpodSync(FibrePhotometryBaseSync):
    priority = 90
    job_size = 'small'

    def __init__(
        self,
        *args,
        digital_inputs_channel: int | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.digital_inputs_channel = digital_inputs_channel

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
        digital_inputs_df = fpio.read_digital_inputs_file(
            digital_inputs_filepath, channel=self.session_params['devices']['neurophotometrics']['sync_channel']
        )
        sync_channel = self.session_params['devices']['neurophotometrics']['sync_channel']
        timestamps_nph = digital_inputs_df.groupby('channel').get_group(sync_channel)['times'].values

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


class FibrePhotometryPassiveChoiceWorld(base_tasks.BehaviourTask):
    priority = 90
    job_size = 'small'

    def __init__(
        self,
        session_path: str | Path,
        one: ONE,
        load_timestamps: bool = True,
        **kwargs,
    ):
        super().__init__(session_path, one=one, **kwargs)
        self.photometry_collection = kwargs.get('collection', 'raw_photometry_data')
        self.kwargs = kwargs
        self.load_timestamps = load_timestamps

    def _run(self, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # load the fixtures - from the relative delays between trials, an "absolute" time vector is
        # created that is used for the synchronization
        fixtures_path = Path(_iblrig_tasks_passiveChoiceWorld.__file__).parent / 'passiveChoiceWorld_trials_fixtures.pqt'

        # getting the task_settings
        with open(self.session_path / self.collection / '_iblrig_taskSettings.raw.json', 'r') as fH:
            task_settings = json.load(fH)

        # getting the fixtures and creating a relative time vector
        fixtures_df = pd.read_parquet(fixtures_path).groupby('session_id').get_group(task_settings['SESSION_TEMPLATE_ID'])

        # stimulus durations
        stim_durations = dict(
            T=task_settings['GO_TONE_DURATION'],
            N=task_settings['WHITE_NOISE_DURATION'],
            G=0.3,  # visual stimulus duration is hardcoded to 300ms
            V=0.1,  # V=0.1102 from a a session # to be replaced later down
        )
        for s in fixtures_df['stim_type'].unique():
            fixtures_df.loc[fixtures_df['stim_type'] == s, 'delay'] = stim_durations[s]

        # the audio go cue times
        mic_go_cue_times_bpod = np.load(self.session_path / self.collection / '_iblmic_audioOnsetGoCue.times_mic.npy')

        # adding the delays
        def obj_fun(x, mic_go_cue_times_bpod, fixtures_df):
            # fit overhead
            for s in ['T', 'N', 'G', 'V']:
                if s == 'T' or s == 'N':
                    fixtures_df.loc[fixtures_df['stim_type'] == s, 'overhead'] = x[0]
                if s == 'G':
                    fixtures_df.loc[fixtures_df['stim_type'] == s, 'overhead'] = x[1]
                if s == 'V':
                    fixtures_df.loc[fixtures_df['stim_type'] == s, 'overhead'] = x[2]

            fixtures_df['t_rel'] = np.cumsum(
                fixtures_df['stim_delay'].values + np.roll(fixtures_df['delay'].values, 1) + fixtures_df['overhead'].values,
            )

            mic_go_cue_times_rel = fixtures_df.groupby('stim_type').get_group('T')['t_rel'].values
            err = np.sum((np.diff(mic_go_cue_times_rel) - np.diff(mic_go_cue_times_bpod)) ** 2)
            return err

        # fitting the overheads
        fixtures_df['overhead'] = 0.0
        bounds = ((0, np.inf), (0, np.inf), (0, np.inf))
        pfit = minimize(obj_fun, (0.0, 0.0, 0.0), args=(mic_go_cue_times_bpod, fixtures_df), bounds=bounds)
        overheads = dict(zip(['T', 'N', 'G', 'V'], [pfit.x[0], pfit.x[0], pfit.x[1], pfit.x[2]]))

        for s in fixtures_df['stim_type'].unique():
            fixtures_df.loc[fixtures_df['stim_type'] == s, 'overhead'] = overheads[s]
        fixtures_df['t_rel'] = np.cumsum(
            fixtures_df['stim_delay'].values + np.roll(fixtures_df['delay'].values, 1) + fixtures_df['overhead'].values
        )

        mic_go_cue_times_rel = fixtures_df.groupby('stim_type').get_group('T')['t_rel'].values

        sync_fun, drift_ppm, ix_nph, ix_bpod = ibldsp.utils.sync_timestamps(
            mic_go_cue_times_rel, mic_go_cue_times_bpod, return_indices=True, linear=True
        )

        assert ix_nph.shape[0] == 40, 'not all microphone onset events are accepted by the sync function'
        if np.absolute(drift_ppm) > 20:
            _logger.warning(f'sync with excessive drift: {drift_ppm}')
        else:
            _logger.info(f'synced with drift: {drift_ppm}')

        # applying the sync to all the timestamps in the fixtures
        fixtures_df['t_bpod'] = sync_fun(fixtures_df['t_rel'])

        # dealing with the valve
        # valve_times_rel = fixtures_df.groupby('stim_type').get_group('V')['t_rel'].values
        # valve_times_bpod = sync_fun(valve_times_rel)
        valve_times_bpod = fixtures_df.groupby('stim_type').get_group('V')['t_bpod'].values

        # getting the valve timestamps from the DAQ
        timestamps_filepath = self.session_path / self.photometry_collection / '_mcc_DAQdata.pkl'
        if self.load_timestamps and timestamps_filepath.exists():
            with open(timestamps_filepath, 'rb') as fH:
                self.timestamps = pickle.load(fH)
        else:  # extract timestamps:
            tdms_filepath = self.session_path / self.photometry_collection / '_mcc_DAQdata.raw.tdms'
            self.timestamps = extract_timestamps_from_tdms_file(tdms_filepath, save_path=timestamps_filepath)

        sync_channel = self.session_params['devices']['neurophotometrics']['sync_channel']
        valve_times_nph = self.timestamps[f'DI{sync_channel}']

        sync_fun, drift_ppm, ix_nph, ix_bpod = ibldsp.utils.sync_timestamps(
            valve_times_nph, valve_times_bpod, return_indices=True, linear=True
        )
        assert ix_bpod.shape[0] == 40, 'not all bpod valve onset events are accepted by the sync function'
        if np.absolute(drift_ppm) > 20:
            _logger.warning(f'sync with excessive drift: {drift_ppm}')
        else:
            _logger.info(f'synced with drift: {drift_ppm}')

        # loads the raw photometry data
        raw_photometry_folder = self.session_path / self.photometry_collection
        photometry_df = fpio.from_neurophotometrics_file_to_photometry_df(
            raw_photometry_folder / '_neurophotometrics_fpData.raw.pqt',
            drop_first=False,
        )
        # apply synchronization
        photometry_df['times'] = sync_fun(photometry_df['times'])
        # verify that all are valid (i.e. mean nothing ... )

        # write to disk
        # the synced photometry signal
        photometry_filepath = self.session_path / 'alf' / 'photometry' / 'photometry.signal.pqt'
        photometry_filepath.parent.mkdir(parents=True, exist_ok=True)
        photometry_df.to_parquet(photometry_filepath)

        # writing the locations
        rois = []
        for k, v in self.session_params['devices']['neurophotometrics']['fibers'].items():
            rois.append({'ROI': k, 'fiber': f'fiber_{v["location"]}', 'brain_region': v['location']})
        locations_df = pd.DataFrame(rois).set_index('ROI')
        locations_filepath = self.session_path / 'alf' / 'photometry' / 'photometryROI.locations.pqt'
        locations_filepath.parent.mkdir(parents=True, exist_ok=True)
        locations_df.to_parquet(locations_filepath)

        # writing the passive events table
        # get the valve open duration
        ttl_durations_filepath = self.session_path / self.photometry_collection / '_mcc_DAQdurations.pkl'
        if self.load_timestamps and ttl_durations_filepath.exists():
            with open(ttl_durations_filepath, 'rb') as fH:
                ttl_durations = pickle.load(fH)
        else:  # extract timestamps:
            tdms_filepath = self.session_path / self.photometry_collection / '_mcc_DAQdata.raw.tdms'
            ttl_durations = extract_ttl_durations_from_tdms_file(tdms_filepath, save_path=ttl_durations_filepath)

        valve_open_dur = np.median(ttl_durations[f'DI{sync_channel}'][ix_nph])
        passiveStims_df = pd.DataFrame(
            dict(
                valveOn=fixtures_df.groupby('stim_type').get_group('V')['t_bpod'],
                valveOff=fixtures_df.groupby('stim_type').get_group('V')['t_bpod'] + valve_open_dur,
                toneOn=fixtures_df.groupby('stim_type').get_group('T')['t_bpod'],
                toneOff=fixtures_df.groupby('stim_type').get_group('T')['t_bpod'] + task_settings['GO_TONE_DURATION'],
                noiseOn=fixtures_df.groupby('stim_type').get_group('N')['t_bpod'],
                noiseOff=fixtures_df.groupby('stim_type').get_group('N')['t_bpod'] + task_settings['WHITE_NOISE_DURATION'],
            )
        )
        passiveStims_filepath = self.session_path / 'alf' / self.collection / '_ibl_passiveStims.table.pqt'
        passiveStims_filepath.parent.mkdir(exist_ok=True, parents=True)
        passiveStims_df.reset_index().to_parquet(passiveStims_filepath)

        return photometry_filepath, locations_filepath, passiveStims_filepath

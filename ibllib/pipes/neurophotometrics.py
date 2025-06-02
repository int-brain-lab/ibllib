import logging
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Tuple

import ibldsp.utils
import ibllib.io.session_params
from ibllib.pipes import base_tasks
from iblutil.io import jsonable

from nptdms import TdmsFile

from abc import abstractmethod
from iblphotometry import io as fpio

_logger = logging.getLogger('ibllib')


def extract_timestamps_from_tdms_file(tdms_filepath: Path) -> dict:
    # extractor for tdms files as written by the daqami software, configured
    # for neurophotometrics experiments: Frameclock is in AI7, DI1-4 are the
    # bpod sync signals

    tdms_file = TdmsFile.read(tdms_filepath)
    (digital_group,) = tdms_file.groups()
    fs = digital_group.properties['ScanRate']  # this should be 10kHz
    df = tdms_file.as_dataframe()
    col = df.columns[-1]
    vals = df[col].values.astype('int64')
    columns = ['DI0', 'DI1', 'DI2', 'DI3']

    # ugly but basically just a binary decoder for the binary data
    # assumes 4 channels
    data = np.array([list(bin(v)[2:].zfill(4)[::-1]) for v in vals], dtype='int64')
    timestamps = {}
    for i, name in enumerate(columns):
        signal = data[:, i]
        timestamps[name] = np.where(np.diff(signal) == 1)[0] / fs

    # frameclock data is recorded on an analog channel
    # for channel in analog_group.channels():
    #     signal = (channel.data > 2.5).astype('int64')  # assumes 0-5V
    #     timestamps[channel.name] = np.where(np.diff(signal) == 1)[0] / fs

    return timestamps


class FibrePhotometryBaseSync(base_tasks.DynamicTask):
    # base clas for syncing fibre photometry
    # derived classes are: FibrePhotometryBpodSync and FibrePhotometryDAQSync
    priority = 90
    job_size = 'small'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.photometry_collection = kwargs['collection']  # raw_photometry_data
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

    def _get_sync_function(self) -> Tuple[callable, list]:
        # returns the synchronization function

        # get the timestamps
        timestamps_bpod, bpod_data = self._get_bpod_timestamps()
        timestamps_nph = self._get_neurophotometrics_timestamps()

        # sync the behaviour events to the photometry timestamps
        sync_nph_to_bpod_fcn, drift_ppm, ix_nph, ix_bpod = ibldsp.utils.sync_timestamps(
            timestamps_nph, timestamps_bpod, return_indices=True, linear=True
        )
        # TODO log drift

        # then we check the alignment, should be less than the camera sampling rate
        tcheck = sync_nph_to_bpod_fcn(timestamps_nph[ix_nph]) - timestamps_bpod[ix_bpod]
        _logger.info(
            f'sync: n trials {len(bpod_data)}, n bpod sync {len(timestamps_bpod)}, n photometry {len(timestamps_nph)}, n match {len(ix_nph)}'
        )
        # TODO the framerate here is hardcoded, infer it instead!
        assert np.all(np.abs(tcheck) < 1 / 60), 'Sync issue detected, residual above 1/60s'
        assert len(ix_nph) / len(timestamps_bpod) > 0.95, 'Sync issue detected, less than 95% of the bpod events matched'
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
        sync_nph_to_bpod_fcn, valid_bounds = self._get_sync_function()

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
        return ibl_df, locations_df


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
        timestamps_nph = timestamps_nph[15:]
        return timestamps_nph


class FibrePhotometryDAQSync(FibrePhotometryBaseSync):
    priority = 90
    job_size = 'small'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync_kwargs = kwargs['sync_metadata']
        self.sync_channel = kwargs['sync_channel']

    @property
    def signature(self):
        signature = {
            'input_files': [
                ('_neurophotometrics_fpData.raw.pqt', self.photometry_collection, True, True),
                ('_iblrig_taskData.raw.jsonable', self.task_collection, True, True),
                ('_neurophotometrics_fpData.channels.csv', self.photometry_collection, True, True),
                ('_mcc_DAQdata.raw.tdms', self.sync_kwargs['collection'], True, True),
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
        tdms_filepath = self.session_path / self.sync_kwargs['collection'] / '_mcc_DAQdata.raw.tdms'
        self.timestamps = extract_timestamps_from_tdms_file(tdms_filepath)
        frame_timestamps = self.timestamps[f'DI{self.sync_kwargs["frameclock_channel"]}']

        # compare number of frame timestamps
        # and put them in the raw_df SystemTimestamp column
        if raw_df.shape[0] == frame_timestamps.shape[0]:
            raw_df['SystemTimestamp'] = frame_timestamps
        elif raw_df.shape[0] == frame_timestamps.shape[0] + 1:
            # there is one extra frame timestamp from the last incomplete frame
            raw_df['SystemTimestamp'] = frame_timestamps[:-1]
        elif raw_df.shape[0] > frame_timestamps:
            # the daqami was stopped / closed before bonsai
            # we discard all frames that can not be mapped
            _logger.warning(
                f'#frames recorded by bonsai: {raw_df.shape[0]} > #frame timestamps recorded by daqami {frame_timestamps.shape[0]}, dropping all frames without recorded timestamps'
            )
            raw_df = raw_df.iloc[: frame_timestamps.shape[0]]

        elif raw_df.shape[0] < frame_timestamps:
            # this should not be possible
            raise ValueError('more timestamps for frames recorded by the daqami than frames were recorded by bonsai.')
        return raw_df

    def _get_neurophotometrics_timestamps(self) -> np.ndarray:
        # get the sync channel
        sync_colname = f'DI{self.kwargs["sync_channel"]}'

        # and the corresponding timestamps
        timestamps_nph = self.timestamps[sync_colname]

        # TODO replace this rudimentary spacer removal
        # to implement: detect spacer / remove spacer methods
        timestamps_nph = timestamps_nph[15:]
        return timestamps_nph

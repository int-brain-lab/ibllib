import logging
import numpy as np
import pandas as pd

import ibldsp.utils
import ibllib.io.session_params
from ibllib.pipes import base_tasks
from iblutil.io import jsonable
import iblphotometry.io as fpio

from ibldsp.utils import rises
from nptdms import TdmsFile

from abc import abstractmethod
import iblphotometry

_logger = logging.getLogger('ibllib')


class FibrePhotometryBaseSync(base_tasks.DynamicTask):
    # base clas for syncing fibre photometry
    # derived classes are: FibrePhotometryBpodSync and FibrePhotometryDAQSync
    priority = 90
    job_size = 'small'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_collection = self.get_device_collection('neurophotometrics', device_collection='raw_photometry_data')
        self.kwargs = kwargs

        # we will work with the first protocol here
        for task in self.session_params['tasks']:
            self.task_protocol = next(k for k in task)
            self.task_collection = ibllib.io.session_params.get_task_collection(self.session_params, self.task_protocol)
            break

    def _get_bpod_timestamps(self):
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
    def _get_neurophotometrics_timestamps(self):
        # this function needs to be implemented in the derived classes:
        # for bpod based syncing, the timestamps are in the digial inputs file
        # for daq based syncing, the timestamps are extracted from the tdms file
        ...

    def _get_sync_function(self):
        """
        Perform the linear clock correction between bpod and neurophotometrics timestamps.
        :return: interpolation function that outputs bpod timestamsp from neurophotometrics timestamps
        """

        # get the timestamps
        timestamps_bpod, bpod_data = self._get_bpod_timestamps(self.task_protocol)
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
        # FIXME the framerate here is hardcoded, infer it instead!
        assert np.all(np.abs(tcheck) < 1 / 60), 'Sync issue detected, residual above 1/60s'
        assert len(ix_nph) / len(timestamps_bpod) > 0.95, 'Sync issue detected, less than 95% of the bpod events matched'
        valid_bounds = [bpod_data[0]['Trial start timestamp'] - 2, bpod_data[-1]['Trial end timestamp'] + 2]

        return sync_nph_to_bpod_fcn, valid_bounds

    def load_data(self):
        raw_photometry_folder = self.session_path / self.device_collection
        raw_neurophotometrics_df = pd.read_parquet(raw_photometry_folder / '_neurophotometrics_fpData.raw.pqt')
        ibl_df = iblphotometry.io.from_raw_neurophotometrics_df_to_ibl_df(
            raw_neurophotometrics_df,
            rois=self.kwargs['fibers'],
        )
        return ibl_df

    def _run(self, **kwargs):
        """ """
        # 1) load photometry data
        # note: when loading daq based syncing, the SystemTimestamp column
        ibl_df = self.load_data()

        # 2) get the synchronization function
        sync_nph_to_bpod_fcn, valid_bounds = self._get_sync_function()
        ibl_df['valid'] = np.logical_and(ibl_df['times'] >= valid_bounds[0], ibl_df['times'] <= valid_bounds[1])

        # 3) apply synchronization
        # for bpod based syncing, we can directly transform the timestamps that are
        # stored with the samples
        ibl_df['times'] = sync_nph_to_bpod_fcn(ibl_df['SystemTimestamp'])

        # 4) write to disk
        output_folder = self.session_path.joinpath('alf', 'photometry')
        output_folder.mkdir(parents=True, exist_ok=True)

        # writing the synced photometry signal
        ibl_df_outpath = output_folder / 'photometry.signal.pqt'
        ibl_df.to_parquet(ibl_df_outpath)

        # writing the locations
        rois = list(self.kwargs['fibers'].keys())
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
                ('_neurophotometrics_fpData.raw.pqt', self.device_collection, True, True),
                ('_iblrig_taskData.raw.jsonable', self.task_collection, True, True),
                ('_neurophotometrics_fpData.channels.csv', self.device_collection, True, True),
                ('_neurophotometrics_fpData.digitalIntputs.pqt', self.device_collection, True),
            ],
            'output_files': [
                ('photometry.signal.pqt', 'alf/photometry', True),
                ('photometryROI.locations.pqt', 'alf/photometry', True),
            ],
        }
        return signature

    def _get_neurophotometrics_timestamps(self):
        # we get the timestamps for the photometry data by loading from the digital inputs file
        raw_photometry_folder = self.session_path / self.device_collection
        digital_inputs_df = pd.read_parquet(raw_photometry_folder / '_neurophotometrics_fpData.digitalIntputs.pqt')
        timestamps_nph = digital_inputs_df['SystemTimestamp'].values[digital_inputs_df['Channel'] == self.kwargs['sync_channel']]

        # simple spacer removal, TODO replace this with something more robust
        # detect spacer / remove spacer methods
        timestamps_nph = timestamps_nph[15:]
        return timestamps_nph


class FibrePhotometryDAQSync(FibrePhotometryBaseSync):
    """ """

    priority = 90
    job_size = 'small'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sync_kwargs = kwargs['daqami']
        self.sync_channel = kwargs['sync_channel']

    @property
    def signature(self):
        signature = {
            'input_files': [
                ('_neurophotometrics_fpData.raw.pqt', self.device_collection, True, True),
                ('_iblrig_taskData.raw.jsonable', self.task_collection, True, True),
                ('_neurophotometrics_fpData.channels.csv', self.device_collection, True, True),
                ('_mcc_DAQdata.raw.tdms', self.sync_kwargs['collection'], True, True),
            ],
            'output_files': [
                ('photometry.signal.pqt', 'alf/photometry', True),
                ('photometryROI.locations.pqt', 'alf/photometry', True),
            ],
        }
        return signature

    def _load_and_parse_tdms(self):
        # loads the tdms file data, and detects the risind edges
        # this probably could use some dsp, potentially trend removal
        tdms_filepath = self.session_path / self.sync_kwargs['collection'] / '_mcc_DAQdata.raw.tdms'
        tdms_df = TdmsFile.read(tdms_filepath).as_dataframe()
        tdms_df.columns = [col[-4:-1] for col in tdms_df.columns]  # hardcoded renaming

        timestamps = {}
        for col in tdms_df.columns:
            timestamps[col] = rises(tdms_df[col]) / self.sync_kwargs['sampling_rate']

        return timestamps

    def load_data(self):
        # the point of this functions is to overwrite the SystemTimestamp column
        # in the ibl_df with the values from the DAQ clock
        # then syncing will work the same as for the bpod based syncing

        ibl_df = super().load_data()

        self.timestamps = self._load_and_parse_tdms()
        frame_timestamps = self.timestamps[f'AI{self.sync_kwargs["frameclock_channel"]}']

        # and put them in the ibl_df SystemTimestamp column
        ibl_df['SystemTimestamp'] = frame_timestamps
        return ibl_df

    def _get_neurophotometrics_timestamps(self):
        # get the sync channel
        sync_colname = f'AI{self.sync_kwargs[""]}'

        # and the corresponding timestamps
        timestamps_nph = self.timestamps[sync_colname]

        # simple spacer removal, TODO replace this with something more robust
        # detect spacer / remove spacer methods
        timestamps_nph = timestamps_nph[15:]
        return timestamps_nph

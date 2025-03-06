"""Extraction tasks for fibrephotometry"""

import logging
import numpy as np
import pandas as pd

import ibldsp.utils
import ibllib.io.session_params
from ibllib.pipes import base_tasks
from iblutil.io import jsonable
import iblphotometry.io as fpio

_logger = logging.getLogger('ibllib')


class FibrePhotometrySync(base_tasks.DynamicTask):
    priority = 90
    job_size = 'small'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_collection = self.get_device_collection('neurophotometrics', device_collection='raw_photometry_data')
        # we will work with the first protocol here
        for task in self.session_params['tasks']:
            self.task_protocol = next(k for k in task)
            self.task_collection = ibllib.io.session_params.get_task_collection(self.session_params, self.task_protocol)
            break

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

    def _sync_bpod_neurophotometrics(self):
        """
        Perform the linear clock correction between bpod and neurophotometrics timestamps.
        :return: interpolation function that outputs bpod timestamsp from neurophotometrics timestamps
        """
        folder_raw_photometry = self.session_path.joinpath(self.device_collection)
        df_digital_inputs = fpio.read_digital_inputs_file(folder_raw_photometry / '_neurophotometrics_fpData.digitalIntputs.pqt')
        # normally we should disregard the states and use the sync label. But bpod doesn't log TTL outs,
        # only the states. This will change in the future but for now we are stuck with this.
        if 'habituation' in self.task_protocol:
            sync_states_names = ['iti', 'reward']
        else:
            sync_states_names = ['trial_start', 'reward', 'exit_state']
        # read in the raw behaviour data for syncing
        file_jsonable = self.session_path.joinpath(self.task_collection, '_iblrig_taskData.raw.jsonable')
        trials_table, bpod_data = jsonable.load_task_jsonable(file_jsonable)
        # we get the timestamps of the states from the bpod data
        tbpod = []
        for sname in sync_states_names:
            tbpod.append(
                np.array(
                    [
                        bd['States timestamps'][sname][0][0] + bd['Trial start timestamp'] - bpod_data[0]['Bpod start timestamp']
                        for bd in bpod_data
                        if sname in bd['States timestamps']
                    ]
                )
            )
        tbpod = np.sort(np.concatenate(tbpod))
        tbpod = tbpod[~np.isnan(tbpod)]
        # we get the timestamps for the photometry data
        sync_channel = self.session_params['devices']['neurophotometrics']['sync_channel']
        tph = df_digital_inputs['SystemTimestamp'].values[df_digital_inputs['Channel'] == sync_channel]
        tph = tph[15:]  # TODO: we may want to detect the spacers before removing it, especially for successive sessions
        # sync the behaviour events to the photometry timestamps
        fcn_nph_to_bpod_times, drift_ppm, iph, ibpod = ibldsp.utils.sync_timestamps(tph, tbpod, return_indices=True, linear=True)
        # then we check the alignment, should be less than the screen refresh rate
        tcheck = fcn_nph_to_bpod_times(tph[iph]) - tbpod[ibpod]
        _logger.info(f'sync: n trials {len(bpod_data)}, n bpod sync {len(tbpod)}, n photometry {len(tph)}, n match {len(iph)}')
        assert np.all(np.abs(tcheck) < 1 / 60), 'Sync issue detected, residual above 1/60s'
        assert len(iph) / len(tbpod) > 0.95, 'Sync issue detected, less than 95% of the bpod events matched'
        valid_bounds = [bpod_data[0]['Trial start timestamp'] - 2, bpod_data[-1]['Trial end timestamp'] + 2]
        return fcn_nph_to_bpod_times, valid_bounds

    def _run(self, **kwargs):
        """
        Extract photometry data from the raw neurophotometrics data in parquet
        The extraction has 3 main steps:
        1. Synchronise the bpod and neurophotometrics timestamps.
        2. Extract the photometry data from the raw neurophotometrics data.
        3. Label the fibers correspondance with brain regions in a small table
        :param kwargs:
        :return:
        """
        # 1) sync: we check the synchronisation, right now we only have bpod but soon the daq will be used
        match list(self.session_params['sync'].keys())[0]:
            case 'bpod':
                fcn_nph_to_bpod_times, valid_bounds = self._sync_bpod_neurophotometrics()
            case _:
                raise NotImplementedError('Syncing with daq is not supported yet.')

        # 2) reformat the raw data with wavelengths and meta-data
        folder_raw_photometry = self.session_path.joinpath(self.device_collection)
        out_df = fpio.from_raw_neurophotometrics_file_to_ibl_df(
            folder_raw_photometry.joinpath('_neurophotometrics_fpData.raw.pqt')
        )
        out_df['times'] = fcn_nph_to_bpod_times(out_df['times'])

        # 3) label the brain regions
        rois = []
        for k, v in self.session_params['devices']['neurophotometrics']['fibers'].items():
            rois.append({'ROI': k, 'fiber': f'fiber_{v["location"]}', 'brain_region': v['location']})
        df_rois = pd.DataFrame(rois).set_index('ROI')

        # 4) to finish we write the dataframes to disk
        out_path = self.session_path.joinpath('alf', 'photometry')
        out_path.mkdir(parents=True, exist_ok=True)
        out_df.to_parquet(file_signal := out_path.joinpath('photometry.signal.pqt'))
        df_rois.to_parquet(file_locations := out_path.joinpath('photometryROI.locations.pqt'))
        return file_signal, file_locations

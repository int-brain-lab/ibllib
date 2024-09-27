"""Extraction tasks for fibrephotometry"""

import logging

from ibllib.pipes import base_tasks
from ibllib.io.extractors.fibrephotometry import FibrePhotometry

_logger = logging.getLogger('ibllib')


class FibrePhotometryRegisterRaw(base_tasks.RegisterRawDataTask):

    priority = 100
    job_size = 'small'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collection = self.get_task_collection(kwargs.get('collection', None))
        self.device_collection = self.get_device_collection('photometry', device_collection='raw_photometry_data')

    @property
    def signature(self):
        signature = {
            'input_files': [],
            'output_files': [('_mcc_DAQdata.raw.tdms', self.device_collection, True),
                             ('_neurophotometrics_fpData.raw.pqt', self.device_collection, True)]
        }
        return signature


class FibrePhotometryPreprocess(base_tasks.DynamicTask):
    @property
    def signature(self):
        signature = {
            'input_files': [('_mcc_DAQdata.raw.tdms', self.device_collection, True),
                            ('_neurophotometrics_fpData.raw.pqt', self.device_collection, True)],
            'output_files': [('photometry.signal.pqt', 'alf/photometry', True)]
        }
        return signature

    priority = 90
    level = 1

    def __init__(self, session_path, regions=None, **kwargs):
        super().__init__(session_path, **kwargs)
        # Task collection (this needs to be specified in the task kwargs)
        self.collection = self.get_task_collection(kwargs.get('collection', None))
        self.device_collection = self.get_device_collection('photometry', device_collection='raw_photometry_data')
        self.regions = regions

    def _run(self, **kwargs):
        _, out_files = FibrePhotometry(self.session_path, collection=self.device_collection).extract(
            regions=self.regions, path_out=self.session_path.joinpath('alf', 'photometry'), save=True)
        return out_files

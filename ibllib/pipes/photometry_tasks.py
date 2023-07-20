"""Extraction tasks for fibrephotometry"""

import logging
from collections import OrderedDict

from ibllib.pipes import tasks, base_tasks
import ibllib.pipes.training_preprocessing as tpp
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


# pipeline
class FibrePhotometryExtractionPipeline(tasks.Pipeline):
    """
    This is a legacy pipeline not using the acquisition description file to acquire previous sessions at Princeton
    """
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        # FIXME This should be agnostic to task protocol, for now let's assume it's only training
        super().__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks['TrainingRegisterRaw'] = tpp.TrainingRegisterRaw(self.session_path)
        tasks['TrainingTrials'] = tpp.TrainingTrials(self.session_path)
        tasks['TrainingVideoCompress'] = tpp.TrainingVideoCompress(self.session_path)
        tasks['TrainingAudio'] = tpp.TrainingAudio(self.session_path)
        # level 1
        tasks['BiasedFibrePhotometry'] = FibrePhotometryPreprocess(self.session_path, parents=[tasks['TrainingTrials']])
        tasks['TrainingStatus'] = tpp.TrainingStatus(self.session_path, parents=[tasks['TrainingTrials']])
        tasks['TrainingDLC'] = tpp.TrainingDLC(
            self.session_path, parents=[tasks['TrainingVideoCompress']])
        self.tasks = tasks

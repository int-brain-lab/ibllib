"""Extraction tasks for fibrephotometry"""

import logging
from collections import OrderedDict

from ibllib.pipes import tasks, base_tasks
from ibllib.pipes.training_preprocessing import (
    TrainingRegisterRaw, TrainingAudio, TrainingTrials, TrainingDLC, TrainingStatus, TrainingVideoCompress)
from ibllib.io.extractors.fibrephotometry import FibrePhotometry

_logger = logging.getLogger('ibllib')


class TaskFibrePhotometryRegisterRaw(base_tasks.RegisterRawDataTask):

    priority = 100
    job_size = 'small'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.collection = self.get_task_collection(kwargs.get('collection', None))

    @property
    def signature(self):
        signature = {
            'input_files': [],
            'output_files': [('_mcc_DAQdata.raw.tdms', self.collection, True),
                             ('_neurophotometrics_fpData.raw.pqt', self.collection, True)]
        }
        return signature


class TaskFibrePhotometryPreprocess(base_tasks.DynamicTask):
    signature = {
        'input_files': [('*fpData.raw*', 'raw_photometry_data', True), ],
        'output_files': [('photometry.signal.pqt', 'alf', True), ]
    }
    priority = 90
    level = 1

    def __init__(self, session_path, regions=None, **kwargs):
        super().__init__(session_path, **kwargs)
        # Task collection (this needs to be specified in the task kwargs)
        self.collection = self.get_task_collection(kwargs.get('collection', None))
        self.regions = regions

    def _run(self, **kwargs):
        _, out_files = FibrePhotometry(self.session_path, collection=self.collection).extract(
            regions=self.regions, save=True)
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
        tasks['TrainingRegisterRaw'] = TrainingRegisterRaw(self.session_path)
        tasks['TrainingTrials'] = TrainingTrials(self.session_path)
        tasks['TrainingVideoCompress'] = TrainingVideoCompress(self.session_path)
        tasks['TrainingAudio'] = TrainingAudio(self.session_path)
        # level 1
        tasks['BiasedFibrePhotometry'] = TaskFibrePhotometryPreprocess(self.session_path, parents=[tasks['TrainingTrials']])
        tasks['TrainingStatus'] = TrainingStatus(self.session_path, parents=[tasks['TrainingTrials']])
        tasks['TrainingDLC'] = TrainingDLC(
            self.session_path, parents=[tasks['TrainingVideoCompress']])
        self.tasks = tasks

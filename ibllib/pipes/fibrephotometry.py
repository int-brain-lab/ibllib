"""Extraction tasks for fibrephotometry"""

import logging
from collections import OrderedDict

from ibllib.pipes import tasks
from ibllib.pipes.training_preprocessing import (
    TrainingRegisterRaw, TrainingAudio, TrainingTrials, TrainingDLC, TrainingStatus, TrainingVideoCompress)
from ibllib.io.extractors.fibrephotometry import FibrePhotometry

_logger = logging.getLogger('ibllib')


class FibrePhotometryPreprocess(tasks.Task):
    signature = {
        'input_files': [('*fpData.raw*', 'raw_photometry_data', True), ],
        'output_files': [('photometry.signal.npy', 'alf', True),
                         ('photometry.photometryLightSource.npy', 'alf', True),
                         ('photometryLightSource.properties.tsv', 'alf', True),
                         ('photometry.times.npy', 'alf', True), ]
    }
    priority = 90
    level = 1

    def _run(self, **kwargs):
        _, out_files = FibrePhotometry(self.session_path).extract(save=True)
        return out_files


# pipeline
class FibrePhotometryExtractionPipeline(tasks.Pipeline):
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
        tasks['BiasedFibrePhotometry'] = FibrePhotometryPreprocess(self.session_path, parents=[tasks['TrainingTrials']])
        tasks['TrainingStatus'] = TrainingStatus(self.session_path, parents=[tasks['TrainingTrials']])
        tasks['TrainingDLC'] = TrainingDLC(
            self.session_path, parents=[tasks['TrainingVideoCompress']])
        self.tasks = tasks

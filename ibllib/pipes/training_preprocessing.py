"""(Deprecated) Training data preprocessing tasks.

These tasks are part of the old pipeline. This module has been replaced by the dynamic pipeline
and the `behavior_tasks` module.
"""

import logging
from collections import OrderedDict
from one.alf.files import session_path_parts
import warnings

from ibllib.pipes.base_tasks import ExperimentDescriptionRegisterRaw
from ibllib.pipes import tasks, training_status
from ibllib.io import ffmpeg
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io.extractors import training_audio, bpod_trials, camera
from ibllib.qc.camera import CameraQC
from ibllib.qc.task_metrics import TaskQC, HabituationQC
from ibllib.qc.task_extractors import TaskQCExtractor

_logger = logging.getLogger(__name__)
warnings.warn('`pipes.training_preprocessing` to be removed in favour of dynamic pipeline', FutureWarning)


#  level 0
class TrainingRegisterRaw(tasks.Task):
    priority = 100

    def _run(self):
        return []


class TrainingTrials(tasks.Task):
    priority = 90
    level = 0
    force = False
    signature = {
        'input_files': [('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
                        ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True),
                        ('_iblrig_encoderEvents.raw*', 'raw_behavior_data', True),
                        ('_iblrig_encoderPositions.raw*', 'raw_behavior_data', True)],
        'output_files': [('*trials.goCueTrigger_times.npy', 'alf', True),
                         ('*trials.table.pqt', 'alf', True),
                         ('*wheel.position.npy', 'alf', True),
                         ('*wheel.timestamps.npy', 'alf', True),
                         ('*wheelMoves.intervals.npy', 'alf', True),
                         ('*wheelMoves.peakAmplitude.npy', 'alf', True)]
    }

    def extract_behaviour(self, save=True):
        """Extracts an iblrig training session."""
        trials, wheel, output_files = bpod_trials.extract_all(self.session_path, save=save)
        if trials is None:
            return None, None
        if wheel is not None:
            trials.update(wheel)
        return trials, output_files

    def run_qc(self, trials_data=None, update=True):
        if trials_data is None:
            trials_data, _ = self.extract_behaviour(save=False)
        if not trials_data:
            raise ValueError('No trials data found')

        # Compile task data for QC
        extractor_type = get_session_extractor_type(self.session_path)
        if extractor_type == 'habituation':
            qc = HabituationQC(self.session_path, one=self.one)
        else:
            qc = TaskQC(self.session_path, one=self.one)
        qc.extractor = TaskQCExtractor(self.session_path, one=self.one, lazy=True)
        qc.extractor.type = extractor_type
        qc.extractor.data = qc.extractor.rename_data(trials_data)
        qc.extractor.load_raw_data()  # re-loads raw data and populates various properties
        # Aggregate and update Alyx QC fields
        qc.run(update=update)

        return qc

    def _run(self, **_):
        """Extracts an iblrig training session and runs QC."""
        trials_data, output_files = self.extract_behaviour()
        if self.one and not self.one.offline and trials_data:
            # Run the task QC
            self.run_qc(trials_data)
        return output_files


class TrainingVideoCompress(tasks.Task):

    priority = 90
    io_charge = 100
    job_size = 'large'

    def _run(self):
        # avi to mp4 compression
        command = ('ffmpeg -i {file_in} -y -nostdin -codec:v libx264 -preset slow -crf 29 '
                   '-nostats -codec:a copy {file_out}')
        output_files = ffmpeg.iblrig_video_compression(self.session_path, command)

        if len(output_files) == 0:
            _logger.info('No compressed videos found; skipping timestamp extraction')
            return  # labels the task as empty if no output

        # Video timestamps extraction
        data, files = camera.extract_all(self.session_path, save=True, video_path=output_files[0])
        output_files.extend(files)

        # Video QC
        CameraQC(self.session_path, 'left', one=self.one, stream=False).run(update=True)
        return output_files


class TrainingAudio(tasks.Task):
    """
    Computes raw electrophysiology QC
    """
    cpu = 2
    priority = 10  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything

    def _run(self, overwrite=False):
        return training_audio.extract_sound(self.session_path, save=True, delete=True)


# level 1
class TrainingDLC(tasks.Task):

    def _run(self):
        """empty placeholder for job creation only"""
        pass


class TrainingStatus(tasks.Task):
    priority = 90
    level = 1
    force = False
    signature = {
        'input_files': [('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
                        ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True),
                        ('*trials.table.pqt', 'alf', True)],
        'output_files': []
    }

    def _run(self, upload=True):
        """
        Extracts training status for subject
        """
        df = training_status.get_latest_training_information(self.session_path, self.one)
        if df is not None:
            training_status.make_plots(self.session_path, self.one, df=df, save=True, upload=upload)
            # Update status map in JSON field of subjects endpoint
            # TODO This requires exposing the json field of the subjects endpoint
            if self.one and not self.one.offline:
                _logger.debug('Updating JSON field of subjects endpoint')
                try:
                    status = (df.set_index('date')[['training_status', 'session_path']].drop_duplicates(
                        subset='training_status', keep='first').to_dict())
                    date, sess = status.items()
                    data = {'trained_criteria': {v.replace(' ', '_'): (k, self.one.path2eid(sess[1][k])) for k, v
                                                 in date[1].items()}}
                    _, subject, *_ = session_path_parts(self.session_path)
                    self.one.alyx.json_field_update('subjects', subject, data=data)
                except KeyError:
                    _logger.error('Failed to update subject training status on Alyx: json field not available')

        output_files = []
        return output_files


class TrainingExtractionPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path, **kwargs):
        super(TrainingExtractionPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks['ExperimentDescriptionRegisterRaw'] = ExperimentDescriptionRegisterRaw(self.session_path)
        tasks['TrainingRegisterRaw'] = TrainingRegisterRaw(self.session_path)
        tasks['TrainingTrials'] = TrainingTrials(self.session_path)
        tasks['TrainingVideoCompress'] = TrainingVideoCompress(self.session_path)
        tasks['TrainingAudio'] = TrainingAudio(self.session_path)
        # level 1
        tasks['TrainingStatus'] = TrainingStatus(self.session_path, parents=[tasks['TrainingTrials']])
        tasks['TrainingDLC'] = TrainingDLC(
            self.session_path, parents=[tasks['TrainingVideoCompress']])
        self.tasks = tasks

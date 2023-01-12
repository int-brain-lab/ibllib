import logging
from collections import OrderedDict

from ibllib.pipes.base_tasks import ExperimentDescriptionRegisterRaw
from ibllib.pipes import tasks, training_status
from ibllib.io import ffmpeg
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io.extractors import training_audio, bpod_trials, camera
from ibllib.qc.camera import CameraQC
from ibllib.qc.task_metrics import TaskQC, HabituationQC
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.oneibl.registration import register_session_raw_data

_logger = logging.getLogger(__name__)


#  level 0
class TrainingRegisterRaw(tasks.Task):
    priority = 100

    def _run(self, overwrite=False):
        out_files, _ = register_session_raw_data(self.session_path, one=self.one, dry=True)
        return out_files


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

    def _run(self):
        """
        Extracts an iblrig training session
        """
        trials, wheel, output_files = bpod_trials.extract_all(self.session_path, save=True)
        if trials is None:
            return None
        if self.one is None or self.one.offline:
            return output_files
        # Run the task QC
        # Compile task data for QC
        type = get_session_extractor_type(self.session_path)
        if type == 'habituation':
            qc = HabituationQC(self.session_path, one=self.one)
            qc.extractor = TaskQCExtractor(self.session_path, one=self.one)
        else:  # Update wheel data
            qc = TaskQC(self.session_path, one=self.one)
            qc.extractor = TaskQCExtractor(self.session_path, one=self.one)
            qc.extractor.wheel_encoding = 'X1'
        # Aggregate and update Alyx QC fields
        qc.run(update=True)
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
        training_status.make_plots(self.session_path, self.one, df=df, save=True, upload=upload)
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

import logging
from collections import OrderedDict

from ibllib.pipes import tasks
from ibllib.io import ffmpeg
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io.extractors import training_audio, bpod_trials
from ibllib.qc.task_metrics import TaskQC, HabituationQC
from ibllib.qc.task_extractors import TaskQCExtractor
from oneibl.registration import register_session_raw_data

_logger = logging.getLogger('ibllib')


#  level 0
class TrainingRegisterRaw(tasks.Task):
    priority = 100

    def _run(self, overwrite=False):
        out_files, _ = register_session_raw_data(self.session_path, one=self.one, dry=True)
        return out_files


class TrainingTrials(tasks.Task):
    priority = 90
    level = 0

    def _run(self):
        """
        Extracts an iblrig training session
        """
        trials, wheel, output_files = bpod_trials.extract_all(self.session_path, save=True)
        if trials is None:
            return None

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

    def _run(self):
        # avi to mp4 compression
        command = ('ffmpeg -i {file_in} -y -nostdin -codec:v libx264 -preset slow -crf 29 '
                   '-nostats -codec:a copy {file_out}')
        output_files = ffmpeg.iblrig_video_compression(self.session_path, command)
        if len(output_files) == 0:
            output_files = None  # labels the task as empty if no output
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
    gpu = 1
    cpu = 4
    io_charge = 90
    level = 1

    def _run(self):
        """empty placeholder for job creation only"""
        pass


class TrainingExtractionPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path, **kwargs):
        super(TrainingExtractionPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks['TrainingRegisterRaw'] = TrainingRegisterRaw(self.session_path)
        tasks['TrainingTrials'] = TrainingTrials(self.session_path)
        tasks['TrainingVideoCompress'] = TrainingVideoCompress(self.session_path)
        tasks['TrainingAudio'] = TrainingAudio(self.session_path)
        # level 1
        tasks['TrainingDLC'] = TrainingDLC(
            self.session_path, parents=[tasks['TrainingVideoCompress']])
        self.tasks = tasks

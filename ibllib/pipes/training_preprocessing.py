import json
import logging
from pathlib import Path
from collections import OrderedDict

from alf.io import get_session_path
from ibllib.pipes import tasks
from ibllib.io import ffmpeg, raw_data_loaders as rawio
from ibllib.io.extractors import (training_trials, biased_trials, training_wheel, biased_wheel,
                                  training_audio)


_logger = logging.getLogger('ibllib')


#  level 0
class TrainingTrials(tasks.Task):
    priority = 90
    level = 0

    def _run(self):
        """
        Extracts an iblrig training session
        """
        _, _, output_files = extract_training(self.session_path, save=True)
        return output_files


class TrainingVideoCompress(tasks.Task):

    def _run(self):
        # avi to mp4 compression
        command = ('ffmpeg -i {file_in} -codec:v libx264 -preset slow -crf 29 '
                   '-nostats -loglevel 0 -codec:a copy {file_out}')
        output_files = ffmpeg.iblrig_video_compression(self.session_path, command)
        return output_files


class TrainingAudio(tasks.Task):
    """
    Computes raw electrophysiology QC
    """
    cpu = 2
    priority = 10  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything

    def _run(self, overwrite=False):
        training_audio.extract_sound(self.session_path, save=True, delete=True)


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
        tasks['TrainingTrials'] = TrainingTrials(self.session_path)
        tasks['TrainingVideoCompress'] = TrainingVideoCompress(self.session_path)
        tasks['TrainingAudio'] = TrainingAudio(self.session_path)
        # level 1
        tasks['TrainingDLC'] = TrainingDLC(
            self.session_path, parents=[tasks['TrainingVideoCompress']])
        self.jobs = tasks


def get_task_extractor_type(task_name):
    """
    Splits the task name according to naming convention:
    -   ignores everything
    _iblrig_tasks_biasedChoiceWorld3.7.0 returns "biased"
    _iblrig_tasks_trainingChoiceWorld3.6.0 returns "training'
    :param task_name:
    :return: one of ['biased', 'habituation', 'training', 'ephys', 'mock_ephys', 'sync_ephys']
    """
    if isinstance(task_name, Path):
        try:
            settings = rawio.load_settings(get_session_path(task_name))
        except json.decoder.JSONDecodeError:
            return
        if settings:
            task_name = settings.get('PYBPOD_PROTOCOL', None)
        else:
            return
    if '_biasedChoiceWorld' in task_name:
        return 'biased'
    elif 'biasedScanningChoiceWorld' in task_name:
        return 'biased'
    elif 'biasedVisOffChoiceWorld' in task_name:
        return 'biased'
    elif '_habituationChoiceWorld' in task_name:
        return 'habituation'
    elif '_trainingChoiceWorld' in task_name:
        return 'training'
    elif 'ephysChoiceWorld' in task_name:
        return 'ephys'
    elif 'ephysMockChoiceWorld' in task_name:
        return 'mock_ephys'
    elif task_name and task_name.startswith('_iblrig_tasks_ephys_certification'):
        return 'sync_ephys'


def get_session_extractor_type(session_path):
    """
    From a session path, loads the settings file, finds the task and checks if extractors exist
    task names examples:
    :param session_path:
    :return: bool
    """
    settings = rawio.load_settings(session_path)
    if settings is None:
        _logger.error(f'ABORT: No data found in "raw_behavior_data" folder {session_path}')
        return False
    extractor_type = get_task_extractor_type(settings['PYBPOD_PROTOCOL'])
    if extractor_type:
        return extractor_type
    else:
        _logger.warning(str(session_path) +
                        f" No extractors were found for {extractor_type} ChoiceWorld")
        return False


def extract_training(session_path, save=True):
    """
    Extracts a training session from its path
    :param session_path:
    :param save:
    :return: trials: Bunch/dict of trials
    :return: wheel: Bunch/dict of wheel positions
    :return: out_Files: list of output files
    """
    extractor_type = get_session_extractor_type(session_path)
    _logger.info(f"Extracting {session_path} as {extractor_type}")
    settings, bpod_trials = rawio.load_bpod(session_path)
    if extractor_type == 'training':
        _logger.info('training session on ' + settings['PYBPOD_BOARD'])
        trials, files_trials = training_trials.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
        wheel, files_wheel = training_wheel.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
    if extractor_type == 'biased':
        _logger.info('biased session on ' + settings['PYBPOD_BOARD'])
        trials, files_trials = biased_trials.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
        wheel, files_wheel = biased_wheel.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
    _logger.info('session extracted \n')  # timing info in log
    return trials, wheel, files_trials + files_wheel

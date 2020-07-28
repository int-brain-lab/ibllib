import logging
from collections import OrderedDict

from ibllib.pipes import tasks
from ibllib.io import ffmpeg, raw_data_loaders as rawio
from ibllib.io.extractors import (training_trials, biased_trials, training_wheel, training_audio)
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


def extract_training(session_path, save=True):
    """
    Extracts a training session from its path.  NB: Wheel must be extracted first in order to
    extract trials.firstMovement_times.
    :param session_path:
    :param save:
    :return: trials: Bunch/dict of trials
    :return: wheel: Bunch/dict of wheel positions
    :return: out_Files: list of output files
    """
    extractor_type = rawio.get_session_extractor_type(session_path)
    _logger.info(f"Extracting {session_path} as {extractor_type}")
    settings, bpod_trials = rawio.load_bpod(session_path)
    if extractor_type == 'training':
        _logger.info('training session on ' + settings['PYBPOD_BOARD'])
        wheel, files_wheel = training_wheel.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
        trials, files_trials = training_trials.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
    elif extractor_type == 'biased':
        _logger.info('biased session on ' + settings['PYBPOD_BOARD'])
        wheel, files_wheel = training_wheel.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
        trials, files_trials = biased_trials.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
    elif extractor_type == 'habituation':
        _logger.info('Skipped trial extraction for habituation session')
        return None, None, None
    else:
        raise ValueError(f"No extractor for task {extractor_type}")
    _logger.info('session extracted \n')  # timing info in log
    return trials, wheel, files_trials + files_wheel

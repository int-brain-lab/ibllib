"""Trials data extraction from raw Bpod output
This module will extract the Bpod trials and wheel data based on the task protocol,
i.e. habituation, training or biased.
"""
import logging

from ibllib.io.extractors import (habituation_trials, training_trials, biased_trials,
                                  training_wheel, opto_trials)
import ibllib.io.extractors.base
import ibllib.io.raw_data_loaders as rawio
from ibllib.misc import version

_logger = logging.getLogger('ibllib')


def extract_all(session_path, save=True, bpod_trials=None, settings=None):
    """
    Extracts a training session from its path.  NB: Wheel must be extracted first in order to
    extract trials.firstMovement_times.
    :param session_path: the path to the session to be extracted
    :param save: if true a subset of the extracted data are saved as ALF
    :param bpod_trials: list of Bpod trial data
    :param settings: the Bpod session settings
    :return: trials: Bunch/dict of trials
    :return: wheel: Bunch/dict of wheel positions
    :return: out_Files: list of output files
    """
    extractor_type = ibllib.io.extractors.base.get_session_extractor_type(session_path)
    _logger.info(f"Extracting {session_path} as {extractor_type}")
    bpod_trials = bpod_trials or rawio.load_data(session_path)
    settings = settings or rawio.load_settings(session_path)
    _logger.info(f'{extractor_type} session on {settings["PYBPOD_BOARD"]}')
    if extractor_type == 'training':
        wheel, files_wheel = training_wheel.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
        trials, files_trials = training_trials.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
    elif extractor_type in ['biased', 'ephys']:
        wheel, files_wheel = training_wheel.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
        trials, files_trials = biased_trials.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
    elif extractor_type in ['biased_opto', 'ephys_biased_opto']:
        wheel, files_wheel = training_wheel.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
        trials, files_trials = opto_trials.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
    elif extractor_type == 'habituation':
        if version.le(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
            _logger.warning("No extraction of legacy habituation sessions")
            return None, None, None
        trials, files_trials = habituation_trials.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save)
        wheel = None
        files_wheel = []
    else:
        raise ValueError(f"No extractor for task {extractor_type}")
    _logger.info('session extracted \n')  # timing info in log
    return trials, wheel, (files_trials + files_wheel) if save else None

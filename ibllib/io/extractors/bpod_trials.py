"""Trials data extraction from raw Bpod output
This module will extract the Bpod trials and wheel data based on the task protocol,
i.e. habituation, training or biased.
"""
import logging
from collections import OrderedDict

from pkg_resources import parse_version
from ibllib.io.extractors import habituation_trials, training_trials, biased_trials, opto_trials
import ibllib.io.extractors.base
import ibllib.io.raw_data_loaders as rawio

_logger = logging.getLogger(__name__)


def extract_all(session_path, save=True, bpod_trials=None, settings=None, task_collection='raw_behavior_data', save_path=None):
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
    extractor_type = ibllib.io.extractors.base.get_session_extractor_type(session_path, task_collection=task_collection)
    _logger.info(f"Extracting {session_path} as {extractor_type}")
    bpod_trials = bpod_trials or rawio.load_data(session_path, task_collection=task_collection)
    settings = settings or rawio.load_settings(session_path, task_collection=task_collection)
    _logger.info(f'{extractor_type} session on {settings["PYBPOD_BOARD"]}')

    # Determine which additional extractors are required
    extra = []
    if extractor_type == 'ephys':  # Should exclude 'ephys_biased'
        _logger.debug('Engaging biased TrialsTableEphys')
        extra.append(biased_trials.TrialsTableEphys)
    if extractor_type in ['biased_opto', 'ephys_biased_opto']:
        _logger.debug('Engaging opto_trials LaserBool')
        extra.append(opto_trials.LaserBool)

    # Determine base extraction
    if extractor_type in ['training', 'ephys_training']:
        trials, files_trials = training_trials.extract_all(session_path, bpod_trials=bpod_trials, settings=settings, save=save,
                                                           task_collection=task_collection, save_path=save_path)
        # This is hacky but avoids extracting the wheel twice.
        # files_trials should contain wheel files at the end.
        files_wheel = []
        wheel = OrderedDict({k: trials.pop(k) for k in tuple(trials.keys()) if 'wheel' in k})
    elif 'biased' in extractor_type or 'ephys' in extractor_type:
        trials, files_trials = biased_trials.extract_all(
            session_path, bpod_trials=bpod_trials, settings=settings, save=save, extra_classes=extra,
            task_collection=task_collection, save_path=save_path)

        files_wheel = []
        wheel = OrderedDict({k: trials.pop(k) for k in tuple(trials.keys()) if 'wheel' in k})
    elif extractor_type == 'habituation':
        if settings['IBLRIG_VERSION_TAG'] and \
                parse_version(settings['IBLRIG_VERSION_TAG']) <= parse_version('5.0.0'):
            _logger.warning("No extraction of legacy habituation sessions")
            return None, None, None
        trials, files_trials = habituation_trials.extract_all(session_path, bpod_trials=bpod_trials, settings=settings, save=save,
                                                              task_collection=task_collection, save_path=save_path)
        wheel = None
        files_wheel = []
    else:
        raise ValueError(f"No extractor for task {extractor_type}")
    _logger.info('session extracted \n')  # timing info in log
    return trials, wheel, (files_trials + files_wheel) if save else None

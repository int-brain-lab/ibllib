"""Trials data extraction from raw Bpod output
This module will extract the Bpod trials and wheel data based on the task protocol,
i.e. habituation, training or biased.
"""
import logging
from collections import OrderedDict

from pkg_resources import parse_version
from ibllib.io.extractors import habituation_trials, training_trials, biased_trials, opto_trials
from ibllib.io.extractors.base import get_session_extractor_type
import ibllib.io.raw_data_loaders as rawio

_logger = logging.getLogger(__name__)


def extract_all(session_path, save=True, bpod_trials=None, settings=None,
                task_collection='raw_behavior_data', extractor_type=None, save_path=None):
    """
    Extracts a training session from its path.  NB: Wheel must be extracted first in order to
    extract trials.firstMovement_times.

    Parameters
    ----------
    session_path : str, pathlib.Path
        The path to the session to be extracted.
    task_collection : str
        The subfolder containing the raw Bpod data files.
    save : bool
        If true, save the output files to save_path.
    bpod_trials : list of dict
        The loaded Bpod trial data. If None, attempts to load _iblrig_taskData.raw from
        raw_task_collection.
    settings : dict
        The loaded Bpod settings.  If None, attempts to load _iblrig_taskSettings.raw from
        raw_task_collection.
    extractor_type : str
        The type of extraction.  Supported types are {'ephys', 'biased', 'biased_opto',
        'ephys_biased_opto', 'training', 'ephys_training', 'habituation'}.  If None, extractor type
        determined from settings.
    save_path : str, pathlib.Path
        The location of the output files if save is true.  Defaults to <session_path>/alf.

    Returns
    -------
    dict
        The extracted trials data.
    dict
        The extracted wheel data.
    list of pathlib.Path
        The output files if save is true.
    """
    if not extractor_type:
        extractor_type = get_session_extractor_type(session_path, task_collection=task_collection)
    _logger.info(f'Extracting {session_path} as {extractor_type}')
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
            _logger.warning('No extraction of legacy habituation sessions')
            return None, None, None
        trials, files_trials = habituation_trials.extract_all(session_path, bpod_trials=bpod_trials, settings=settings, save=save,
                                                              task_collection=task_collection, save_path=save_path)
        wheel = None
        files_wheel = []
    else:
        raise ValueError(f'No extractor for task {extractor_type}')
    _logger.info('session extracted \n')  # timing info in log
    return trials, wheel, (files_trials + files_wheel) if save else None

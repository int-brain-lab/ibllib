"""Trials data extraction from raw Bpod output.

This module will extract the Bpod trials and wheel data based on the task protocol,
i.e. habituation, training or biased.
"""
import logging
import importlib

from ibllib.io.extractors.base import get_bpod_extractor_class, protocol2extractor
from ibllib.io.extractors.habituation_trials import HabituationTrials
from ibllib.io.extractors.training_trials import TrainingTrials
from ibllib.io.extractors.biased_trials import BiasedTrials, EphysTrials
from ibllib.io.extractors.base import BaseBpodTrialsExtractor

_logger = logging.getLogger(__name__)


def get_bpod_extractor(session_path, protocol=None, task_collection='raw_behavior_data') -> BaseBpodTrialsExtractor:
    """
    Returns an extractor for a given session.

    Parameters
    ----------
    session_path : str, pathlib.Path
        The path to the session to be extracted.
    protocol : str, optional
        The protocol name, otherwise uses the PYBPOD_PROTOCOL key in iblrig task settings files.
    task_collection : str
        The folder within the session that contains the raw task data.

    Returns
    -------
    BaseBpodTrialsExtractor
        An instance of the task extractor class, instantiated with the session path.
    """
    builtins = {
        'HabituationTrials': HabituationTrials,
        'TrainingTrials': TrainingTrials,
        'BiasedTrials': BiasedTrials,
        'EphysTrials': EphysTrials
    }
    if protocol:
        class_name = protocol2extractor(protocol)
    else:
        class_name = get_bpod_extractor_class(session_path, task_collection=task_collection)
    if class_name in builtins:
        return builtins[class_name](session_path)

    # look if there are custom extractor types in the personal projects repo
    if not class_name.startswith('projects.'):
        class_name = 'projects.' + class_name
    module, class_name = class_name.rsplit('.', 1)
    mdl = importlib.import_module(module)
    extractor_class = getattr(mdl, class_name, None)
    if extractor_class:
        return extractor_class(session_path)
    else:
        raise ValueError(f'extractor {class_name} not found')

"""Base Extractor classes
A module for the base Extractor classes.  The Extractor, given a session path, will extract the
processed data from raw hardware files and optionally save them.
"""

import abc
from collections import OrderedDict
import json
from pathlib import Path

import numpy as np
import pandas as pd
from one.alf.files import get_session_path
from ibllib.io import raw_data_loaders as raw
from ibllib.io.raw_data_loaders import load_settings, _logger


class BaseExtractor(abc.ABC):
    """
    Base extractor class
    Writing an extractor checklist:
    -   on the child class, overload the _extract method
    -   this method should output one or several numpy.arrays or dataframe with a consistent shape
    -   save_names is a list or a string of filenames, there should be one per dataset
    -   set save_names to None for a dataset that doesn't need saving (could be set dynamically
    in the _extract method)
    :param session_path: Absolute path of session folder
    :type session_path: str/Path
    """

    session_path = None
    save_names = None
    default_path = Path("alf")  # relative to session

    def __init__(self, session_path=None):
        # If session_path is None Path(session_path) will fail
        self.session_path = Path(session_path)

    def extract(self, save=False, path_out=None, **kwargs):
        """
        :return: numpy.ndarray or list of ndarrays, list of filenames
        :rtype: dtype('float64')
        """
        out = self._extract(**kwargs)
        files = self._save(out, path_out=path_out) if save else None
        return out, files

    def _save(self, data, path_out=None):
        # Chack if self.save_namesis of the same length of out
        if not path_out:
            path_out = self.session_path.joinpath(self.default_path)
        path_out.mkdir(exist_ok=True, parents=True)

        def _write_to_disk(file_path, data):
            """Implements different save calls depending on file extension"""
            csv_separators = {
                ".csv": ",",
                ".ssv": " ",
                ".tsv": "\t",
            }
            file_path = Path(file_path)
            if file_path.suffix == ".npy":
                np.save(file_path, data)
            elif file_path.suffix in [".parquet", ".pqt"]:
                if not isinstance(data, pd.DataFrame):
                    _logger.error("Data is not a panda's DataFrame object")
                    raise TypeError("Data is not a panda's DataFrame object")
                data.to_parquet(file_path)
            elif file_path.suffix in [".csv", ".ssv", ".tsv"]:
                sep = csv_separators[file_path.suffix]
                data.to_csv(file_path, sep=sep)
                # np.savetxt(file_path, data, delimiter=sep)
            else:
                _logger.error(f"Don't know how to save {file_path.suffix} files yet")

        if self.save_names is None:
            file_paths = []
        elif isinstance(self.save_names, str):
            file_paths = path_out.joinpath(self.save_names)
            _write_to_disk(file_paths, data)
        else:  # Should be list or tuple...
            assert len(data) == len(self.save_names)
            file_paths = []
            for data, fn in zip(data, self.save_names):
                if fn:
                    fpath = path_out.joinpath(fn)
                    _write_to_disk(fpath, data)
                    file_paths.append(fpath)
        return file_paths

    @abc.abstractmethod
    def _extract(self):
        pass


class BaseBpodTrialsExtractor(BaseExtractor):
    """
    Base (abstract) extractor class for bpod jsonable data set
    Wrps the _extract private method

    :param session_path: Absolute path of session folder
    :type session_path: str
    :param bpod_trials
    :param settings
    """

    bpod_trials = None
    settings = None

    def extract(self, bpod_trials=None, settings=None, **kwargs):
        """
        :param: bpod_trials (optional) bpod trials from jsonable in a dictionary
        :param: settings (optional) bpod iblrig settings json file in a dictionary
        :param: save (bool) write output ALF files, defaults to False
        :param: path_out (pathlib.Path) output path (defaults to `{session_path}/alf`)
        :return: numpy.ndarray or list of ndarrays, list of filenames
        :rtype: dtype('float64')
        """
        self.bpod_trials = bpod_trials
        self.settings = settings
        if self.bpod_trials is None:
            self.bpod_trials = raw.load_data(self.session_path)
        if not self.settings:
            self.settings = raw.load_settings(self.session_path)
        if self.settings is None:
            self.settings = {"IBLRIG_VERSION_TAG": "100.0.0"}
        elif self.settings["IBLRIG_VERSION_TAG"] == "":
            self.settings["IBLRIG_VERSION_TAG"] = "100.0.0"
        return super(BaseBpodTrialsExtractor, self).extract(**kwargs)


def run_extractor_classes(classes, session_path=None, **kwargs):
    """
    Run a set of extractors with the same inputs
    :param classes: list of Extractor class
    :param save: True/False
    :param path_out: (defaults to alf path)
    :param kwargs: extractor arguments (session_path...)
    :return: dictionary of arrays, list of files
    """
    files = []
    outputs = OrderedDict({})
    assert session_path
    # if a single class is passed, convert as a list
    try:
        iter(classes)
    except TypeError:
        classes = [classes]
    for classe in classes:
        cls = classe(session_path=session_path)
        out, fil = cls.extract(**kwargs)
        if isinstance(fil, list):
            files.extend(fil)
        elif fil is not None:
            files.append(fil)
        if isinstance(cls.var_names, str):
            outputs[cls.var_names] = out
        else:
            for i, k in enumerate(cls.var_names):
                outputs[k] = out[i]
    return outputs, files


def _get_task_types_json_config():
    with open(Path(__file__).parent.joinpath('extractor_types.json')) as fp:
        task_types = json.load(fp)
    return task_types


def get_task_protocol(session_path):
    try:
        settings = load_settings(get_session_path(session_path))
    except json.decoder.JSONDecodeError:
        _logger.error(f"Can't read settings for {session_path}")
        return
    if settings:
        return settings.get('PYBPOD_PROTOCOL', None)
    else:
        return


def get_task_extractor_type(task_name):
    """
    Returns the task type string from the full pybpod task name:
    _iblrig_tasks_biasedChoiceWorld3.7.0 returns "biased"
    _iblrig_tasks_trainingChoiceWorld3.6.0 returns "training'
    :param task_name:
    :return: one of ['biased', 'habituation', 'training', 'ephys', 'mock_ephys', 'sync_ephys']
    """
    if isinstance(task_name, Path):
        task_name = get_task_protocol(task_name)
        if task_name is None:
            return
    task_types = _get_task_types_json_config()
    task_type = next((task_types[tt] for tt in task_types if tt in task_name), None)
    if task_type is None:
        _logger.warning(f"No extractor type found for {task_name}")
    return task_type


def get_session_extractor_type(session_path):
    """
    From a session path, loads the settings file, finds the task and checks if extractors exist
    task names examples:
    :param session_path:
    :return: bool
    """
    settings = load_settings(session_path)
    if settings is None:
        _logger.error(f'ABORT: No data found in "raw_behavior_data" folder {session_path}')
        return False
    extractor_type = get_task_extractor_type(settings['PYBPOD_PROTOCOL'])
    if extractor_type:
        return extractor_type
    else:
        return False


def get_pipeline(session_path):
    """
    Get the pre-processinf pipeline name from a session path
    :param session_path:
    :return:
    """
    stype = get_session_extractor_type(session_path)
    return _get_pipeline_from_task_type(stype)


def _get_pipeline_from_task_type(stype):
    """
    Returns the pipeline from the task type. Some tasks types directly define the pipeline
    :param stype: session_type or task extractor type
    :return:
    """
    if stype in ['ephys_biased_opto', 'ephys', 'ephys_training', 'mock_ephys', 'sync_ephys']:
        return 'ephys'
    elif stype in ['habituation', 'training', 'biased', 'biased_opto']:
        return 'training'
    else:
        return stype

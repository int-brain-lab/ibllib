from pathlib import Path
import abc

import numpy as np

from ibllib.io import raw_data_loaders as raw


class BaseBpodTrialsExtractor(abc.ABC):
    """
    Base extractor class for bpod jsonable data set
    Wrps the _extract private method

    :param session_path: Absolute path of session folder
    :type session_path: str
    :param bpod_trials
    :param settings
    :return: numpy.ndarray or list of ndarrays, list of filenames
    :rtype: dtype('float64')
    """
    bpod_trials = None
    settings = None
    session_path = None
    save_names = None

    def __init__(self, session_path=None, bpod_trials=None, settings=None):
        self.session_path = Path(session_path)
        self.bpod_trials = bpod_trials
        self.settings = settings
        if not self.bpod_trials:
            self.bpod_trials = raw.load_data(session_path)
        if not self.settings:
            self.settings = raw.load_settings(session_path)

    def extract(self, save=False, path_out=None, return_files=False, **kwargs):
        if self.settings is None:
            self.settings = {'IBLRIG_VERSION_TAG': '100.0.0'}
        elif self.settings['IBLRIG_VERSION_TAG'] == '':
            self.settings['IBLRIG_VERSION_TAG'] = '100.0.0'
        out = self._extract(**kwargs)
        if not save:
            return out, None
        else:
            files = self._save(out, path_out=path_out) if save else None
        return out, files

    def _save(self, out, path_out=None):
        if not path_out:
            path_out = self.session_path.joinpath('alf')
        path_out.mkdir(exist_ok=True, parents=True)
        if isinstance(self.save_names, str):
            files = path_out.joinpath(self.save_names)
            np.save(files, out)
        else:
            files = []
            for i, fn in enumerate(self.save_names):
                np.save(path_out.joinpath(fn), out[i])
                files.append(path_out.joinpath(fn))
        return files

    @abc.abstractmethod
    def _extract(self):
        pass


def run_extractor_classes(classes, save=False, **kwargs):
    """
    Run a set of extractor with the same inputs
    :param classes: list of Extractor class
    :param save: True/False
    :param kwargs: extractor arguments (session_path...)
    :return: arrays, files
    """
    files = []
    outputs = {}
    for classe in classes:
        out, fil = classe(**kwargs).extract(save=save)
        if isinstance(fil, list):
            files.append(fil)
        else:
            files.extend([fil])
        var = [classe.var_names] if isinstance(classe.var_names, str) else classe.var_names
        for k in var:
            outputs[k] = out
    return outputs, files

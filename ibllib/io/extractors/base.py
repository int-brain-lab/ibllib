from pathlib import Path
import abc
from collections import OrderedDict
import numpy as np

from ibllib.io import raw_data_loaders as raw


class BaseExtractor(abc.ABC):
    """
    Base extractor class

    :param session_path: Absolute path of session folder
    :type session_path: str
    """
    session_path = None
    save_names = None
    default_path = Path('alf')  # relative to session

    def __init__(self, session_path=None):
        self.session_path = Path(session_path)

    def extract(self, save=False, path_out=None, **kwargs):
        """
        :return: numpy.ndarray or list of ndarrays, list of filenames
        :rtype: dtype('float64')
        """
        out = self._extract(**kwargs)
        if not save:
            return out, None
        else:
            files = self._save(out, path_out=path_out) if save else None
        return out, files

    def _save(self, out, path_out=None):
        if not path_out:
            path_out = self.session_path.joinpath(self.default_path)
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
            self.settings = {'IBLRIG_VERSION_TAG': '100.0.0'}
        elif self.settings['IBLRIG_VERSION_TAG'] == '':
            self.settings['IBLRIG_VERSION_TAG'] = '100.0.0'
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
    for classe in classes:
        out, fil = classe(session_path=session_path).extract(**kwargs)
        if isinstance(fil, list):
            files.extend(fil)
        elif fil is not None:
            files.append(fil)
        if isinstance(classe.var_names, str):
            outputs[classe.var_names] = out
        else:
            for i, k in enumerate(classe.var_names):
                outputs[k] = out[i]
    assert (len(files) == 0) or (len(files) == len(outputs.keys()))
    return outputs, files

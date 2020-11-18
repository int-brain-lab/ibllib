import abc
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from ibllib.io import raw_data_loaders as raw
import logging


log = logging.getLogger("ibllib")


class BaseExtractor(abc.ABC):
    """
    Base extractor class

    :param session_path: Absolute path of session folder
    :type session_path: str
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
            if data is None or not data:
                log.error("Data is empty or None, not saving")
                return
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
                    log.error("Data is not a panda's DataFrame object")
                    raise TypeError("Data is not a panda's DataFrame object")
                data.to_parquet(file_path)
            elif file_path.suffix in [".csv", ".ssv", ".tsv"]:
                sep = csv_separators[file_path.suffix]
                data.to_csv(file_path, sep=sep)
                # np.savetxt(file_path, data, delimiter=sep)
            else:
                log.error(f"Don't know how to save {file_path.suffix} files yet")
            return file_path

        if isinstance(self.save_names, str):
            fpath = path_out.joinpath(self.save_names)
            out_paths = _write_to_disk(fpath, data)
        else:  # Should be list or tuple...
            assert len(data) == len(self.save_names)
            out_paths = []
            for data, fn in zip(data, self.save_names):
                fpath = path_out.joinpath(fn)
                saved_path = _write_to_disk(fpath, data)
                out_paths.append(saved_path)
        return out_paths

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

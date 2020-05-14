from pathlib import Path, PureWindowsPath
from ibllib.io.extractors.base import BaseBpodTrialsExtractor, run_extractor_classes
import numpy as np

import ibllib.io.raw_data_loaders as raw
from ibllib.io.extractors.biased_trials import (
    Choice, ContrastLR, FeedBackType, GoCueTriggerTimes, Intervals, ItiDuration, ResponseTimes,
    RewardVolume)


class ProbabilityLeft(BaseBpodTrialsExtractor):
    save_names = '_ibl_trials.probabilityLeft.npy'
    var_names = 'probabilityLeft'

    def _extract(self):
        num = self.settings.get("PRELOADED_SESSION_NUM", None)
        if num is None:
            num = self.settings.get("PREGENERATED_SESSION_NUM", None)
        if num is None:
            fn = self.settings.get('SESSION_LOADED_FILE_PATH', None)
            fn = PureWindowsPath(fn).name
            num = ''.join([d for d in fn if d.isdigit()])
            if num == '':
                raise ValueError("Can't extract left probability behaviour.")
        # Load the pregenerated file
        sessions_folder = Path(raw.__file__).parent.joinpath('extractors', 'ephys_sessions')
        fname = f"session_{num}_ephys_pcqs.npy"
        pcqsp = np.load(sessions_folder.joinpath(fname))
        pLeft = pcqsp[:, 4]
        pLeft = pLeft[: len(self.bpod_trials)]
        return pLeft


def extract_all(session_path, save=False, data=False, output_path=None, return_files=False):
    """
    Extract all behaviour data from Bpod whithin the specified folder.
    The timing information from FPGA is extracted in
    :func:`~ibllib.io.extractors.ephys_fpga`

    :param session_path: folder containing sessions
    :type session_path: str or pathlib.Path
    :param save: bool
    :param data: raw Bpod data dictionary
    :param return_files: (bool) if True returns list of pathlib.Path output. Defaults to False.
    :return: dictionary of trial related vectors (one row per trial)
    """
    if not data:
        data = raw.load_data(session_path)
    base = [FeedBackType, ContrastLR, ProbabilityLeft, Choice, RewardVolume, ItiDuration,
            Intervals, ResponseTimes, GoCueTriggerTimes]
    out, fil = run_extractor_classes(
        base, save=save, session_path=session_path, bpod_trials=data)
    return out, fil

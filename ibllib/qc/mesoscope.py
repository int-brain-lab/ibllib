"""Mesoscope quality control.

This module runs a list of quality control metrics on the extracted imaging and cell detection
data.
"""
import logging
from inspect import getmembers, isfunction
import unittest
from pathlib import Path

import numpy as np
import scipy.stats

from . import base

_log = logging.getLogger(__name__)


def get_neural_quality_metrics(F, Fneu, badframes=None, iscell=None, F0_percentile=20, neuropil_factor=.7, **kwargs):
    """Compute neural quality metrics based on raw fluorescence traces.

    Parameters
    ----------
    F : numpy.array
        Raw fluorescence trace (nROIs by nTimepoints), e.g. mpci.ROINeuropilActivityF.
    Fneu : numpy.array
        Raw neuropil trace (nROIs by nTimepoints), e.g. mpci.ROIActivityF.
    badframes : numpy.array
        Indices of frames that should be excluded (default = all false).
    iscell : numpy.array
        Boolean array with true for cells, and false for not cells (default = all true).
    F0_percentile : int
        Percentile to be used for computing baseline fluorescence F0 (default = 20).
    neuropil_factor : float
        Factor to multiply neuropil with to get neuropil-corrected trace (default = 0.7).
        Must be between 0 and 1.
    times : numpy.array
        An array of frame times used to infer the frame rate, e.g. mpci.times.
    frame_rate : float
        The known frame rate of acquisition in Hz.  This value takes precedence when times
        also passed in. If neither times nor frame_rate passed, defaults to 7 Hz.

    Returns
    -------
    dict
        A dictionary with the following keys:
            noiseLevel: standardized shot noise level
            mean: time-averaged raw fluorescence (proxy for overall brightness)
            std: standard deviation of neuropil-corrected activity
            skew: skewness of neuropil-corrected activity
        Each averaged across all ROIs.
    numpy.array
        A structured numpy array with the fields noise_level, mean, std, and skew. One value
        per ROI.
    """
    if 'frame_rate' in kwargs:
        if (frame_rate := kwargs['frame_rate']) <= 0:
            raise ValueError('frame_rate must be positive')
        _log.info('Frame rate: %.2f Hz', frame_rate)
    elif 'times' in kwargs:
        frame_rate = 1 / np.median(np.diff(kwargs['times']))
        _log.info('Inferred frame rate of %.2f Hz from frame times', frame_rate)
    else:
        frame_rate = 7
        _log.warning('Assuming frame rate of %.2f Hz', frame_rate)
    if badframes is None:
        badframes = np.zeros(F.shape[0], dtype=bool)
    elif not isinstance(badframes, np.array):
        raise TypeError(f'expected `badframes` to by numpy array, got `{type(badframes)}` instead')
    if iscell is None:
        iscell = np.ones(F.shape[0], dtype=bool)
    if neuropil_factor <= 0 or neuropil_factor > 1:
        raise ValueError('neuropil_factor must be between zero and one')

    # only take the good frames
    F = F[~badframes, :]
    Fneu = Fneu[~badframes, :]

    # F_npc is neuropil corrected trace
    F_npc = F - neuropil_factor * Fneu

    # dFF is deltaF / F0 in %, i.e. baseline-normalized fluorescence trace
    if F0_percentile is None:
        dFF = F
    else:
        F0 = np.percentile(F, F0_percentile, axis=0)  # F0 is some percentile of full trace
        F0_ = np.tile(F0, (F.shape[0], 1))
        dFF = (F - F0_) / F0_ * 100

    # compute noise level
    noise_levels = np.nanmedian(np.abs(np.diff(dFF, axis=0)), axis=0) / np.sqrt(frame_rate)  # Rupprecht et al. 2021

    # compute time-averaged raw fluorescence (proxy for overall brightness)
    means = np.mean(F, axis=0)

    # compute standard deviation of neuropil-corrected activity
    stds = np.std(F_npc, axis=0, ddof=0)

    # compute skewness of neuropil-corrected activity
    skews = scipy.stats.skew(F_npc, axis=0, bias=True)

    metrics = {'noise_level': noise_levels, 'mean': means, 'std': stds, 'skew': skews}

    # return all the neural quality metrics in a struct
    fov_metrics = np.rec.fromarrays(metrics.values(), dtype=np.dtype([(k, F.dtype) for k in metrics]))

    # return all the FOV-wide mean quality metrics in a struct
    neural_metrics = {k: np.nanmean(v[iscell]) for k, v in metrics.items()}
    return neural_metrics, fov_metrics


class TestQM(unittest.TestCase):
    def setUp(self):
        self.data_path = Path(r'E:\integration\mesoscope\SP037\2023-03-23\002\alf\suite2p\plane2')

    def test_neural_qm(self):
        F = np.load(self.data_path / 'F.npy').T
        Fneu = np.load(self.data_path / 'Fneu.npy').T
        iscell = np.load(self.data_path / 'iscell.npy')
        neural_metrics, fov_metrics = get_neural_quality_metrics(F, Fneu, iscell=iscell[:, 0].astype(bool))
        expected = {'noise_level', 'mean', 'std', 'skew'}
        self.assertCountEqual(expected, neural_metrics)
        np.testing.assert_approx_equal(neural_metrics['noise_level'], 3.8925, significant=5)
        np.testing.assert_approx_equal(neural_metrics['mean'], 983.0024)
        np.testing.assert_approx_equal(neural_metrics['std'], 120.6435)
        np.testing.assert_approx_equal(neural_metrics['skew'], 1.0478, significant=5)
        self.assertCountEqual(expected, fov_metrics.dtype.names)
        self.assertTrue(all(fov_metrics[x].size == F.shape[1] for x in expected))


class MesoscopeQC(base.QC):
    """A class for computing camera QC metrics."""

    def run(self, update: bool = False, **kwargs) -> (str, dict):
        """
        Run mesoscope QC checks and return outcome.

        Parameters
        ----------
        update : bool
            If true, updates the session QC fields on Alyx.

        Returns
        -------
        str
            The overall outcome.
        dict
            A map of checks and their outcomes.
        """
        _log.info(f'Computing QC outcome for session {self.eid}')

        namespace = 'mesoscope'
        if all(x is None for x in self.data.values()):
            self.load_data(**kwargs)
        if self.data['frame_samples'] is None or self.data['timestamps'] is None:
            return 'NOT_SET', {}
        if self.data['timestamps'].shape[0] == 0:
            _log.error(f'No timestamps for {self.label} camera; setting outcome to CRITICAL')
            return 'CRITICAL', {}

        def is_metric(x):
            return isfunction(x) and x.__name__.startswith('check_')

        checks = getmembers(self.__class__, is_metric)
        self.metrics = {f'_{namespace}_' + k[6:]: fn(self) for k, fn in checks}

        values = [x if isinstance(x, str) else x[0] for x in self.metrics.values()]
        code = max(base.CRITERIA[x] for x in values)
        outcome = next(k for k, v in base.CRITERIA.items() if v == code)

        if update:
            extended = {
                k: 'NOT_SET' if v is None else v
                for k, v in self.metrics.items()
            }
            self.update_extended_qc(extended)
            self.update(outcome, namespace)
        return outcome, self.metrics

    def check_data_lengths(self, **kwargs):
        return 'NOT_SET'


if __name__ == '__main__':
    unittest.main()

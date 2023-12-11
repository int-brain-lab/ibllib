"""Mesoscope quality control.

This module runs a list of quality control metrics on the extracted imaging and cell detection
data.
"""
import logging
from inspect import getmembers, isfunction

from . import base

_log = logging.getLogger(__name__)


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

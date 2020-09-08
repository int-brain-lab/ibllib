import logging
from abc import abstractmethod
from pathlib import Path

import numpy as np

from oneibl.one import ONE
from alf.io import is_session_path, is_uuid_string


# Map for comparing QC outcomes
CRITERIA = {'CRITICAL': 4,
            'FAIL': 3,
            'WARNING': 2,
            'PASS': 1,
            'NOT_SET': 0
            }


class QC:
    """A base class for data quality control"""
    def __init__(self, session, one=None, log=None):
        self.one = one or ONE()
        self.log = log or logging.getLogger('ibllib')
        self._set_eid_or_path(session)

        self.outcome = "NOT_SET"

    @abstractmethod
    def run(self):
        """Run the QC tests and return the outcome
        :return: One of "CRITICAL", "FAIL", "WARNING" or "PASS"
        """
        pass

    @abstractmethod
    def load_data(self):
        """Load the data required to compute the QC
        Subclasses may implement this for loading raw data
        """
        pass

    def _set_eid_or_path(self, session_path_or_eid):
        """Parse a given eID or session path
        If a session UUID is given, resolves and stores the local path and vice versa
        :param session_path_or_eid:
        :return:
        """
        if is_uuid_string(str(session_path_or_eid)):
            self.eid = session_path_or_eid
            # Try to set session_path if data is found locally
            self.session_path = self.one.path_from_eid(self.eid)
        elif is_session_path(session_path_or_eid):
            self.session_path = Path(session_path_or_eid)
            self.eid = self.one.eid_from_path(self.session_path)
            if not self.eid:
                self.log.warning('Failed to determine eID from session path')
        else:
            self.log.error('Cannot run QC: an experiment uuid or session path is required')
            raise ValueError("'session' must be a valid session path or uuid")

    def update(self, outcome, namespace='experimenter', override=False):
        """Update the qc field in Alyx
        Updates the 'qc' field in Alyx if the new QC outcome is worse than the current value.
        :param outcome: A string; one of "CRITICAL", "FAIL", "WARNING", "PASS" or "NOT_SET"
        :param namespace: The extended QC key specifying the type of QC associated with the outcome
        :param override: If True the QC field is updated even if new value is better than previous
        :return: The current QC outcome str on Alyx

        Example:
            qc = QC('path/to/session')
            qc.update('PASS')  # Update current QC field to 'PASS' if not set
        """
        outcome = outcome.upper()  # Ensure outcome is uppercase
        if outcome not in CRITERIA:
            raise ValueError('Invalid outcome; must be one of ' + ', '.join(CRITERIA.keys()))
        assert self.eid, 'Unable to update Alyx; eID not set'
        if namespace:  # Record in extended qc
            self.update_extended_qc({namespace: outcome})
        current_status = self.one.alyx.rest('sessions', 'read', id=self.eid)['qc']
        if CRITERIA[current_status] < CRITERIA[outcome] or override:
            r = self.one.alyx.rest('sessions', 'partial_update', id=self.eid, data={'qc': outcome})
            current_status = r['qc'].upper()
            assert current_status == outcome, 'Failed to update session QC'
            self.log.info(f'QC field successfully updated to {outcome} for session {self.eid}')
        self.outcome = current_status
        return self.outcome

    def update_extended_qc(self, data):
        """Update the extended_qc field in Alyx
        Subclasses should choin a call to this.
        :param data: a dict of qc tests and their outcomes, typically a value between 0. and 1.
        :return: the updated extended_qc field
        """
        assert self.eid, 'Unable to update Alyx; eID not set'

        # Ensure None instead of NaNs
        for k, v in data.items():
            if (v is not None and not isinstance(v, str)) and np.isnan(v):
                data[k] = None

        extended_qc = self.one.alyx.rest('sessions', 'read', id=self.eid)['extended_qc'] or {}
        extended_qc.update(data)
        out = self.one.alyx.json_field_update(
            endpoint='sessions', uuid=self.eid, field_name='extended_qc', data=extended_qc)
        self.log.info(f'Extended QC field successfully updated for session {self.eid}')
        return out

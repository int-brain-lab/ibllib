import logging
from abc import ABC, abstractmethod

from oneibl.one import ONE
from alf.io import is_session_path, is_uuid_string


class QC(ABC):
    """A base class for data quality control"""
    def __init__(self, session, one=None, log=None):
        self.one = one or ONE()
        self.log = log or logging.getLogger("ibllib")
        self._set_eid_or_path(session)

        self.metrics = None
        self.passed = None

    @abstractmethod
    def compute(self):
        pass

    def load_data(self):
        """Load the data required to compute the QC
        Subclasses may implement this for loading raw data
        """
        pass

    def _set_eid_or_path(self, session_path_or_eid):
        if is_uuid_string(str(session_path_or_eid)):
            self.eid = session_path_or_eid
            # Try to set session_path if data is found locally
            self.session_path = self.one.path_from_eid(self.eid)
        elif is_session_path(session_path_or_eid):
            self.session_path = session_path_or_eid
            self.eid = None
        else:
            self.log.error("Cannot run QC: an experiment uuid or session path is requried")
            raise ValueError("'session' must be a valid session path or uuid")

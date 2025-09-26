"""Tests for ibllib.pipes.mesoscope_tasks."""

import unittest
import tempfile
from pathlib import Path
import iblphotometry_tests
from ibllib.pipes.neurophotometrics import FibrePhotometryBpodSync
from ibllib.io import session_params

# Mock suit2p which is imported in MesoscopePreprocess
# attrs = {'default_ops.return_value': {}}
# sys.modules['suite2p'] = mock.MagicMock(**attrs)

# from iblscripts.ci.tests import base


class TestNeurophotometricsExtractor(unittest.TestCase):
    """
    this class tests
    that the correct extractor is run based on the experiment description file
    this requires the setup to have

    """

    def setUp(self) -> None:
        self.tmp_folder = tempfile.TemporaryDirectory()
        # self.session_folder = Path(self.tmp_folder.name) / 'subject' / '2020-01-01' / '001'
        # self.raw_photometry_folder = self.session_folder / 'raw_photometry_data'
        # self.raw_photometry_folder.mkdir(parents=True)

    def test_bpod_extractor(self):
        session_folder = Path(iblphotometry_tests.__file__).parent / 'data' / 'neurophotometrics' / 'raw_bpod_session'
        assert session_folder.exists()
        self.experiment_description = session_params.read_params(session_folder)
        FibrePhotometryBpodSync()

    # def test_daqami_extractor(self):
        # path = Path(__file__).parent / 'fixtures' / 'neurophotometrics' / '_ibl_experiment_description_bpod.yaml'
        # self.experiment_description = session_params.read_params(path)

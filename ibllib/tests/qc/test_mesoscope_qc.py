import unittest
from unittest import mock
from ibllib.qc.mesoscope import MesoscopeQC, update_dataset_qc_for_collection
import numpy as np
from one.alf import spec
from pathlib import Path
from uuid import uuid4


class TestMesoscopeQC(unittest.TestCase):

    def setUp(self):
        one = mock.MagicMock()
        self.qc = MesoscopeQC('FOV_00', one=one, endpoint='fields-of-view')
        self.qc.load_data = mock.MagicMock()

    def test_only_dset(self):
        out = self.qc.check_timestamps_consistency(only_dsets=True)
        self.assertEqual(out, ['mpci.times.npy'])

    def test_check_timestamps_pass(self):
        self.qc.data = {'F': np.ones((10, 3)), 'times': np.ones(10)}
        out = self.qc.check_timestamps_consistency()
        self.assertEqual(out, spec.QC.PASS)

    def test_check_timestamps_fail(self):
        self.qc.data = {'F': np.ones((10, 3)), 'times': np.ones(8)}
        out = self.qc.check_timestamps_consistency()
        self.assertEqual(out, spec.QC.FAIL)

    @mock.patch('ibllib.qc.mesoscope.get_neural_quality_metrics')
    def test_run(self, mock_neural_quality_metrics):
        # We provide data such that the check_timestamps will fail
        self.qc.data = {'F': np.ones((10, 3)), 'times': np.ones(8)}

        # TODO change once actual thresholds are being applied
        mock_neural_quality_metrics.return_value = {}, {}

        out, metrics, dsets = self.qc.run()

        # TODO change if dependent on other qc tests
        self.assertEqual(out, spec.QC.FAIL)
        self.assertEqual(len(dsets), 4)
        # TODO update datasets as necessary with checks
        for name in ['mpci.ROIActivityF.npy', 'mpci.ROINeuropilActivityF.npy', 'mpciROIs.mpciROITypes.npy']:
            self.assertEqual(dsets.get(name), spec.QC.WARNING)

        # Make sure for the mpci.times we get the worst of the qc values available
        self.assertEqual(dsets['mpci.times.npy'], spec.QC.FAIL)


class TestMesoscopeSessionQC(unittest.TestCase):

    def setUp(self):
        self.one = mock.MagicMock()
        self.one.offline = False

    @mock.patch('ibllib.qc.base.QC.update')
    @mock.patch('ibllib.qc.mesoscope.MesoscopeQC.update')
    @mock.patch('ibllib.qc.mesoscope.MesoscopeQC.run')
    def test_session_qc(self, meso_qc, meso_update, base_update):

        names = [{'name': 'FOV_00'}, {'name': 'FOV_01'}, {'name': 'FOV_02'}]

        def rest_function(*args, **kwargs):
            # This gets the rest method
            method = args[1]
            if method == 'list':
                return [{'id': str(uuid4()), 'name': 'FOV_00'}, {'id': str(uuid4()), 'name': 'FOV_01'},
                        {'id': str(uuid4()), 'name': 'FOV_02'},]
            elif method == 'read':
                return names.pop(0)
            else:
                return None

        base_update.return_value = lambda *args, **kwargs: {'qc': spec.QC.NOT_SET.name,
                                                            'json': {'extended_qc': None}}
        meso_update.return_value = lambda *args, **kwargs: {'qc': spec.QC.NOT_SET.name,
                                                            'json': {'extended_qc': None}}
        meso_qc.side_effect = [(spec.QC.PASS, {}, []), (spec.QC.WARNING, {}, []), (spec.QC.WARNING, {}, [])]

        self.one.list_collections.return_value = ['alf/FOV_00', 'alf/FOV_01', 'alf/FOV_02']
        self.one.alyx.rest.side_effect = rest_function
        self.one.eid2path.return_value = Path('/mnt/s0/Data/Subjects/SP054/2022-03-23/001')

        out = MesoscopeQC.qc_session(str(uuid4()), self.one)
        self.assertEqual(len(out), 3)
        self.assertEqual(list(out.keys()), ['alf/FOV_00', 'alf/FOV_01', 'alf/FOV_02'])
        self.assertEqual(out['alf/FOV_00'][0], spec.QC.PASS)
        self.assertEqual(out['alf/FOV_01'][0], spec.QC.WARNING)
        self.assertEqual(out['alf/FOV_02'][0], spec.QC.WARNING)


class TestDatasetQC(unittest.TestCase):

    def setUp(self):

        self.one = mock.MagicMock()

        self.list_datasets = [
            [{'name': 'mpci.ROIActivityF.npy', 'qc': 'NOT_SET', 'url': f'lala/{str(uuid4())}', 'default_dataset': True}],
            [{'name': 'mpci.ROINeuropilActivityF.npy', 'qc': 'NOT_SET', 'url': f'lala/{str(uuid4())}', 'default_dataset': True}],
            [], []
        ]

        self.dataset_qc = {
            'mpci.ROIActivityF.npy': spec.QC.WARNING,
            'mpci.ROINeuropilActivityF.npy': spec.QC.PASS,
            'mpciROIs.mpciROITypes.npy': spec.QC.WARNING,
            'mpci.times.npy': spec.QC.PASS,
        }

        self.registered_datasets = [
            {'name': 'mpci.times.npy', 'qc': 'NOT_SET', 'id': str(uuid4()), 'default_dataset': True},
            {'name': 'mpciROIs.mpciROITypes.npy', 'qc': 'NOT_SET', 'id': str(uuid4()), 'default_dataset': True},
        ]

    def test_update_dataset_qc(self):
        """Test task_metrics.update_dataset_qc function."""

        def rest_function(*args, **kwargs):
            # This gets the rest method
            method = args[1]
            if method == 'list':
                return list_datasets.pop(0)
            elif method == 'partial_update':
                return kwargs.get('data')
            else:
                return None

        self.one.alyx.rest.side_effect = rest_function
        self.one.offline = False

        # Test when no registered datasets, original dataset qc is NOT_SET and override=False
        self.one.alyx.get.side_effect = lambda *args, **kwargs: {'qc': spec.QC.NOT_SET.name, 'json': {'extended_qc': None}}
        list_datasets = self.list_datasets.copy()
        out = update_dataset_qc_for_collection(str(uuid4()), 'alf/FOV_00',
                                               self.dataset_qc, [], self.one, override=False)
        self.assertEqual(len(out), 2)
        self.assertEqual(next(o for o in out if o['name'] == 'mpci.ROIActivityF.npy')['qc'], spec.QC.WARNING.name)
        self.assertEqual(next(o for o in out if o['name'] == 'mpci.ROINeuropilActivityF.npy')['qc'], spec.QC.PASS.name)

        # Test when no registered datasets and  original dataset qc is FAIL and override=False
        self.one.reset_mock()
        self.one.alyx.get.side_effect = lambda *args, **kwargs: {'qc': spec.QC.FAIL.name, 'json': {'extended_qc': None}}
        list_datasets = self.list_datasets.copy()
        out = update_dataset_qc_for_collection(str(uuid4()), 'alf/FOV_00',
                                               self.dataset_qc, [], self.one, override=False)
        self.assertEqual(len(out), 2)
        self.assertEqual(next(o for o in out if o['name'] == 'mpci.ROIActivityF.npy')['qc'], spec.QC.FAIL.name)
        self.assertEqual(next(o for o in out if o['name'] == 'mpci.ROINeuropilActivityF.npy')['qc'], spec.QC.FAIL.name)

        # Test when no registered datasets and  original dataset qc is FAIL and override=TRUE
        self.one.reset_mock()
        self.one.alyx.get.side_effect = lambda *args, **kwargs: {'qc': spec.QC.FAIL.name, 'json': {'extended_qc': None}}
        list_datasets = self.list_datasets.copy()
        out = update_dataset_qc_for_collection(str(uuid4()), 'alf/FOV_00',
                                               self.dataset_qc, [], self.one, override=True)
        self.assertEqual(len(out), 2)
        self.assertEqual(next(o for o in out if o['name'] == 'mpci.ROIActivityF.npy')['qc'], spec.QC.WARNING.name)
        self.assertEqual(next(o for o in out if o['name'] == 'mpci.ROINeuropilActivityF.npy')['qc'], spec.QC.PASS.name)

        # Test when passing registered datasets
        self.one.reset_mock()
        self.one.alyx.get.side_effect = lambda *args, **kwargs: {'qc': spec.QC.NOT_SET.name,
                                                                 'json': {'extended_qc': None}}
        list_datasets = self.list_datasets.copy()[0:2]
        out = update_dataset_qc_for_collection(str(uuid4()), 'alf/FOV_00',
                                               self.dataset_qc, self.registered_datasets, self.one, override=True)
        self.assertEqual(len(out), 4)


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

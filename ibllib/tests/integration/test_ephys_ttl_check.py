"""
The ttl_check pipeline is to validate a setup before doing recordings,
on the ephysChoiceWorld task, doing a dummy without animal and check that all syncs are there

This test is also a synchronization extraction test, as it checks the ouput. The tear down function
removes all _spikeglx_ files so they are regenerated from the small files accessible
"""
import ibllib.ephys.ephysqc

from ci.tests import base


class TestEphysCheckList(base.IntegrationTest):

    required_files = ['ephys/ttl_check']

    def setUp(self):
        self.init_folder = self.data_path.joinpath('ephys', 'ttl_check')
        if not self.init_folder.exists():
            return
        for sf in self.init_folder.joinpath('ttl_3B_single').rglob('_spikeglx_sync.*.npy'):
            sf.unlink()

    def test_checklist_mock_3B_single(self):
        ses_path = self.init_folder / 'ttl_3B_single'
        self.assertTrue(ibllib.ephys.ephysqc.validate_ttl_test(ses_path))

    def test_checklist_mock_3A_single(self):
        ses_path = self.init_folder / 'ttl_3A_single'
        self.assertTrue(ibllib.ephys.ephysqc.validate_ttl_test(ses_path))

    def test_checklist_mock_3A_dual(self):
        ses_path = self.init_folder / 'ttl_3A_dual'
        self.assertTrue(ibllib.ephys.ephysqc.validate_ttl_test(ses_path))

    def test_checklist_from_arbitrary_folder(self):
        ses_path = self.init_folder / 'ttl_3A_dual'
        self.assertTrue(ibllib.ephys.ephysqc.validate_ttl_test(ses_path))
        ibllib.ephys.ephysqc.validate_ttl_test(ses_path / 'raw_ephys_data')

    def test_checklist_mock_3B_dual(self):
        ses_path = self.init_folder / 'ttl_3B_dual'
        self.assertTrue(ibllib.ephys.ephysqc.validate_ttl_test(ses_path))

    def test_checklist_empty_folder(self):
        ses_path = self.init_folder / 'empty'
        with self.assertRaises(FileNotFoundError):
            ibllib.ephys.ephysqc.validate_ttl_test(ses_path)

import unittest
from pathlib import Path
from oneibl.one import ONE

one = ONE(
    base_url="https://test.alyx.internationalbrainlab.org",
    username="test_user",
    password="TapetesBloc18",
)


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Init connection to the database
        self.eids = [
            "cf264653-2deb-44cb-aa84-89b82507028a",
            "4e0b3320-47b7-416e-b842-c34dc9004cf8",
            "a9d89baf-9905-470c-8565-859ff212c7be",
            "aaf101c3-2581-450a-8abd-ddb8f557a5ad",
        ]
        self.partial_eid_paths = [
            Path('FlatIron/mainenlab/Subjects/clns0730/2018-08-24/002'),
            Path('FlatIron/zadorlab/Subjects/flowers/2018-07-13/001'),
            Path("FlatIron/mainenlab/Subjects/ZM_1743/2019-06-04/001"),
            Path("FlatIron/cortexlab/Subjects/KS005/2019-04-04/004"),
        ]
        self.det_keys = [
            "subject",
            "start_time",
            "number",
            "lab",
            "project",
            "url",
            "task_protocol",
            "local_path",
        ]
        self.full_det_keys = [
            "subject",
            "users",
            "location",
            "procedures",
            "lab",
            "project",
            "type",
            "task_protocol",
            "number",
            "start_time",
            "end_time",
            "narrative",
            "parent_session",
            "n_correct_trials",
            "n_trials",
            "url",
            "extended_qc",
            "qc",
            "wateradmin_session_related",
            "data_dataset_session_related",
            "json",
            "probe_insertion",
        ]

    def test_path_from_eid(self):
        # Test if eid's produce correct output
        for e, p in zip(self.eids, self.partial_eid_paths):
            self.assertTrue(str(p) in str(one.path_from_eid(e)))
        # Test if list input produces valid list output
        list_output = one.path_from_eid(self.eids)
        self.assertTrue(isinstance(list_output, list))
        self.assertTrue(
            all([str(p) in str(o) for p, o in zip(self.partial_eid_paths, list_output)])
        )

    def test_eid_from_path(self):
        # test if paths produce expected eid's
        paths = self.partial_eid_paths[-2:]
        paths.append("FlatIron/mainenlab/Subjects/ZM_1743/2019-06-04/001/bla.ble")
        paths.append(
            "some/other/root/FlatIron/cortexlab/" "Subjects/KS005/2019-04-04/004/bli/blo.blu"
        )
        eids = self.eids[-2:]
        eids.append("a9d89baf-9905-470c-8565-859ff212c7be")
        eids.append("aaf101c3-2581-450a-8abd-ddb8f557a5ad")
        for p, e in zip(paths, eids):
            self.assertTrue(e == str(one.eid_from_path(p)))
        # Test if list input produces correct list output
        list_output = one.eid_from_path(paths)
        self.assertTrue(isinstance(list_output, list))
        self.assertTrue(all([e == o for e, o in zip(eids, list_output)]))

    def test_get_details(self):
        # Get one known eid details dics
        det_from_eid = one.get_details(self.eids[0])
        full_det_from_eid = one.get_details(self.eids[0], full=True)
        # Test if all keys in returned dict are in expected keys
        # Full dict output
        self.assertTrue(
            all([x in full_det_from_eid for x in self.full_det_keys]),
            "Missing details key"
        )
        # Partial dict output
        self.assertTrue(
            all([x in det_from_eid for x in self.det_keys]),
            "Missing details key"
        )
        # Test local path append to details
        self.assertTrue(det_from_eid['local_path'] is not None)
        # Test is known local_path is not None
        self.assertTrue(one.get_details(self.eids[2])['local_path'] is not None)
        # Test list input
        self.assertTrue(len(self.eids) == len(one.get_details(self.eids)))


if __name__ == "__main__":
    unittest.main(exit=False)

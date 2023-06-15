import unittest
import tempfile
from pathlib import Path
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from one.api import ONE
from one.registration import RegistrationClient

from ibllib.pipes import base_tasks
from ibllib.tests import TEST_DB


class TestRegisterRawDataTask(unittest.TestCase):
    tmpdir = None
    one = None
    session_path = None
    eid = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.one = ONE(**TEST_DB, cache_rest=None)
        ses_dict = {
            'subject': 'algernon',
            'start_time': RegistrationClient.ensure_ISO8601(None),
            'number': 1,
            'users': ['test_user']}
        ses = cls.one.alyx.rest('sessions', 'create', data=ses_dict)
        cls.session_path = Path(cls.tmpdir.name).joinpath(
            ses['subject'], ses['start_time'][:10], str(ses['number']).zfill(3))
        cls.eid = ses['url'][-36:]

        # Add a couple of images
        cls.session_path.joinpath('snapshots').mkdir(parents=True)
        for ext in ('.PNG', '.tif'):
            plt.imshow(np.random.random((7, 7)))
            plt.savefig(cls.session_path.joinpath('snapshots', 'foo').with_suffix(ext))
            plt.close()

    def test_register_snapshots(self):
        """Test ibllib.pipes.base_tasks.RegisterRawDataTask.register_snapshots.

        A more thorough test for this exists in ibllib.tests.test_pipes.TestRegisterRawDataTask.
        This test does not mock REST (and therefore requires a test database), while the other does.
        This test could be removed as it's rather redundant.
        """
        task = base_tasks.RegisterRawDataTask(self.session_path, one=self.one)
        notes = task.register_snapshots()
        self.assertEqual(2, len(notes))
        self.assertTrue(self.session_path.joinpath('snapshots').exists())
        task.register_snapshots(unlink=True)
        self.assertFalse(self.session_path.joinpath('snapshots').exists())

    def test_rename_files(self):
        collection = 'raw_sync_data'
        task = base_tasks.RegisterRawDataTask(self.session_path, one=self.one)
        task.input_files = task.output_files = []
        task.rename_files()  # Returns without raising
        task.input_files = [('foo.*', collection, True), ]
        task.output_files = [('_ns_DAQdata.raw.bar', collection, True), ]
        self.session_path.joinpath(collection).mkdir()
        self.session_path.joinpath(collection, 'foo.bar').touch()
        task.rename_files()
        self.assertTrue(self.session_path.joinpath(collection, '_ns_DAQdata.raw.bar').exists())
        self.assertFalse(self.session_path.joinpath(collection, 'foo.bar').exists())
        with self.assertRaises(FileNotFoundError):
            task.rename_files()
        # Check asserts number of inputs == number of outputs
        task.output_files.append(('_ns_DAQdata.baz.bar', collection, True),)
        with self.assertRaises(AssertionError):
            task.rename_files()

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.tmpdir:
            cls.tmpdir.cleanup()
        if cls.one and cls.eid:
            cls.one.alyx.rest('sessions', 'delete', id=cls.eid)


class TestBehaviourTask(unittest.TestCase):
    def test_spacer_support(self) -> None:
        """Test for BehaviourTask._spacer_support method."""
        to_test = [('100.0.0', False), ('8.0.0', False), ('7.1.0', True), ('8.0.1', True), ('7.2.0', True)]
        settings = {}
        spacer_support = partial(base_tasks.BehaviourTask._spacer_support, settings)
        for version, expected in to_test:
            settings['IBLRIG_VERSION_TAG'] = version
            with self.subTest(version):
                self.assertIs(spacer_support(), expected)


if __name__ == '__main__':
    unittest.main()

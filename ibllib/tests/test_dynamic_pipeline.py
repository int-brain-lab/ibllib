import tempfile
from pathlib import Path
import unittest
from unittest import mock
from itertools import chain

import yaml

import ibllib.tests
import ibllib.pipes.dynamic_pipeline as dyn
from ibllib.pipes.tasks import Pipeline, Task
from ibllib.pipes import ephys_preprocessing
from ibllib.pipes import training_preprocessing
from ibllib.io import session_params
from ibllib.tests.fixtures.utils import populate_task_settings


def test_read_write_params_yaml():
    ad = dyn.get_acquisition_description('choice_world_recording')
    with tempfile.TemporaryDirectory() as td:
        session_path = Path(td)
        session_params.write_params(session_path, ad)
        add = session_params.read_params(session_path)
    assert ad == add


class TestCreateLegacyAcqusitionDescriptions(unittest.TestCase):

    def test_legacy_biased(self):
        session_path = Path(ibllib.tests.__file__).parent.joinpath('extractors', 'data', 'session_biased_ge5')
        ad = dyn.acquisition_description_legacy_session(session_path)
        protocols = list(chain(*map(dict.keys, ad.get('tasks', []))))
        self.assertCountEqual(['biasedChoiceWorld'], protocols)
        self.assertEqual(1, len(ad['devices']['cameras']))

    def test_legacy_ephys(self):
        session_path = Path(ibllib.tests.__file__).parent.joinpath('extractors', 'data', 'session_ephys')
        ad_ephys = dyn.acquisition_description_legacy_session(session_path)
        self.assertEqual(2, len(ad_ephys['devices']['neuropixel']))
        self.assertEqual(3, len(ad_ephys['devices']['cameras']))
        protocols = list(chain(*map(dict.keys, ad_ephys.get('tasks', []))))
        self.assertEqual(protocols, ['ephysChoiceWorld', 'passiveChoiceWorld'])

    def test_legacy_training(self):
        session_path = Path(ibllib.tests.__file__).parent.joinpath('extractors', 'data', 'session_training_ge5')
        ad = dyn.acquisition_description_legacy_session(session_path)
        protocols = list(chain(*map(dict.keys, ad.get('tasks', []))))
        self.assertCountEqual(['trainingChoiceWorld'], protocols)
        self.assertEqual(1, len(ad['devices']['cameras']))


class TestGetTrialsTasks(unittest.TestCase):
    """Test pipes.dynamic_pipeline.get_trials_tasks function."""

    def setUp(self):
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        # The github CI root dir contains an alias/symlink so we must resolve it
        self.tempdir = Path(tmpdir.name).resolve()
        self.session_path_dynamic = self.tempdir / 'subject' / '2023-01-01' / '001'
        self.session_path_dynamic.mkdir(parents=True)
        description = {'version': '1.0.0',
                       'sync': {'nidq': {'collection': 'raw_ephys_data', 'extension': 'bin', 'acquisition_software': 'spikeglx'}},
                       'tasks': [
                           {'ephysChoiceWorld': {'task_collection': 'raw_task_data_00'}},
                           {'passiveChoiceWorld': {'task_collection': 'raw_task_data_01'}},
                       ]}
        with open(self.session_path_dynamic / '_ibl_experiment.description.yaml', 'w') as fp:
            yaml.safe_dump(description, fp)

        self.session_path_legacy = self.session_path_dynamic.with_name('002')
        (collection := self.session_path_legacy.joinpath('raw_behavior_data')).mkdir(parents=True)
        self.settings = {'IBLRIG_VERSION': '7.2.2', 'PYBPOD_PROTOCOL': '_iblrig_tasks_ephysChoiceWorld'}
        self.settings_path = populate_task_settings(collection, self.settings)

    def test_get_trials_tasks(self):
        """Test pipes.dynamic_pipeline.get_trials_tasks function."""
        # A dynamic pipeline session
        tasks = dyn.get_trials_tasks(self.session_path_dynamic)
        self.assertEqual(2, len(tasks))
        self.assertEqual('raw_task_data_00', tasks[0].collection)

        # Check behaviour with ONE
        one = mock.MagicMock()
        one.offline = False
        one.alyx = mock.MagicMock()
        one.alyx.cache_mode = None  # sneaky hack as this is checked by the pipeline somewhere
        tasks = dyn.get_trials_tasks(self.session_path_dynamic, one)
        self.assertEqual(2, len(tasks))
        one.load_datasets.assert_called()  # check that description file is checked on disk

        # An ephys session
        tasks = dyn.get_trials_tasks(self.session_path_legacy)
        self.assertEqual(1, len(tasks))
        self.assertIsInstance(tasks[0], ephys_preprocessing.EphysTrials)

        # A training session
        self.settings['PYBPOD_PROTOCOL'] = '_iblrig_tasks_trainingChoiceWorld'
        populate_task_settings(self.settings_path, self.settings)

        tasks = dyn.get_trials_tasks(self.session_path_legacy, one=one)
        self.assertEqual(1, len(tasks))
        self.assertIsInstance(tasks[0], training_preprocessing.TrainingTrials)
        self.assertIs(tasks[0].one, one, 'failed to assign ONE instance to task')

        # A personal project
        self.settings['PYBPOD_PROTOCOL'] = '_misc_foobarChoiceWorld'
        populate_task_settings(self.settings_path, self.settings)

        m = mock.MagicMock()  # Mock the project_extractors repo
        m.base.__file__ = str(self.tempdir / 'base.py')
        # Create the personal project extractor types map
        task_type_map = {'_misc_foobarChoiceWorld': 'foobar'}
        extractor_types_path = Path(m.base.__file__).parent.joinpath('extractor_types.json')
        populate_task_settings(extractor_types_path, task_type_map)
        # Simulate the instantiation of the personal project module's pipeline class
        pipeline = mock.Mock(spec=Pipeline)
        pipeline.name = 'custom'
        task_mock = mock.Mock(spec=Task)
        pipeline.tasks = {'RegisterRaw': mock.MagicMock(), 'FooBarTrials': task_mock}
        m.base.get_pipeline().return_value = pipeline
        with mock.patch.dict('sys.modules', projects=m):
            """For unknown reasons this method of mocking the personal projects repo (which is
            imported within various functions) fails on the Github test builds. This we check
            here and skip the rest of the test if patch didn't work."""
            try:
                import projects.base
                assert isinstance(projects.base, mock.Mock)
            except (AssertionError, ModuleNotFoundError):
                self.skipTest('Failed to mock projects module import')
            tasks = dyn.get_trials_tasks(self.session_path_legacy)
            self.assertEqual(1, len(tasks))
            task_mock.assert_called_once_with(self.session_path_legacy)
            # Should handle absent trials tasks
            pipeline.tasks.pop('FooBarTrials')
            self.assertEqual([], dyn.get_trials_tasks(self.session_path_legacy))

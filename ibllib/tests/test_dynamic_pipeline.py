import tempfile
from pathlib import Path
import unittest
from itertools import chain

import ibllib.tests
from ibllib.pipes import dynamic_pipeline
from ibllib.io import session_params


def test_read_write_params_yaml():
    ad = dynamic_pipeline.get_acquisition_description('choice_world_recording')
    with tempfile.TemporaryDirectory() as td:
        session_path = Path(td)
        session_params.write_params(session_path, ad)
        add = session_params.read_params(session_path)
    assert ad == add


class TestCreateLegacyAcqusitionDescriptions(unittest.TestCase):

    def test_legacy_biased(self):
        session_path = Path(ibllib.tests.__file__).parent.joinpath('extractors', 'data', 'session_biased_ge5')
        ad = dynamic_pipeline.acquisition_description_legacy_session(session_path)
        protocols = list(chain(*map(dict.keys, ad.get('tasks', []))))
        self.assertCountEqual(['biasedChoiceWorld'], protocols)
        self.assertEqual(1, len(ad['devices']['cameras']))

    def test_legacy_ephys(self):
        session_path = Path(ibllib.tests.__file__).parent.joinpath('extractors', 'data', 'session_ephys')
        ad_ephys = dynamic_pipeline.acquisition_description_legacy_session(session_path)
        self.assertEqual(2, len(ad_ephys['devices']['neuropixel']))
        self.assertEqual(3, len(ad_ephys['devices']['cameras']))
        protocols = list(chain(*map(dict.keys, ad_ephys.get('tasks', []))))
        self.assertEqual(protocols, ['ephysChoiceWorld', 'passiveChoiceWorld'])

    def test_legacy_training(self):
        session_path = Path(ibllib.tests.__file__).parent.joinpath('extractors', 'data', 'session_training_ge5')
        ad = dynamic_pipeline.acquisition_description_legacy_session(session_path)
        protocols = list(chain(*map(dict.keys, ad.get('tasks', []))))
        self.assertCountEqual(['trainingChoiceWorld'], protocols)
        self.assertEqual(1, len(ad['devices']['cameras']))

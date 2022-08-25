import tempfile
import yaml
from pathlib import Path
import unittest

import ibllib.tests
from ibllib.pipes import dynamic_pipeline
from ibllib.io import session_params


def test_read_write_params_yaml():
    ad = dynamic_pipeline.get_acquisition_description('choice_world_recording')
    with tempfile.NamedTemporaryFile(mode='w+') as fid:
        yaml.dump(data=ad, stream=fid)
        add = session_params.read_params(fid.name)
    assert ad == add


class TestCreateLegacyAcqusitionDescriptions(unittest.TestCase):

    def test_legacy_biased(self):
        session_path = Path(ibllib.tests.__file__).parent.joinpath('extractors', 'data', 'session_biased_ge5')
        ad = dynamic_pipeline.acquisition_description_legacy_session(session_path)
        assert set(list(ad['tasks'].keys())) == set(['biasedChoiceWorld'])
        assert len(ad['devices']['cameras']) == 1

    def test_legacy_ephys(self):
        session_path = Path(ibllib.tests.__file__).parent.joinpath('extractors', 'data', 'session_ephys')
        ad_ephys = dynamic_pipeline.acquisition_description_legacy_session(session_path)
        assert len(ad_ephys['devices']['neuropixel']) == 2
        assert len(ad_ephys['devices']['cameras']) == 3
        assert set(list(ad_ephys['tasks'].keys())) == set(['ephysChoiceWorld', 'passiveChoiceWorld'])

    def test_legacy_training(self):
        session_path = Path(ibllib.tests.__file__).parent.joinpath('extractors', 'data', 'session_training_ge5')
        ad = dynamic_pipeline.acquisition_description_legacy_session(session_path)
        assert set(list(ad['tasks'].keys())) == set(['trainingChoiceWorld'])
        assert len(ad['devices']['cameras']) == 1

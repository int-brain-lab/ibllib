from ibllib.pipes.dynamic_pipeline import get_acquisition_description
from ibllib.io import session_params
import tempfile
import yaml


def test_read_write_params_yaml():
    ad = get_acquisition_description('choice_world_recording')
    with tempfile.NamedTemporaryFile(mode='w+') as fid:
        yaml.dump(data=ad, stream=fid)
        add = session_params.read_params(fid.name)
    assert ad == add

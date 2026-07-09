"""Test conversion of reference image MLAP to FOV MLAPDV coordinates.

TODO Add histology file locaion to MesoscopeFOVHistology arguments
"""

import unittest
import logging

from one.alf.path import ALFPath
from one.api import ONE
import numpy as np

import ibllib.oneibl.data_handlers as dh

logger = logging.getLogger('ibllib')


class TestRefSession(unittest.TestCase):
    """Test extraction of FOV coordinates for aligned reference session."""

    def setUp(self):
        self.session_path = ALFPath(r'D:\Flatiron\alyx.internationalbrainlab.org\cortexlab\Subjects\SP037\2024-08-01\001')
        self.reference_session = '839bb5b1-120f-49d0-b7c9-5174c0c66b5a'  # SP037/2023-02-20/001
        self.one = ONE()

    def test_ref_session(self):
        # 1. Download the MLAPDV coordinates of the reference session
        registered_mlapdv = self.get_atlas_registered_reference_mlap(self.reference_session)
        ref_mlapdv = np.load(registered_mlapdv)
        # 2. Load reference image shape
        reference_image_path = self.session_path / 'raw_imaging_data_02' / 'reference' / 'referenceImage.stack.tif'
    
    def get_atlas_registered_reference_mlap(self, reference_eid, clobber=False, client_name='server'):
        """Download the MLAPDV coordinates of the reference session.
        
        This is the file created by the histology pipeline, one per subject.
        """
        reference_session_path = self.one.eid2path(reference_eid)
        assert reference_session_path.subject == self.session_path.subject
        local_file = self.session_path.parents[3] / reference_session_path.relative_to_lab() / 'histology' / 'registered_mlapdv.npy'
        if clobber or not local_file.exists():
            # Download remote file
            # TODO use task data handler here
            # TODO Support http download
            local_file.parent.mkdir(parents=True, exist_ok=True)
            try:  # if isinstance(self.data_handler, dh.ServerGlobusDataHandler):
                handler = dh.ServerGlobusDataHandler(reference_session_path, {'input_files': [], 'output_files': []}, one=self.one)
                endpoint_id = next(v['id'] for k, v in handler.globus.endpoints.items() if k.startwith('flatiron'))
                handler.globus.add_endpoint(endpoint_id, label='flatiron_histology', root_path='/histology/')
                remote_file = f'{reference_session_path.lab}/{reference_session_path.session_path_short()}/{local_file.name}'
                handler.globus.mv('flatiron_histology', 'local', [remote_file], ['/'.join(local_file.parts[-5:])])
                assert local_file.exists(), f'Failed to download {remote_file} to {local_file}'
            except Exception as e:
                logger.error(f'Failed to download via Globus: {e}')
                import tempfile, shutil
                with tempfile.TemporaryDirectory() as tmpdir:
                    remote_file = f'{self.one.alyx._par.HTTP_DATA_SERVER}/histology/{reference_session_path.lab}/{reference_session_path.subject}/registered_mlapdv.npy'
                    logger.warning(f'Using HTTP download for {remote_file}')
                    file = self.one.alyx.download_file(remote_file, target_dir=tmpdir)
                    shutil.move(file, local_file)
        return local_file
            

if __name__ == '__main__':
    unittest.main()
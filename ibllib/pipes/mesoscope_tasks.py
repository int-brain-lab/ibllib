"""The mesoscope data extraction pipeline.

The mesoscope pipeline currently comprises raw image registration and timestamps extraction.  In
the future there will be compression (and potential cropping), FOV metadata extraction, and ROI
extraction.

Pipeline:
    1. Data renamed to be ALF-compliant and registered
"""
import logging
import subprocess
import json
import shutil
from pathlib import Path
from itertools import chain
import numpy as np

from ibllib.pipes import base_tasks

_logger = logging.getLogger(__name__)


class MesoscopeRegisterSnapshots(base_tasks.RegisterRawDataTask):

    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('*.tif', f'{self.device_collection}/snapshots', False),
                            ('*.png', f'{self.device_collection}/snapshots', False),
                            ('*.jpg', f'{self.device_collection}/snapshots', False)],
            'output_files': []
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data'))

    def _run(self):
        out_files = super()._run()
        self.register_snapshots()
        return out_files


class MesoscopeCompress(base_tasks.DynamicTask):

    priority = 90
    job_size = 'large'

    @property
    def signature(self):
        signature = {
            'input_files': [('*.tif', self.device_collection, True)],
            'output_files': [('imaging.frames.tar.bz', self.device_collection, True)]
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data'))

    def _run(self, remove_uncompressed=False, verify_output=True, **kwargs):
        in_dir = self.session_path.joinpath(self.device_collection or '')
        outfile = self.session_path.joinpath(*filter(None, reversed(self.output_files[0][:2])))
        infiles = list(chain(*map(lambda x: in_dir.glob(x[0]), self.input_files)))  # glob for all input patterns
        if not infiles:
            _logger.info('No image files found; returning')
            return []

        uncompressed_size = sum(x.stat().st_size for x in infiles)
        _logger.info('Compressing %i files', len(infiles))
        cmd = 'tar -I lbzip2 -cvf "{output}" "{input}"'.format(
            output=outfile.relative_to(in_dir), input='" "'.join(str(x.relative_to(in_dir)) for x in infiles))
        _logger.debug(cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=in_dir)
        info, error = process.communicate()
        assert process.returncode == 0, f'compression failed: {error}'

        # Check the output
        assert outfile.exists(), 'output file missing'
        compressed_size = outfile.stat().st_size
        _logger.info('Compression ratio = %.3f, saving %.2f pct (%.2f MB)',
                     uncompressed_size / compressed_size,
                     round((1 - (compressed_size / uncompressed_size)) * 10000) / 100,
                     (uncompressed_size - compressed_size) / 1024 / 1024)

        if verify_output:
            cmd = f'tar -tzf "{outfile.relative_to(in_dir)}"'
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=in_dir)
            info, error = process.communicate()
            assert process.returncode == 0, f'failed to read compressed file: {error}'
            # TODO Assert number of files in tar match input files

        if remove_uncompressed:
            for file in infiles:
                file.unlink()

        return [outfile]


class MesoscopePreprocess(base_tasks.DynamicTask):

    priority = 80
    job_size = 'large'

    def __init__(self, session_path, **kwargs):
        super(MesoscopePreprocess, self).__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data'))
        # TODO: make sure that we are happy with these defaults
        # TODO: decide if we want to code them here or in the construction of the meta json
        self.db = {
            'data_path': [str(self.session_path.joinpath(self.device_collection))],
            'save_path0': str(self.session_path.joinpath('alf')),  # is also used as fast_disk unless thats defined
            'move_bin': True,
            'keep_movie_raw': False,
            'delete_bin': False,
            'batch_size': 1000,
            'combined': True,
        }

    # TODO: write function to get output file list (depend on field of views)
    @property
    def signature(self):
        signature = {
            'input_files': [('rawImagingData.meta.json', self.device_collection, True)],
            'output_files': []
        }
        return signature

    def _rename_outputs(self, rename_dict=None):
        if rename_dict is None:
            rename_dict = {
                'F.npy': 'mpci.ROIActivityF.npy',
                'Fneu.npy': 'mpci.ROIActivityFneu.npy',
                'spks.npy': 'mpci.ROIActivityDeconvolved.npy',
                'iscell.npy': 'mpciROIs.included.npy'
            }
        # Rename the outputs, first the subdirectories
        suite2p_dir = Path(self.db['save_path0']).joinpath('suite2p')
        for plane_dir in suite2p_dir.iterdir():
            if plane_dir.name == 'combined':
                # TODO: is this renaming what we want?
                plane_dir.rename(plane_dir.parent.joinpath('fov_combined'))
            else:
                n = int(plane_dir.name.split('plane')[1])
                plane_dir.rename(plane_dir.parent.joinpath(f'fov{n:02}'))
        # Now rename the content of the new directories and move them out of suite2p
        for fov_dir in suite2p_dir.iterdir():
            for k in rename_dict.keys():
                try:
                    fov_dir.joinpath(k).rename(fov_dir.joinpath(rename_dict[k]))
                except FileNotFoundError:
                    _logger.error(f"Output file {k} expected but not found in {fov_dir}")
                    self.status = -1
            # extract bad frames from ops.npy file and save separately
            badframes = np.load(fov_dir.joinpath('ops.npy'), allow_pickle=True).item()['badframes']
            np.save(fov_dir.joinpath('mpci.validFrames.npy'), badframes)
            shutil.move(fov_dir, suite2p_dir.parent.joinpath(fov_dir.name))
        # TODO: what about ops.npy, stats.npy?
        # TODO: on the long run remove data.bin
        # Remove empty suite2p folder
        suite2p_dir.rmdir()
        # Collect all files in those directories
        return list(suite2p_dir.parent.rglob('fov*/*'))

    def _run(self, run_suite2p=True, rename_files=True, **kwargs):
        import suite2p
        # Get default ops
        ops = suite2p.default_ops()
        # Some options we get from the meta data json, we put them in db, which overwrites ops if the keys are the same
        # TODO: get the right path here
        with open(self.session_path.joinpath(self.device_collection, 'rawImagingData.meta.json'), 'r') as meta_file:
            meta = json.load(meta_file)
        # Inputs extracted from imaging data to a json
        # TODO: check that these are the right and complete inputs from the meta file
        for k in ['nrois', 'mesoscan', 'nplanes', 'nchannels', 'tau', 'fs', 'dx', 'dy', 'lines']:
            if k in meta.keys():
                self.db[k] = meta[k]
            else:
                _logger.warning(f'Setting for {k} not found in metadata file. Keeping default.')
        # Anything can be overwritten by keyword arguments passed to the tasks run() method
        for k, v in kwargs.items():
            if k in ops.keys() or k in self.db.keys():
                # db overwrites ops when passed to run_s2p, so we only need to update / add it here
                self.db[k] = v
        # Update the task kwargs attribute as it will be stored in the arguments json field in alyx
        self.kwargs = {**self.kwargs, **self.db}
        # Run suite2p
        if run_suite2p:
            _ = suite2p.run_s2p(ops=ops, db=self.db)
        # Rename files and return outputs
        if rename_files:
            out_files = self._rename_outputs()
        else:
            out_files = list(Path(self.db['save_path0']).joinpath('suite2p').rglob('*'))

        return out_files


class MesoscopeSync(base_tasks.DynamicTask):

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [(f'_{self.sync_namespace}_sync.channels.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.polarities.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.times.npy', self.sync_collection, True)],
            'output_files': [('imaging.times.npy', 'alf/mesoscope', True), ]
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data'))

    def _run(self):
        raise NotImplementedError
        # TODO QC

        return  # out_files


class MesoscopeFOV(base_tasks.DynamicTask):

    priority = 40
    job_size = 'small'

    signature = {
        'input_files': [('mesoscopeLandmarks.dorsalCortex.json', 'alf', True),
                        ('mesoscopeSVT.uncorrected.npy', 'alf', True),
                        ('mesoscopeSVT.haemoCorrected.npy', 'alf', True)],
        'output_files': []
    }

    def _run(self):
        # TODO make task that computes location

        return []

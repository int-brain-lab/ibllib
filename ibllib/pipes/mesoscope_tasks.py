"""The mesoscope data extraction pipeline.

The mesoscope pipeline currently comprises raw image registration and timestamps extraction.  In
the future there will be compression (and potential cropping), FOV metadata extraction, and ROI
extraction.

Pipeline:
    1. Data renamed to be ALF-compliant and registered
"""
import logging
import subprocess
import shutil
from pathlib import Path
from itertools import chain

import numpy as np
import one.alf.io as alfio
import one.alf.exceptions as alferr

from ibllib.pipes import base_tasks
from ibllib.io.extractors import mesoscope

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
    cpu = 4  # TODO: see if this works on the local servers or blows the RAM
    job_size = 'large'

    def __init__(self, session_path, **kwargs):
        super(MesoscopePreprocess, self).__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data_*'))
        self.db = {
            'data_path': [str(s) for s in self.session_path.glob(f'{self.device_collection}')],
            'save_path0': str(self.session_path.joinpath('alf')),  # is also used as fast_disk unless that's defined
            'look_one_level_down': False,  # don't look in the children folders as that is where the reference data is
            'num_workers': self.cpu,  # this selects number of cores to parallelize over for the registration step
            'num_workers_roi': -1,  # for parallelization over FOVs during cell detection, for now don't
            'keep_movie_raw': True,
            'delete_bin': False,
            'batch_size': 500,  # SP reduced this from 1000
            'nimg_init': 400,
            'tau': 1.5,  # 1.5 is recommended for GCaMP6s TODO: potential deduct the GCamp used from Alyx mouse line?
            'combined': True,
            'nonrigid': True,
            'maxregshift': 0.05,  # default = 1
            'denoise': 1,  # whether or not binned movie should be denoised before cell detection
            'block_size': [128, 128],
            'save_mat': True,  # save the data to Fall.mat
            'move_bin': True,  # move the binary file to save_path
            'scalefactor': 1,  # OPTIONAL: scale manually in x to account for overlap between adjacent ribbons in UCL mesoscope
        }
        self.from_meta = [
            'nrois',
            'mesoscan',
            'nplanes',
            'nchannels',
            'tau',
            'fs',
            'dx',
            'dy',
            'lines'
            'functional_chan',
            'align_by_chan'
        ]

    # TODO: write function to get in and output file list (depend on FOV numbers)
    @property
    def signature(self):
        signature = {
            'input_files': [('_ibl_rawImagingData.meta.json', self.device_collection, True)],
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
        rawImagingData = alfio.load_object(self.session_path / self.device_collection, 'rawImagingData')
        # Inputs extracted from imaging data to a json
        for k in self.from_meta:
            if k in rawImagingData['meta'].keys():
                self.db[k] = rawImagingData['meta'][k]
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
    """Extract the frame times from the main DAQ."""

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [(f'_{self.sync_namespace}_DAQData.raw.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_DAQData.timestamps.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_DAQData.meta.json', self.sync_collection, True),
                            ('_ibl_rawImagingData.meta.json', self.device_collection, True),
                            ('rawImagingData.times_scanImage.npy', self.device_collection, True),],
            'output_files': [('mpci.times.npy', 'alf/mesoscope/FOV*', True),
                             ('mpciStack.timeshift.npy', 'alf/mesoscope/FOV*', True), ]
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data'))

    def _run(self):
        self.rawImagingData = alfio.load_object(self.session_path / self.device_collection, 'rawImagingData')
        n_ROIs = len(self.rawImagingData['meta']['FOV'])
        mesosync = mesoscope.MesoscopeSyncTimeline(self.session_path, n_ROIs)
        sync, chmap = self.load_sync()  # Extract sync data from raw DAQ data
        mesosync.extract(save=True, sync=sync, chmap=chmap, device_collection=self.device_collection)

    def load_sync(self):
        """
        Load the sync and channel map.

        This method may be expanded to support other raw DAQ data formats.

        Returns
        -------
        one.alf.io.AlfBunch
            A dictionary with keys ('times', 'polarities', 'channels'), containing the sync pulses and
            the corresponding channel numbers.
        dict
            A map of channel names and their corresponding indices.
        """
        ns = self.get_sync_namespace()
        alf_path = self.session_path / self.sync_collection
        try:
            sync = alfio.load_object(alf_path, 'sync', namespace=ns)
            chmap = None
        except alferr.ALFObjectNotFound:
            if self.get_sync_namespace() == 'timeline':
                # Load the sync and channel map from the raw DAQ data
                timeline = alfio.load_object(alf_path, 'DAQData', namespace=ns)
                sync, chmap = mesoscope.timeline2sync(timeline)
            else:
                raise NotImplementedError
        return sync, chmap


class MesoscopeFOV(base_tasks.DynamicTask):

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('_ibl_rawImagingData.meta.json', self.device_collection, True)],
            'output_files': []
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope', kwargs.get('device_collection', 'raw_mesoscope_data'))

    def _run(self):
        """
        Returns
        -------
        dict
            The newly created FOV Alyx record.
        list
            The newly created FOV location Alyx records.

        Notes
        -----
        TODO move out of run method for convenience
        TODO Deal with already created FOVs

        """
        FACTOR = 1e3  # The meta data are in mm, while the FOV in alyx is in um
        dry = self.one is None or self.one.offline
        (filename, collection, _), = self.signature['input_files']
        meta = alfio.load_file_content(self.session_path / collection / filename) or {}

        alyx_FOV = {
            'session': self.session_path if dry else self.path2eid(),
            'type': 'mesoscope'
        }
        if dry:
            print(alyx_FOV)
        else:
            alyx_FOV = self.one.alyx.rest('FOV', 'create', data=alyx_FOV)

        locations = []
        for fov in meta.get('FOV', []):
            data = {'field_of_view': alyx_FOV.get('id'), 'provenance': 'Landmark'}
            # TODO Get z values
            x1, y1 = map(lambda x: float(x) * FACTOR, fov['topLeftMM'])
            x2, y2 = map(lambda x: float(x) * FACTOR, fov['topLeftMM'])
            # TODO Brain region estimate
            if dry:
                print(data)
                locations.append(data)
                continue
            locations.append(self.one.alyx.rest('FOVLocation', 'create', data=data))
        return alyx_FOV, locations

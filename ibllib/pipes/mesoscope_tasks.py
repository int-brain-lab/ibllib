"""The mesoscope data extraction pipeline.

The mesoscope pipeline currently comprises raw image registration and timestamps extraction.  In
the future there will be compression (and potential cropping), FOV metadata extraction, and ROI
extraction.

Pipeline:
    1. Register reference images and upload snapshots and notes to Alyx
    2. Run ROI cell detection
    3. Calculate the pixel and ROI brain locations and register fields of view to Alyx
    4. Compress the raw imaging data
    5. Extract the imaging times from the main DAQ
"""
import json
import logging
import subprocess
import shutil
import uuid
from pathlib import Path
from itertools import chain
from collections import defaultdict, Counter
from fnmatch import fnmatch
import enum
import re
import time

import numba as nb
import numpy as np
import pandas as pd
import sparse
from scipy.io import loadmat
from scipy.interpolate import interpn
import one.alf.io as alfio
from one.alf.spec import is_valid, to_alf
from one.alf.files import filename_parts, session_path_parts
import one.alf.exceptions as alferr

from ibllib.pipes import base_tasks
from ibllib.io.extractors import mesoscope
from iblatlas.atlas import ALLEN_CCF_LANDMARKS_MLAPDV_UM, MRITorontoAtlas


_logger = logging.getLogger(__name__)
Provenance = enum.Enum('Provenance', ['ESTIMATE', 'FUNCTIONAL', 'LANDMARK', 'HISTOLOGY'])  # py3.11 make StrEnum


class MesoscopeRegisterSnapshots(base_tasks.MesoscopeTask, base_tasks.RegisterRawDataTask):
    """Upload snapshots as Alyx notes and register the 2P reference image(s)."""
    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('referenceImage.raw.tif', f'{self.device_collection}/reference', False),
                            ('referenceImage.stack.tif', f'{self.device_collection}/reference', False),
                            ('referenceImage.meta.json', f'{self.device_collection}/reference', False)],
            'output_files': [('referenceImage.raw.tif', f'{self.device_collection}/reference', False),
                             ('referenceImage.stack.tif', f'{self.device_collection}/reference', False),
                             ('referenceImage.meta.json', f'{self.device_collection}/reference', False)]
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope',
                                                            kwargs.get('device_collection', 'raw_imaging_data_*'))

    def _run(self):
        """
        Assert one reference image per collection and rename it. Register snapshots.

        Returns
        -------
        list of pathlib.Path containing renamed reference image.
        """
        # Assert that only one tif file exists per collection
        file, collection, _ = self.signature['input_files'][0]
        reference_images = list(self.session_path.rglob(f'{collection}/{file}'))
        assert len(set(x.parent for x in reference_images)) == len(reference_images)
        # Rename the reference images
        out_files = super()._run()
        # Register snapshots in base session folder and raw_imaging_data folders
        self.register_snapshots(collection=[self.device_collection, ''])
        return out_files


class MesoscopeCompress(base_tasks.MesoscopeTask):
    """ Tar compress raw 2p tif files, optionally remove uncompressed data."""

    priority = 90
    job_size = 'large'
    _log_level = None

    @property
    def signature(self):
        signature = {
            'input_files': [('*.tif', self.device_collection, True)],
            'output_files': [('imaging.frames.tar.bz2', self.device_collection, True)]
        }
        return signature

    def setUp(self, **kwargs):
        """Run at higher log level"""
        self._log_level = _logger.level
        _logger.setLevel(logging.DEBUG)
        return super().setUp(**kwargs)

    def tearDown(self):
        _logger.setLevel(self._log_level or logging.INFO)
        return super().tearDown()

    def _run(self, remove_uncompressed=False, verify_output=True, clobber=False, **kwargs):
        """
        Run tar compression on all tif files in the device collection.

        Parameters
        ----------
        remove_uncompressed: bool
            Whether to remove the original, uncompressed data. Default is False.
        verify_output: bool
            Whether to check that the compressed tar file can be uncompressed without errors.
            Default is True.

        Returns
        -------
        list of pathlib.Path
            Path to compressed tar file.
        """
        outfiles = []  # should be one per raw_imaging_data folder
        input_files = defaultdict(list)
        for file, in_dir, _ in self.input_files:
            input_files[self.session_path.joinpath(in_dir)].append(file)

        for in_dir, files in input_files.items():
            outfile = in_dir / self.output_files[0][0]
            if outfile.exists() and not clobber:
                _logger.info('%s already exists; skipping...', outfile.relative_to(self.session_path))
                continue
            # glob for all input patterns
            infiles = list(chain(*map(lambda x: in_dir.glob(x), files)))
            if not infiles:
                _logger.info('No image files found in %s', in_dir.relative_to(self.session_path))
                continue

            _logger.debug(
                'Input files:\n\t%s', '\n\t'.join(map(Path.as_posix, (x.relative_to(self.session_path) for x in infiles)))
            )

            uncompressed_size = sum(x.stat().st_size for x in infiles)
            _logger.info('Compressing %i file(s)', len(infiles))
            cmd = 'tar -cjvf "{output}" "{input}"'.format(
                output=outfile.relative_to(in_dir), input='" "'.join(str(x.relative_to(in_dir)) for x in infiles))
            _logger.debug(cmd)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=in_dir)
            info, error = process.communicate()  # b'2023-02-17_2_test_2P_00001_00001.tif\n'
            _logger.debug(info.decode())
            assert process.returncode == 0, f'compression failed: {error.decode()}'

            # Check the output
            assert outfile.exists(), 'output file missing'
            outfiles.append(outfile)
            compressed_size = outfile.stat().st_size
            min_size = kwargs.pop('verify_min_size', 1024)
            assert compressed_size > int(min_size), f'Compressed file < {min_size / 1024:.0f}KB'
            _logger.info('Compression ratio = %.3f, saving %.2f pct (%.2f MB)',
                         uncompressed_size / compressed_size,
                         round((1 - (compressed_size / uncompressed_size)) * 10000) / 100,
                         (uncompressed_size - compressed_size) / 1024 / 1024)

            if verify_output:
                # Test bzip
                cmd = f'bzip2 -tv {outfile.relative_to(in_dir)}'
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=in_dir)
                info, error = process.communicate()
                _logger.debug(info.decode())
                assert process.returncode == 0, f'bzip compression test failed: {error}'
                # Check tar
                cmd = f'bunzip2 -dc {outfile.relative_to(in_dir)} | tar -tvf -'
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=in_dir)
                info, error = process.communicate()
                _logger.debug(info.decode())
                assert process.returncode == 0, 'tarball decompression test failed'
                compressed_files = set(x.split()[-1] for x in filter(None, info.decode().split('\n')))
                assert compressed_files == set(x.name for x in infiles)

            if remove_uncompressed:
                _logger.info(f'Removing input files for {in_dir.relative_to(self.session_path)}')
                for file in infiles:
                    file.unlink()

        return outfiles


class MesoscopePreprocess(base_tasks.MesoscopeTask):
    """Run suite2p preprocessing on tif files"""

    priority = 80
    cpu = -1
    job_size = 'large'

    @property
    def signature(self):
        # The number of in and outputs will be dependent on the number of input raw imaging folders and output FOVs
        signature = {
            'input_files': [('_ibl_rawImagingData.meta.json', self.device_collection, True),
                            ('*.tif', self.device_collection, True),
                            ('exptQC.mat', self.device_collection, False)],
            'output_files': [('mpci.ROIActivityF.npy', 'alf/FOV*', True),
                             ('mpci.ROINeuropilActivityF.npy', 'alf/FOV*', True),
                             ('mpci.ROIActivityDeconvolved.npy', 'alf/FOV*', True),
                             ('mpci.badFrames.npy', 'alf/FOV*', True),
                             ('mpci.mpciFrameQC.npy', 'alf/FOV*', True),
                             ('mpciFrameQC.names.tsv', 'alf/FOV*', True),
                             ('mpciMeanImage.images.npy', 'alf/FOV*', True),
                             ('mpciROIs.stackPos.npy', 'alf/FOV*', True),
                             ('mpciROIs.mpciROITypes.npy', 'alf/FOV*', True),
                             ('mpciROIs.cellClassifier.npy', 'alf/FOV*', True),
                             ('mpciROIs.uuids.csv', 'alf/FOV*', True),
                             ('mpciROITypes.names.tsv', 'alf/FOV*', True),
                             ('mpciROIs.masks.npy', 'alf/FOV*', True),
                             ('mpciROIs.neuropilMasks.npy', 'alf/FOV*', True),
                             ('_suite2p_ROIData.raw.zip', self.device_collection, False)]
        }
        return signature

    @staticmethod
    def _masks2sparse(stat, ops):
        """
        Extract 3D sparse mask arrays from suit2p output.

        Parameters
        ----------
        stat : numpy.array
            The loaded stat.npy file. A structured array with fields ('lam', 'ypix', 'xpix', 'neuropil_mask').
        ops : numpy.array
            The loaded ops.npy file. A structured array with fields ('Ly', 'Lx').

        Returns
        -------
        sparse.GCXS
            A pydata sparse array of type float32, representing the ROI masks.
        sparse.GCXS
            A pydata sparse array of type float32, representing the neuropil ROI masks.

        Notes
        -----
        Save using sparse.save_npz.
        """
        shape = (stat.shape[0], ops['Ly'], ops['Lx'])
        npx = np.prod(shape[1:])  # Number of pixels per time point
        coords = [[], [], []]
        data = []
        pil_coords = []
        for i, s in enumerate(stat):
            coords[0].append(np.full(s['ypix'].shape, i))
            coords[1].append(s['ypix'])
            coords[2].append(s['xpix'])
            data.append(s['lam'])
            pil_coords.append(s['neuropil_mask'] + i * npx)
        roi_mask_sp = sparse.COO(list(map(np.concatenate, coords)), np.concatenate(data), shape=shape)
        pil_mask_sp = sparse.COO(np.unravel_index(np.concatenate(pil_coords), shape), True, shape=shape)
        return sparse.GCXS.from_coo(roi_mask_sp), sparse.GCXS.from_coo(pil_mask_sp)

    def _rename_outputs(self, suite2p_dir, frameQC_names, frameQC, rename_dict=None):
        """
        Convert suite2p output files to ALF datasets.

        Parameters
        ----------
        suite2p_dir : pathlib.Path
        rename_dict : dict or None
            The suite2p output filenames and the corresponding ALF name. NB: These files are saved
            after transposition. Default is None, i.e. using the default mapping hardcoded in the function below.

        Returns
        -------
        list of pathlib.Path
            All paths found in FOV folders.
        """
        if rename_dict is None:
            rename_dict = {
                'F.npy': 'mpci.ROIActivityF.npy',
                'spks.npy': 'mpci.ROIActivityDeconvolved.npy',
                'Fneu.npy': 'mpci.ROINeuropilActivityF.npy'
            }
        # Rename the outputs, first the subdirectories
        for plane_dir in suite2p_dir.iterdir():
            # ignore the combined dir
            if plane_dir.name != 'combined':
                n = int(plane_dir.name.split('plane')[1])
                fov_dir = plane_dir.parent.joinpath(f'FOV_{n:02}')
                if fov_dir.exists():
                    shutil.rmtree(str(fov_dir), ignore_errors=False, onerror=None)
                plane_dir.rename(fov_dir)
        # Now rename the content of the new directories and move them out of suite2p
        for fov_dir in suite2p_dir.iterdir():
            # Compress suite2p output files
            target = suite2p_dir.parent.joinpath(fov_dir.name)
            target.mkdir(exist_ok=True)
            # Move bin file out of the way first
            if fov_dir.joinpath('data.bin').exists():
                dst = self.session_path.joinpath('raw_bin_files', fov_dir.name, 'data.bin')
                dst.parent.mkdir(parents=True, exist_ok=True)
                _logger.debug('Moving bin file to %s', dst.relative_to(self.session_path))
                fov_dir.joinpath('data.bin').replace(dst)
            # Set logger to warning for the moment to not clutter the logs
            prev_level = _logger.level
            _logger.setLevel(logging.WARNING)
            shutil.make_archive(str(target / '_suite2p_ROIData.raw'), 'zip', fov_dir, logger=_logger)
            _logger.setLevel(prev_level)
            if fov_dir != 'combined':
                # save frameQC in each dir (for now, maybe there will be fov specific frame QC eventually)
                if frameQC is not None and len(frameQC) > 0:
                    np.save(fov_dir.joinpath('mpci.mpciFrameQC.npy'), frameQC)
                    frameQC_names.to_csv(fov_dir.joinpath('mpciFrameQC.names.tsv'), sep='\t', index=False)

                # extract some other data from suite2p outputs
                ops = np.load(fov_dir.joinpath('ops.npy'), allow_pickle=True).item()
                stat = np.load(fov_dir.joinpath('stat.npy'), allow_pickle=True)
                iscell = np.load(fov_dir.joinpath('iscell.npy'))

                # Save suite2p ROI activity outputs in transposed from (n_frames, n_ROI)
                for k, v in rename_dict.items():
                    np.save(fov_dir.joinpath(v), np.load(fov_dir.joinpath(k)).T)
                    # fov_dir.joinpath(k).unlink()  # Keep original files for suite2P GUI
                np.save(fov_dir.joinpath('mpci.badFrames.npy'), np.asarray(ops['badframes'], dtype=bool))
                np.save(fov_dir.joinpath('mpciMeanImage.images.npy'), np.asarray(ops['meanImg'], dtype=float))
                np.save(fov_dir.joinpath('mpciROIs.stackPos.npy'), np.asarray([(*s['med'], 0) for s in stat], dtype=int))
                np.save(fov_dir.joinpath('mpciROIs.cellClassifier.npy'), np.asarray(iscell[:, 1], dtype=float))
                np.save(fov_dir.joinpath('mpciROIs.mpciROITypes.npy'), np.asarray(iscell[:, 0], dtype=np.int16))
                # clusters uuids
                uuid_list = ['uuids'] + list(map(str, [uuid.uuid4() for _ in range(len(iscell))]))
                with open(fov_dir.joinpath('mpciROIs.uuids.csv'), 'w+') as fid:
                    fid.write('\n'.join(uuid_list))

                pd.DataFrame([(0, 'no cell'), (1, 'cell')], columns=['roi_values', 'roi_labels']
                             ).to_csv(fov_dir.joinpath('mpciROITypes.names.tsv'), sep='\t', index=False)
                # ROI and neuropil masks
                roi_mask, pil_mask = self._masks2sparse(stat, ops)
                with open(fov_dir.joinpath('mpciROIs.masks.sparse_npz'), 'wb') as fp:
                    sparse.save_npz(fp, roi_mask)
                with open(fov_dir.joinpath('mpciROIs.neuropilMasks.sparse_npz'), 'wb') as fp:
                    sparse.save_npz(fp, pil_mask)
                # move folders out of suite2p dir
                # We overwrite existing files
                for file in filter(lambda x: is_valid(x.name), fov_dir.iterdir()):
                    target_file = target.joinpath(file.name)
                    if target_file.exists():
                        target_file.unlink()
                    file.rename(target_file)
        shutil.rmtree(str(suite2p_dir), ignore_errors=False, onerror=None)
        # Collect all files in those directories
        return list(suite2p_dir.parent.rglob('FOV*/*'))

    @staticmethod
    def _check_meta_data(meta_data_all: list) -> dict:
        """
        Check that the meta data is consistent across all raw imaging folders.

        Parameters
        ----------
        meta_data_all: list of dicts
            List of metadata dictionaries to be checked for consistency.

        Returns
        -------
        dict
            Single, consolidated dictionary containing metadata.
        """
        # Ignore the things we don't expect to match
        ignore = ('acquisitionStartTime', 'nFrames')
        ignore_sub = {'rawScanImageMeta': ('ImageDescription', 'Software')}

        def equal_dicts(a, b, skip=None):
            ka = set(a).difference(skip or ())
            kb = set(b).difference(skip or ())
            return ka == kb and all(a[key] == b[key] for key in ka)

        # Compare each dict with the first one in the list
        for i, meta in enumerate(meta_data_all[1:]):
            if meta != meta_data_all[0]:  # compare entire object first
                for k, v in meta_data_all[0].items():  # check key by key
                    if not (equal_dicts(v, meta[k], ignore_sub[k])  # compare sub-dicts...
                            if k in ignore_sub else  # ... if we have keys to ignore in test
                            not (k in ignore or v == meta[k])):
                        _logger.warning(f'Mismatch in meta data between raw_imaging_data folders for key {k}. '
                                        f'Using meta_data from first folder!')
        return meta_data_all[0]

    @staticmethod
    def _consolidate_exptQC(exptQC):
        """
        Consolidate exptQC.mat files into a single file.

        Parameters
        ----------
        exptQC : list of pandas.DataFrame
            The loaded 'exptQC.mat' files as squeezed and simplified data frames, with columns
            {'frameQC_frames', 'frameQC_names'}.

        Returns
        -------
        numpy.array
            An array of uint8 where 0 indicates good frames, and other values correspond to
            experimenter-defined issues (in 'qc_values' column of output data frame).
        pandas.DataFrame
            A data frame with columns {'qc_values', 'qc_labels'}, the former an unsigned int
            corresponding to a QC code; the latter a human-readable QC explanation.
        numpy.array
            An array of frame indices where QC code != 0.
        """

        # Merge and make sure same indexes have same names across all files
        frameQC_names_list = [e['frameQC_names'] for e in exptQC]
        frameQC_names_list = [{f: 0} if isinstance(f, str) else {f[i]: i for i in range(len(f))}
                              for f in frameQC_names_list]
        frameQC_names = {k: v for d in frameQC_names_list for k, v in d.items()}
        for d in frameQC_names_list:
            for k, v in d.items():
                if frameQC_names[k] != v:
                    raise IOError(f'exptQC.mat files have different values for name "{k}"')

        frameQC_names = pd.DataFrame(sorted([(v, k) for k, v in frameQC_names.items()]),
                                     columns=['qc_values', 'qc_labels'])

        # Concatenate frames
        frameQC = np.concatenate([e['frameQC_frames'] for e in exptQC], axis=0)

        # Transform to bad_frames as expected by suite2p
        bad_frames = np.where(frameQC != 0)[0]

        return frameQC, frameQC_names, bad_frames

    def get_default_tau(self):
        """
        Determine the tau (fluorescence decay) from the subject's genotype.

        Returns
        -------
        float
            The tau value to use.

        See Also
        --------
        https://suite2p.readthedocs.io/en/latest/settings.html
        """
        # These settings are from the suite2P documentation
        TAU_MAP = {'G6s': 1.5, 'G6m': 1., 'G6f': .7, 'default': 1.5}
        _, subject, *_ = session_path_parts(self.session_path)
        genotype = self.one.alyx.rest('subjects', 'read', id=subject)['genotype']
        match = next(filter(None, (re.match(r'.+-(G\d[fms])$', g['allele']) for g in genotype)), None)
        key = match.groups()[0] if match else 'default'
        return TAU_MAP.get(key, TAU_MAP['default'])

    def _create_db(self, meta):
        """
        Create the ops dictionary for suite2p based on metadata.

        Parameters
        ----------
        meta: dict
            Imaging metadata.

        Returns
        -------
        dict
            Inputs to suite2p run that deviate from default parameters.
        """

        # Computing dx and dy
        cXY = np.array([fov['Deg']['topLeft'] for fov in meta['FOV']])
        cXY -= np.min(cXY, axis=0)
        nXnYnZ = np.array([fov['nXnYnZ'] for fov in meta['FOV']])

        # Currently supporting z-stacks but not supporting dual plane / volumetric imaging, assert that this is not the case
        if np.any(nXnYnZ[:, 2] > 1):
            raise NotImplementedError('Dual-plane imaging not yet supported, data seems to more than one plane per FOV')

        sW = np.sqrt(np.sum((np.array([fov['Deg']['topRight'] for fov in meta['FOV']]) - np.array(
            [fov['Deg']['topLeft'] for fov in meta['FOV']])) ** 2, axis=1))
        sH = np.sqrt(np.sum((np.array([fov['Deg']['bottomLeft'] for fov in meta['FOV']]) - np.array(
            [fov['Deg']['topLeft'] for fov in meta['FOV']])) ** 2, axis=1))
        pixSizeX = nXnYnZ[:, 0] / sW
        pixSizeY = nXnYnZ[:, 1] / sH
        dx = np.round(cXY[:, 0] * pixSizeX).astype(dtype=np.int32)
        dy = np.round(cXY[:, 1] * pixSizeY).astype(dtype=np.int32)
        nchannels = len(meta['channelSaved']) if isinstance(meta['channelSaved'], list) else 1

        # Computing number of unique z-planes (slices in tiff)
        # FIXME this should work if all FOVs are discrete or if all FOVs are continuous, but may not work for combination of both
        slice_ids = [fov['slice_id'] for fov in meta['FOV']]
        nplanes = len(set(slice_ids))

        # Figuring out how many SI Rois we have (one unique ROI may have several FOVs)
        # FIXME currently unused
        # roiUUIDs = np.array([fov['roiUUID'] for fov in meta['FOV']])
        # nrois = len(np.unique(roiUUIDs))

        db = {
            'data_path': sorted(map(str, self.session_path.glob(f'{self.device_collection}'))),
            'save_path0': str(self.session_path.joinpath('alf')),
            'fast_disk': '',  # TODO
            'look_one_level_down': False,  # don't look in the children folders as that is where the reference data is
            'num_workers': self.cpu,  # this selects number of cores to parallelize over for the registration step
            'num_workers_roi': -1,  # for parallelization over FOVs during cell detection, for now don't
            'keep_movie_raw': False,
            'delete_bin': False,  # TODO: delete this on the long run
            'batch_size': 500,  # SP reduced this from 1000
            'nimg_init': 400,
            'combined': False,
            'nonrigid': True,
            'maxregshift': 0.05,  # default = 1
            'denoise': 1,  # whether binned movie should be denoised before cell detection
            'block_size': [128, 128],
            'save_mat': True,  # save the data to Fall.mat
            'move_bin': True,  # move the binary file to save_path
            'mesoscan': True,
            'nplanes': nplanes,
            'nrois': len(meta['FOV']),
            'nchannels': nchannels,
            'fs': meta['scanImageParams']['hRoiManager']['scanVolumeRate'],
            'lines': [list(np.asarray(fov['lineIdx']) - 1) for fov in meta['FOV']],  # subtracting 1 to make 0-based
            'slices': slice_ids,  # this tells us which FOV corresponds to which tiff slices
            'tau': self.get_default_tau(),  # deduce the GCamp used from Alyx mouse line (defaults to 1.5; that of GCaMP6s)
            'functional_chan': 1,  # for now, eventually find(ismember(meta.channelSaved == meta.channelID.green))
            'align_by_chan': 1,  # for now, eventually find(ismember(meta.channelSaved == meta.channelID.red))
            'dx': dx,
            'dy': dy
        }

        return db

    def _run(self, run_suite2p=True, rename_files=True, use_badframes=True, **kwargs):
        """
        Process inputs, run suite2p and make outputs alf compatible.

        Parameters
        ----------
        run_suite2p: bool
            Whether to run suite2p, default is True.
        rename_files: bool
            Whether to rename and reorganize the suite2p outputs to be alf compatible. Defaults is True.
        use_badframes: bool
            Whether to exclude bad frames indicated by the experimenter in exptQC.mat. Default is currently False
            due to bug in suite2p. Change this in the future.

        Returns
        -------
        list of pathlib.Path
            All files created by the task.
        """
        import suite2p

        """ Metadata and parameters """
        # Load metadata and make sure all metadata is consistent across FOVs
        meta_files = sorted(self.session_path.glob(f'{self.device_collection}/*rawImagingData.meta.*'))
        collections = set(f.parts[-2] for f in meta_files)
        # Check there is exactly 1 meta file per collection
        assert len(meta_files) == len(list(self.session_path.glob(self.device_collection))) == len(collections)
        rawImagingData = [mesoscope.patch_imaging_meta(alfio.load_file_content(filepath)) for filepath in meta_files]
        if len(rawImagingData) > 1:
            meta = self._check_meta_data(rawImagingData)
        else:
            meta = rawImagingData[0]
        # Get default ops
        ops = suite2p.default_ops()
        # Create db which overwrites ops when passed to suite2p, with information from meta data and hardcoded
        db = self._create_db(meta)
        # Anything can be overwritten by keyword arguments passed to the tasks run() method
        for k, v in kwargs.items():
            if k in ops.keys() or k in db.keys():
                # db overwrites ops when passed to run_s2p, so we only need to update / add it here
                db[k] = v
        # Update the task kwargs attribute as it will be stored in the arguments json field in alyx
        self.kwargs = {**self.kwargs, **db}

        """ Bad frames """
        qc_paths = (self.session_path.joinpath(f[1], 'exptQC.mat')
                    for f in self.input_files if f[0] == 'exptQC.mat')
        qc_paths = map(str, filter(Path.exists, qc_paths))
        exptQC = [loadmat(p, squeeze_me=True, simplify_cells=True) for p in qc_paths]
        if len(exptQC) > 0:
            frameQC, frameQC_names, bad_frames = self._consolidate_exptQC(exptQC)
        else:
            _logger.warning('No frame QC (exptQC.mat) files found.')
            frameQC, bad_frames = np.array([], dtype='u1'), np.array([], dtype='i8')
            frameQC_names = pd.DataFrame(columns=['qc_values', 'qc_labels'])
        # If applicable, save as bad_frames.npy in first raw_imaging_folder for suite2p
        if len(bad_frames) > 0 and use_badframes is True:
            np.save(Path(db['data_path'][0]).joinpath('bad_frames.npy'), bad_frames)

        """ Suite2p """
        # Create alf it is doesn't exist
        self.session_path.joinpath('alf').mkdir(exist_ok=True)
        # Remove existing suite2p dir if it exists
        suite2p_dir = Path(db['save_path0']).joinpath('suite2p')
        if suite2p_dir.exists():
            shutil.rmtree(str(suite2p_dir), ignore_errors=True, onerror=None)
        # Run suite2p
        if run_suite2p:
            _ = suite2p.run_s2p(ops=ops, db=db)

        """ Outputs """
        # Save and rename other outputs
        if rename_files:
            out_files = self._rename_outputs(suite2p_dir, frameQC_names, frameQC)
        else:
            out_files = list(Path(db['save_path0']).joinpath('suite2p').rglob('*'))
        # Only return output file that are in the signature (for registration)
        out_files = [f for f in out_files if f.name in [f[0] for f in self.output_files]]
        return out_files


class MesoscopeSync(base_tasks.MesoscopeTask):
    """Extract the frame times from the main DAQ."""

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [(f'_{self.sync_namespace}_DAQdata.raw.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_DAQdata.timestamps.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_DAQdata.meta.json', self.sync_collection, True),
                            ('_ibl_rawImagingData.meta.json', self.device_collection, True),
                            ('rawImagingData.times_scanImage.npy', self.device_collection, True),
                            (f'_{self.sync_namespace}_softwareEvents.log.htsv', self.sync_collection, False), ],
            'output_files': [('mpci.times.npy', 'alf/mesoscope/FOV*', True),
                             ('mpciStack.timeshift.npy', 'alf/mesoscope/FOV*', True), ]
        }
        return signature

    def _run(self):
        """
        Extract the imaging times for all FOVs.

        Returns
        -------
        list of pathlib.Path
            Files containing frame timestamps for individual FOVs and time offsets for each line scan.

        """
        # TODO function to determine nFOVs
        try:
            alf_path = self.session_path / self.sync_collection
            events = alfio.load_object(alf_path, 'softwareEvents').get('log')
        except alferr.ALFObjectNotFound:
            events = None
        if events is None or events.empty:
            _logger.debug('No software events found for session %s', self.session_path)
        collections = set(collection for _, collection, _ in self.input_files
                          if fnmatch(collection, self.device_collection))
        # Load first meta data file to determine the number of FOVs
        # Changing FOV between imaging bouts is not supported currently!
        self.rawImagingData = alfio.load_object(self.session_path / next(iter(collections)), 'rawImagingData')
        self.rawImagingData['meta'] = mesoscope.patch_imaging_meta(self.rawImagingData['meta'])
        n_FOVs = len(self.rawImagingData['meta']['FOV'])
        sync, chmap = self.load_sync()  # Extract sync data from raw DAQ data
        mesosync = mesoscope.MesoscopeSyncTimeline(self.session_path, n_FOVs)
        _, out_files = mesosync.extract(
            save=True, sync=sync, chmap=chmap, device_collection=collections, events=events)
        return out_files


class MesoscopeFOV(base_tasks.MesoscopeTask):
    """Create FOV and FOV location objects in Alyx from metadata."""

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('_ibl_rawImagingData.meta.json', self.device_collection, True),
                            ('mpciROIs.stackPos.npy', 'alf/FOV*', True)],
            'output_files': [('mpciMeanImage.brainLocationIds*.npy', 'alf/FOV_*', True),
                             ('mpciMeanImage.mlapdv*.npy', 'alf/FOV_*', True),
                             ('mpciROIs.mlapdv*.npy', 'alf/FOV_*', True),
                             ('mpciROIs.brainLocationIds*.npy', 'alf/FOV_*', True),
                             ('_ibl_rawImagingData.meta.json', self.device_collection, True)]
        }
        return signature

    def _run(self, *args, provenance=Provenance.ESTIMATE, **kwargs):
        """
        Register fields of view (FOV) to Alyx and extract the coordinates and IDs of each ROI.

        Steps:
            1. Save the mpciMeanImage.brainLocationIds_estimate and mlapdv datasets.
            2. Use mean image coordinates and ROI stack position datasets to extract brain location
             of each ROI.
            3. Register the location of each FOV in Alyx.

        Parameters
        ----------
        provenance : Provenance
            The provenance of the coordinates in the meta file. For all but 'HISTOLOGY', the
            provenance is added as a dataset suffix.  Defaults to ESTIMATE.

        Returns
        -------
        dict
            The newly created FOV Alyx record.
        list
            The newly created FOV location Alyx records.

        Notes
        -----
        - Once the FOVs have been registered they cannot be updated with this task. Rerunning this
          task will result in an error.
        - This task modifies the first meta JSON file.  All meta files are registered by this task.
        """
        # Load necessary data
        (filename, collection, _), *_ = self.signature['input_files']
        meta_files = sorted(self.session_path.glob(f'{collection}/{filename}'))
        meta = mesoscope.patch_imaging_meta(alfio.load_file_content(meta_files[0]) or {})
        nFOV = len(meta.get('FOV', []))

        suffix = None if provenance is Provenance.HISTOLOGY else provenance.name.lower()
        _logger.info('Extracting %s MLAPDV datasets', suffix or 'final')

        # Extract mean image MLAPDV coordinates and brain location IDs
        mean_image_mlapdv, mean_image_ids = self.project_mlapdv(meta)

        # Save the meta data file with new coordinate fields
        with open(meta_files[0], 'w') as fp:
            json.dump(meta, fp)

        # Save the mean image datasets
        mean_image_files = []
        assert set(mean_image_mlapdv.keys()) == set(mean_image_ids.keys()) and len(mean_image_ids) == nFOV
        for i in range(nFOV):
            alf_path = self.session_path.joinpath('alf', f'FOV_{i:02}')
            for attr, arr, sfx in (('mlapdv', mean_image_mlapdv[i], suffix),
                                   ('brainLocationIds', mean_image_ids[i], ('ccf', '2017', suffix))):
                mean_image_files.append(alf_path / to_alf('mpciMeanImage', attr, 'npy', timescale=sfx))
                np.save(mean_image_files[-1], arr)

        # Extract ROI MLAPDV coordinates and brain location IDs
        roi_mlapdv, roi_brain_ids = self.roi_mlapdv(nFOV, suffix=suffix)

        # Write MLAPDV + brain location ID of ROIs to disk
        roi_files = []
        assert set(roi_mlapdv.keys()) == set(roi_brain_ids.keys()) and len(roi_mlapdv) == nFOV
        for i in range(nFOV):
            alf_path = self.session_path.joinpath('alf', f'FOV_{i:02}')
            for attr, arr, sfx in (('mlapdv', roi_mlapdv[i], suffix),
                                   ('brainLocationIds', roi_brain_ids[i], ('ccf', '2017', suffix))):
                roi_files.append(alf_path / to_alf('mpciROIs', attr, 'npy', timescale=sfx))
                np.save(roi_files[-1], arr)

        # Register FOVs in Alyx
        self.register_fov(meta, suffix)

        return sorted([*meta_files, *roi_files, *mean_image_files])

    def update_surgery_json(self, meta, normal_vector):
        """
        Update surgery JSON with surface normal vector.

        Adds the key 'surface_normal_unit_vector' to the most recent surgery JSON, containing the
        provided three element vector.  The recorded craniotomy center must match the coordinates
        in the provided meta file.

        Parameters
        ----------
        meta : dict
            The imaging meta data file containing the 'centerMM' key.
        normal_vector : array_like
            A three element unit vector normal to the surface of the craniotomy center.

        Returns
        -------
        dict
            The updated surgery record, or None if no surgeries found.
        """
        if not self.one or self.one.offline:
            _logger.warning('failed to update surgery JSON: ONE offline')
            return
        # Update subject JSON with unit normal vector of craniotomy centre (used in histology)
        subject = self.one.path2ref(self.session_path, parse=False)['subject']
        surgeries = self.one.alyx.rest('surgeries', 'list', subject=subject, procedure='craniotomy')
        if not surgeries:
            _logger.error(f'Surgery not found for subject "{subject}"')
            return
        surgery = surgeries[0]  # Check most recent surgery in list
        center = (meta['centerMM']['ML'], meta['centerMM']['AP'])
        match = (k for k, v in surgery['json'].items() if
                 str(k).startswith('craniotomy') and np.allclose(v['center'], center))
        if (key := next(match, None)) is None:
            _logger.error('Failed to update surgery JSON: no matching craniotomy found')
            return surgery
        data = {key: {**surgery['json'][key], 'surface_normal_unit_vector': tuple(normal_vector)}}
        surgery['json'] = self.one.alyx.json_field_update('subjects', subject, data=data)
        return surgery

    def roi_mlapdv(self, nFOV: int, suffix=None):
        """
        Extract ROI MLAPDV coordinates and brain location IDs.

        MLAPDV coordinates are in um relative to bregma.  Location IDs are from the 2017 Allen
        common coordinate framework atlas.

        Parameters
        ----------
        nFOV : int
            The number of fields of view acquired.
        suffix : {None, 'estimate'}
            The attribute suffix of the mpciMeanImage datasets to load. If generating from
            estimates, the suffix should be 'estimate'.

        Returns
        -------
        dict of int : numpy.array
            A map of field of view to ROI MLAPDV coordinates.
        dict of int : numpy.array
            A map of field of view to ROI brain location IDs.
        """
        all_mlapdv = {}
        all_brain_ids = {}
        for n in range(nFOV):
            alf_path = self.session_path.joinpath('alf', f'FOV_{n:02}')

            # Load neuron centroids in pixel space
            stack_pos_file = next(alf_path.glob('mpciROIs.stackPos*'), None)
            if not stack_pos_file:
                raise FileNotFoundError(alf_path / 'mpci.stackPos*')
            stack_pos = alfio.load_file_content(stack_pos_file)

            # Load MLAPDV + brain location ID maps of pixels
            mpciMeanImage = alfio.load_object(
                alf_path, 'mpciMeanImage', attribute=['mlapdv', 'brainLocationIds'])

            # Get centroid MLAPDV + brainID by indexing pixel-map with centroid locations
            mlapdv = np.full(stack_pos.shape, np.nan)
            brain_ids = np.full(stack_pos.shape[0], np.nan)
            for i in np.arange(stack_pos.shape[0]):
                idx = (stack_pos[i, 0], stack_pos[i, 1])
                sfx = f'_{suffix}' if suffix else ''
                mlapdv[i, :] = mpciMeanImage['mlapdv' + sfx][idx]
                brain_ids[i] = mpciMeanImage['brainLocationIds_ccf_2017' + sfx][idx]
            assert ~np.isnan(brain_ids).any()
            all_brain_ids[n] = brain_ids.astype(int)
            all_mlapdv[n] = mlapdv

        return all_mlapdv, all_brain_ids

    @staticmethod
    def get_provenance(filename):
        """
        Get the field of view provenance from a mpciMeanImage or mpciROIs dataset.

        Parameters
        ----------
        filename : str, pathlib.Path
            A filename to get the provenance from.

        Returns
        -------
        Provenance
            The provenance of the file.
        """
        filename = Path(filename).name
        timescale = (filename_parts(filename)[3] or '').split('_')
        provenances = [i.name.lower() for i in Provenance]
        provenance = (Provenance[x.upper()] for x in timescale if x in provenances)
        return next(provenance, None) or Provenance.HISTOLOGY

    def register_fov(self, meta: dict, suffix: str = None) -> (list, list):
        """
        Create FOV on Alyx.

        Assumes field of view recorded perpendicular to objective.
        Assumes field of view is plane (negligible volume).

        Required Alyx fixtures:
            - experiments.ImagingType(name='mesoscope')
            - experiments.CoordinateSystem(name='IBL-Allen')

        Parameters
        ----------
        meta : dict
            The raw imaging meta data from _ibl_rawImagingData.meta.json.
        suffix : str
            The file attribute suffixes to load from the mpciMeanImage object. Either 'estimate' or
            None. No suffix means the FOV location provenance will be L (Landmark).

        Returns
        -------
        list of dict
            A list registered of field of view entries from Alyx.

        TODO Determine dual plane ID for JSON field
        """
        dry = self.one is None or self.one.offline
        alyx_fovs = []
        # Count the number of slices per stack ID: only register stacks that contain more than one slice.
        slice_counts = Counter(f['roiUUID'] for f in meta.get('FOV', []))
        # Create a new stack in Alyx for all stacks containing more than one slice.
        # Map of ScanImage ROI UUID to Alyx ImageStack UUID.
        if dry:
            stack_ids = {i: uuid.uuid4() for i in slice_counts if slice_counts[i] > 1}
        else:
            stack_ids = {i: self.one.alyx.rest('imaging-stack', 'create', data={'name': i})['id']
                         for i in slice_counts if slice_counts[i] > 1}

        for i, fov in enumerate(meta.get('FOV', [])):
            assert set(fov.keys()) >= {'MLAPDV', 'nXnYnZ', 'roiUUID'}
            # Field of view
            alyx_FOV = {
                'session': self.session_path.as_posix() if dry else self.path2eid(),
                'imaging_type': 'mesoscope', 'name': f'FOV_{i:02}',
                'stack': stack_ids.get(fov['roiUUID'])
            }
            if dry:
                print(alyx_FOV)
                alyx_FOV['location'] = []
                alyx_fovs.append(alyx_FOV)
            else:
                alyx_fovs.append(self.one.alyx.rest('fields-of-view', 'create', data=alyx_FOV))

            # Field of view location
            data = {'field_of_view': alyx_fovs[-1].get('id'),
                    'default_provenance': True,
                    'coordinate_system': 'IBL-Allen',
                    'n_xyz': fov['nXnYnZ']}
            if suffix:
                data['provenance'] = suffix[0].upper()

            # Convert coordinates to 4 x 3 array (n corners by n dimensions)
            # x1 = top left ml, y1 = top left ap, y2 = top right ap, etc.
            coords = [fov['MLAPDV'][key] for key in ('topLeft', 'topRight', 'bottomLeft', 'bottomRight')]
            coords = np.vstack(coords).T
            data.update({k: arr.tolist() for k, arr in zip('xyz', coords)})

            # Load MLAPDV + brain location ID maps of pixels
            filename = 'mpciMeanImage.brainLocationIds_ccf_2017' + (f'_{suffix}' if suffix else '') + '.npy'
            filepath = self.session_path.joinpath('alf', f'FOV_{i:02}', filename)
            mean_image_ids = alfio.load_file_content(filepath)

            data['brain_region'] = np.unique(mean_image_ids).astype(int).tolist()

            if dry:
                print(data)
                alyx_FOV['location'].append(data)
            else:
                alyx_fovs[-1]['location'].append(self.one.alyx.rest('fov-location', 'create', data=data))
        return alyx_fovs

    def load_triangulation(self):
        """
        Load the surface triangulation file.

        A triangle mesh of the smoothed convex hull of the dorsal surface of the mouse brain,
        generated from the 2017 Allen 10um annotation volume. This triangulation was generated in
        MATLAB.

        Returns
        -------
        points : numpy.array
            An N by 3 float array of x-y vertices, defining all points of the triangle mesh. These
            are in um relative to the IBL bregma coordinates.
        connectivity_list : numpy.array
            An N by 3 integer array of vertex indices defining all points that form a triangle.
        """
        fixture_path = Path(mesoscope.__file__).parent.joinpath('mesoscope')
        surface_triangulation = np.load(fixture_path / 'surface_triangulation.npz')
        points = surface_triangulation['points'].astype('f8')
        connectivity_list = surface_triangulation['connectivity_list']
        surface_triangulation.close()
        return points, connectivity_list

    def project_mlapdv(self, meta, atlas=None):
        """
        Calculate the mean image pixel locations in MLAPDV coordinates and determine the brain
        location IDs.

        MLAPDV coordinates are in um relative to bregma.  Location IDs are from the 2017 Allen
        common coordinate framework atlas.

        Parameters
        ----------
        meta : dict
            The raw imaging data meta file, containing coordinates for the centre of each field of
            view.
        atlas : ibllib.atlas.Atlas
            An atlas instance.

        Returns
        -------
        dict
            A map of FOV number (int) to mean image MLAPDV coordinates as a 2D numpy array.
        dict
            A map of FOV number (int) to mean image brain location IDs as a 2D numpy int array.
        """
        mlapdv = {}
        location_id = {}
        # Use the MRI atlas as this applies scaling, particularly along the DV axis to (hopefully)
        # more accurately represent the living brain.
        atlas = atlas or MRITorontoAtlas(res_um=10)
        # The centre of the craniotomy / imaging window
        coord_ml = meta['centerMM']['ML'] * 1e3  # mm -> um
        coord_ap = meta['centerMM']['AP'] * 1e3  # mm -> um
        pt = np.array([coord_ml, coord_ap])

        points, connectivity_list = self.load_triangulation()
        # Only keep faces that have normals pointing up (positive DV value).
        # Calculate the normal vector pointing out of the convex hull.
        triangles = points[connectivity_list, :]
        normals = surface_normal(triangles)
        up_faces, = np.where(normals[:, -1] > 0)
        # only keep triangles that have normal vector with positive DV component
        dorsal_connectivity_list = connectivity_list[up_faces, :]
        # Flatten triangulation by dropping the dorsal coordinates and find the location of the
        # window center (we convert mm -> um here)
        face_ind = find_triangle(pt * 1e-3, points[:, :2] * 1e-3, dorsal_connectivity_list.astype(np.intp))
        assert face_ind != -1

        dorsal_triangle = points[dorsal_connectivity_list[face_ind, :], :]

        # Get the surface normal unit vector of dorsal triangle
        normal_vector = surface_normal(dorsal_triangle)

        # Update the surgery JSON field with normal unit vector, for use in histology alignment
        self.update_surgery_json(meta, normal_vector)

        # find the coordDV that sits on the triangular face and had [coordML, coordAP] coordinates;
        # the three vertices defining the triangle
        face_vertices = points[dorsal_connectivity_list[face_ind, :], :]

        # all the vertices should be on the plane ax + by + cz = 1, so we can find
        # the abc coefficients by inverting the three equations for the three vertices
        abc, *_ = np.linalg.lstsq(face_vertices, np.ones(3), rcond=None)

        # and then find a point on that plane that corresponds to a given x-y
        # coordinate (which is ML-AP coordinate)
        coord_dv = (1 - pt @ abc[:2]) / abc[2]

        # We should not use the actual surface of the brain for this, as it might be in one of the sulci
        # DO NOT USE THIS:
        # coordDV = interp2(axisMLmm, axisAPmm, surfaceDV, coordML, coordAP)

        # Now we need to span the plane of the coverslip with two orthogonal unit vectors.
        # We start with vY, because the order is important and we usually have less
        # tilt along AP (pitch), which will cause less deviation in vX from pure ML.
        vY = np.array([0, normal_vector[2], -normal_vector[1]])  # orthogonal to the normal of the plane
        vX = np.cross(vY, normal_vector)  # orthogonal to n and to vY
        # normalize and flip the sign if necessary
        vX = vX / np.sqrt(vX @ vX) * np.sign(vX[0])  # np.sqrt(vY @ vY) == LR norm of vX
        vY = vY / np.sqrt(vY @ vY) * np.sign(vY[1])

        # what are the dimensions of the data arrays (ap, ml, dv)
        (nAP, nML, nDV) = atlas.image.shape
        # Let's shift the coordinates relative to bregma
        voxel_size = atlas.res_um  # [um] resolution of the atlas
        bregma_coords = ALLEN_CCF_LANDMARKS_MLAPDV_UM['bregma'] / voxel_size  # (ml, ap, dv)
        axis_ml_um = (np.arange(nML) - bregma_coords[0]) * voxel_size
        axis_ap_um = (np.arange(nAP) - bregma_coords[1]) * voxel_size * -1.
        axis_dv_um = (np.arange(nDV) - bregma_coords[2]) * voxel_size * -1.

        # projection of FOVs on the brain surface to get ML-AP-DV coordinates
        _logger.info('Projecting in 3D')
        for i, fov in enumerate(meta['FOV']):  # i, fov = next(enumerate(meta['FOV']))
            start_time = time.time()
            _logger.info(f'FOV {i + 1}/{len(meta["FOV"])}')
            y_px_idx, x_px_idx = np.mgrid[0:fov['nXnYnZ'][0], 0:fov['nXnYnZ'][1]]

            # xx and yy are in mm in coverslip space
            points = ((0, fov['nXnYnZ'][0] - 1), (0, fov['nXnYnZ'][1] - 1))
            # The four corners of the FOV, determined by taking the center of the craniotomy in MM,
            # the x-y coordinates of the imaging window center (from the tiled reference image) in
            # galvanometer units, and the x-y coordinates of the FOV center in galvanometer units.
            values = [[fov['MM']['topLeft'][0], fov['MM']['topRight'][0]],
                      [fov['MM']['bottomLeft'][0], fov['MM']['bottomRight'][0]]]
            values = np.array(values) * 1e3  # mm -> um
            xx = interpn(points, values, (y_px_idx, x_px_idx))

            values = [[fov['MM']['topLeft'][1], fov['MM']['topRight'][1]],
                      [fov['MM']['bottomLeft'][1], fov['MM']['bottomRight'][1]]]
            values = np.array(values) * 1e3  # mm -> um
            yy = interpn(points, values, (y_px_idx, x_px_idx))

            xx = xx.flatten() - coord_ml
            yy = yy.flatten() - coord_ap

            # rotate xx and yy in 3D
            # the coords are still on the coverslip, but now have 3D values
            coords = np.outer(xx, vX) + np.outer(yy, vY)  # (vX * xx) + (vY * yy)
            coords = coords + [coord_ml, coord_ap, coord_dv]

            # for each point of the FOV create a line parametrization (trajectory normal to the coverslip plane).
            # start just above the coverslip and go 3 mm down, should be enough to 'meet' the brain
            t = np.arange(-voxel_size, 3e3, voxel_size)

            # Find the MLAPDV atlas coordinate and brain location of each pixel.
            MLAPDV, annotation = _update_points(
                t, normal_vector, coords, axis_ml_um, axis_ap_um, axis_dv_um, atlas.label)
            annotation = atlas.regions.index2id(annotation)  # convert annotation indices to IDs

            if np.any(np.isnan(MLAPDV)):
                _logger.warning('Areas of FOV lie outside the brain')
            _logger.info(f'done ({time.time() - start_time:3.1f} seconds)\n')
            MLAPDV = np.reshape(MLAPDV, [*x_px_idx.shape, 3])
            annotation = np.reshape(annotation, x_px_idx.shape)

            fov['MLAPDV'] = {
                'topLeft': MLAPDV[0, 0, :].tolist(),
                'topRight': MLAPDV[0, -1, :].tolist(),
                'bottomLeft': MLAPDV[-1, 0, :].tolist(),
                'bottomRight': MLAPDV[-1, -1, :].tolist(),
                'center': MLAPDV[round(x_px_idx.shape[0] / 2) - 1, round(x_px_idx.shape[1] / 2) - 1, :].tolist()
            }

            # Save the brain regions of the corners/centers of FOV (annotation field)
            fov['brainLocationIds'] = {
                'topLeft': int(annotation[0, 0]),
                'topRight': int(annotation[0, -1]),
                'bottomLeft': int(annotation[-1, 0]),
                'bottomRight': int(annotation[-1, -1]),
                'center': int(annotation[round(x_px_idx.shape[0] / 2) - 1, round(x_px_idx.shape[1] / 2) - 1])
            }

            mlapdv[i] = MLAPDV
            location_id[i] = annotation
        return mlapdv, location_id


def surface_normal(triangle):
    """
    Calculate the surface normal unit vector of one or more triangles.

    Parameters
    ----------
    triangle : numpy.array
        An array of shape (n_triangles, 3, 3) representing (Px Py Pz).

    Returns
    -------
    numpy.array
        The surface normal unit vector(s).
    """
    if triangle.shape == (3, 3):
        triangle = triangle[np.newaxis, :, :]
    if triangle.shape[1:] != (3, 3):
        raise ValueError('expected array of shape (3, 3); 3 coordinates in x, y, and z')
    V = triangle[:, 1, :] - triangle[:, 0, :]  # V = P2 - P1
    W = triangle[:, 2, :] - triangle[:, 0, :]  # W = P3 - P1

    Nx = (V[:, 1] * W[:, 2]) - (V[:, 2] * W[:, 1])  # Nx = (Vy * Wz) - (Vz * Wy)
    Ny = (V[:, 2] * W[:, 0]) - (V[:, 0] * W[:, 2])  # Ny = (Vz * Wx) - (Vx * Wz)
    Nz = (V[:, 0] * W[:, 1]) - (V[:, 1] * W[:, 0])  # Nz = (Vx * Wy) - (Vy * Wx)
    N = np.c_[Nx, Ny, Nz]
    # Calculate unit vector. Transpose allows vectorized operation.
    A = N / np.sqrt((Nx ** 2) + (Ny ** 2) + (Nz ** 2))[np.newaxis].T
    return A.squeeze()


@nb.njit('b1(f8[:,:], f8[:])')
def in_triangle(triangle, point):
    """
    Check whether `point` lies within `triangle`.

    Parameters
    ----------
    triangle : numpy.array
        A (2 x 3) array of x-y coordinates; A(x1, y1), B(x2, y2) and C(x3, y3).
    point : numpy.array
        A point, P(x, y).

    Returns
    -------
    bool
        True if coordinate lies within triangle.
    """
    def area(x1, y1, x2, y2, x3, y3):
        """Calculate the area of a triangle, given its vertices."""
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.)

    x1, y1, x2, y2, x3, y3 = triangle.flat
    x, y = point
    A = area(x1, y1, x2, y2, x3, y3)  # area of triangle ABC
    A1 = area(x, y, x2, y2, x3, y3)  # area of triangle PBC
    A2 = area(x1, y1, x, y, x3, y3)  # area of triangle PAC
    A3 = area(x1, y1, x2, y2, x, y)  # area of triangle PAB
    # Check if sum of A1, A2 and A3 equals that of A
    diff = np.abs((A1 + A2 + A3) - A)
    REL_TOL = 1e-9
    return diff <= np.abs(REL_TOL * A)  # isclose not yet implemented in numba 0.57


@nb.njit('i8(f8[:], f8[:,:], intp[:,:])', nogil=True)
def find_triangle(point, vertices, connectivity_list):
    """
    Find which vertices contain a given point.

    Currently O(n) but could take advantage of connectivity order to be quicker.

    Parameters
    ----------
    point : numpy.array
        The (x, y) coordinate of a point to locate within one of the triangles.
    vertices : numpy.array
        An N x 3 array of vertices representing a triangle mesh.
    connectivity_list : numpy.array
        An N x 3 array of indices representing the connectivity of `points`.

    Returns
    -------
    int
        The index of the vertices containing `point`, or -1 if not within any triangle.
    """
    face_ind = -1
    for i in nb.prange(connectivity_list.shape[0]):
        triangle = vertices[connectivity_list[i, :], :]
        if in_triangle(triangle, point):
            face_ind = i
            break
    return face_ind


@nb.njit('Tuple((f8[:], intp[:]))(f8[:], f8[:])', nogil=True)
def _nearest_neighbour_1d(x, x_new):
    """
    Nearest neighbour interpolation with extrapolation.

    This was adapted from scipy.interpolate.interp1d but returns the indices of each nearest x
    value.  Assumes x is not sorted.

    Parameters
    ----------
    x : (N,) array_like
        A 1-D array of real values.
    x_new : (N,) array_like
        A 1D array of values to apply function to.

    Returns
    -------
    numpy.array
        A 1D array of interpolated values.
    numpy.array
        A 1D array of indices.
    """
    SIDE = 'left'  # use 'right' to round up to nearest int instead of rounding down
    # Sort values
    ind = np.argsort(x, kind='mergesort')
    x = x[ind]
    x_bds = x / 2.0  # Do division before addition to prevent possible integer overflow
    x_bds = x_bds[1:] + x_bds[:-1]
    # Find where in the averaged data the values to interpolate would be inserted.
    x_new_indices = np.searchsorted(x_bds, x_new, side=SIDE)
    # Clip x_new_indices so that they are within the range of x indices.
    x_new_indices = x_new_indices.clip(0, len(x) - 1).astype(np.intp)
    # Calculate the actual value for each entry in x_new.
    y_new = x[x_new_indices]
    return y_new, ind[x_new_indices]


@nb.njit('Tuple((f8[:,:], u2[:]))(f8[:], f8[:], f8[:,:], f8[:], f8[:], f8[:], u2[:,:,:])', nogil=True)
def _update_points(t, normal_vector, coords, axis_ml_um, axis_ap_um, axis_dv_um, atlas_labels):
    """
    Determine the MLAPDV coordinate and brain location index for each of the given coordinates.

    This has been optimized in numba. The majority of the time savings come from replacing iterp1d
    and ismember with _nearest_neighbour_1d which were extremely slow. Parallel iteration further
    halved the time it took per 512x512 FOV.

    Parameters
    ----------
    t : numpy.array
        An N x 3 evenly spaced set of coordinates representing points going down from the coverslip
        towards the brain.
    normal_vector : numpy.array
        The unit vector of the face normal to the center of the window.
    coords : numpy.array
        A set of N x 3 coordinates representing the MLAPDV coordinates of each pixel relative to
        the center of the window, in micrometers (um).
    axis_ml_um : numpy.array
        An evenly spaced array of medio-lateral brain coordinates relative to bregma in um, at the
        resolution of the atlas image used.
    axis_ap_um : numpy.array
        An evenly spaced array of anterio-posterior brain coordinates relative to bregma in um, at
        the resolution of the atlas image used.
    axis_dv_um : numpy.array
        An evenly spaced array of dorso-ventral brain coordinates relative to bregma in um, at
        the resolution of the atlas image used.
    atlas_labels : numpy.array
        A 3D array of integers representing the brain location index of each voxel of a given
        atlas. The shape is expected to be (nAP, nML, nDV).

    Returns
    -------
    numpy.array
        An N by 3 array containing the MLAPDV coordinates in um of each pixel coordinate.
        Coordinates outside of the brain are NaN.
    numpy.array
        A 1D array of atlas label indices the length of `coordinates`.
    """
    # passing through the center of the craniotomy/coverslip
    traj_coords_centered = np.outer(t, -normal_vector)
    MLAPDV = np.full_like(coords, np.nan)
    annotation = np.zeros(coords.shape[0], dtype=np.uint16)
    n_points = coords.shape[0]
    for p in nb.prange(n_points):
        # Shifted to the correct point on the coverslip, in true ML-AP-DV coords
        traj_coords = traj_coords_centered + coords[p, :]

        # Find intersection coordinate with the brain.
        # Only use coordinates that exist in the atlas (kind of nearest neighbour interpolation)
        ml, ml_idx = _nearest_neighbour_1d(axis_ml_um, traj_coords[:, 0])
        ap, ap_idx = _nearest_neighbour_1d(axis_ap_um, traj_coords[:, 1])
        dv, dv_idx = _nearest_neighbour_1d(axis_dv_um, traj_coords[:, 2])

        # Iterate over coordinates to find the first (if any) that is within the brain
        ind = -1
        area = 0  # 0 = void; 1 = root
        for i in nb.prange(traj_coords.shape[0]):
            anno = atlas_labels[ap_idx[i], ml_idx[i], dv_idx[i]]
            if anno > 0:  # first coordinate in the brain
                ind = i
                area = anno
                if area > 1:  # non-root brain area; we're done
                    break
        if area > 1:
            point = traj_coords[ind, :]
            MLAPDV[p, :] = point  # in um
            annotation[p] = area
        else:
            MLAPDV[p, :] = np.nan
            annotation[p] = area  # root or void

    return MLAPDV, annotation

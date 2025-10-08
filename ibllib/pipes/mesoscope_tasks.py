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
import logging
import subprocess
from zipfile import ZipFile, ZIP_DEFLATED
import uuid
from pathlib import Path
from itertools import chain, groupby
from fnmatch import fnmatch
import shutil
import re

import numpy as np
import pandas as pd
import sparse
from scipy.io import loadmat
import one.alf.io as alfio
from one.alf.path import session_path_parts
import one.alf.exceptions as alferr
from iblutil.util import flatten, ensure_list

from ibllib.pipes import base_tasks
from ibllib.oneibl.data_handlers import ExpectedDataset, dataset_from_name
from ibllib.io.extractors import mesoscope


_logger = logging.getLogger(__name__)


class MesoscopeRegisterSnapshots(base_tasks.MesoscopeTask, base_tasks.RegisterRawDataTask):
    """Upload snapshots as Alyx notes and register the 2P reference image(s)."""
    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        I = ExpectedDataset.input  # noqa
        signature = {
            'input_files': [I('referenceImage.raw.tif', f'{self.device_collection}/reference', False, register=True),
                            I('referenceImage.stack.tif', f'{self.device_collection}/reference', False, register=True),
                            I('referenceImage.meta.json', f'{self.device_collection}/reference', False, register=True)],
            'output_files': []
        }
        return signature

    def __init__(self, session_path, **kwargs):
        super().__init__(session_path, **kwargs)
        self.device_collection = self.get_device_collection('mesoscope',
                                                            kwargs.get('device_collection', 'raw_imaging_data_??'))

    def _run(self):
        """
        Assert one reference image per collection and rename it. Register snapshots.

        Returns
        -------
        list of pathlib.Path containing renamed reference image.
        """
        # Assert that only one tif file exists per collection
        dsets = dataset_from_name('referenceImage.raw.tif', self.input_files)
        reference_images = list(chain.from_iterable(map(lambda x: x.find_files(self.session_path)[1], dsets)))
        assert len(set(x.parent for x in reference_images)) == len(reference_images)
        # Rename the reference images
        out_files = super()._run()
        # Register snapshots in base session folder and raw_imaging_data folders
        self.register_snapshots(collection=[self.device_collection, ''])
        return out_files


class MesoscopeCompress(base_tasks.MesoscopeTask):
    """ Tar compress raw 2p tif files, optionally remove uncompressed data."""

    priority = 90
    io_charge = 100
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

    def _run(self, remove_uncompressed=False, verify_output=True, overwrite=False, **kwargs):
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
        _, all_tifs, _ = zip(*(x.find_files(self.session_path) for x in self.input_files))
        if self.input_files[0].operator:  # multiple device collections
            output_identifiers = self.output_files[0].identifiers
            # Check that the number of input ollections and output files match
            assert len(self.input_files[0].identifiers) == len(output_identifiers)
        else:
            output_identifiers = [self.output_files[0].identifiers]
            assert self.output_files[0].operator is None, 'only one output file expected'

        # A list of tifs, grouped by raw imaging data collection
        input_files = groupby(chain.from_iterable(all_tifs), key=lambda x: x.parent)
        for (in_dir, infiles), out_id in zip(input_files, output_identifiers):
            infiles = list(infiles)
            outfile = self.session_path.joinpath(*filter(None, out_id))
            if outfile.exists() and not overwrite:
                _logger.info('%s already exists; skipping...', outfile.relative_to(self.session_path))
                outfiles.append(outfile)
            else:
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
    """Run suite2p preprocessing on tif files."""

    priority = 100
    io_charge = 100
    cpu = -1
    job_size = 'large'
    env = 'suite2p'

    def __init__(self, *args, **kwargs):
        self._teardown_files = []
        self.overwrite = False
        super().__init__(*args, **kwargs)

    def setUp(self, **kwargs):
        """Set up task.

        This will check the local filesystem for the raw tif files and if not present, will assume
        they have been compressed and deleted, in which case the signature will be replaced with
        the compressed input.

        Note: this will not work correctly if only some collections have compressed tifs.
        """
        self.overwrite = kwargs.get('overwrite', False)
        all_files_present = super().setUp(**kwargs)  # Ensure files present
        if not self.overwrite:
            # Check if the bin files already exist on disk, in which case we don't need to extract tifs
            bin_sig, = dataset_from_name('data.bin', self.input_files)
            renamed_bin_sig, = dataset_from_name('imaging.frames_motionRegistered.bin', self.input_files)
            if (bin_sig | renamed_bin_sig).find_files(self.session_path)[0]:
                return all_files_present  # We have local bin files; no need to extract tifs
        tif_sig = dataset_from_name('*.tif', self.input_files)
        if not tif_sig:
            return all_files_present  # No tifs in the signature; just return
        tif_sig = tif_sig[0]
        tifs_present, *_ = tif_sig.find_files(self.session_path)
        if tifs_present or not all_files_present:
            return all_files_present  # Tifs present on disk; no need to decompress
        # Decompress imaging files
        tif_sigs = dataset_from_name('imaging.frames.tar.bz2', self.input_files)
        present, files, _ = zip(*(x.find_files(self.session_path) for x in tif_sigs))
        if not all(present):
            return False  # Compressed files missing; return
        files = flatten(files)
        _logger.info('Decompressing %i file(s)', len(files))
        for file in files:
            cmd = 'tar -xvjf "{input}"'.format(input=file.name)
            _logger.debug(cmd)
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=file.parent)
            stdout, _ = process.communicate()  # b'x 2023-02-17_2_test_2P_00001_00001.tif\n'
            _logger.debug(stdout.decode())
            tifs = [file.parent.joinpath(x.split()[-1]) for x in stdout.decode().splitlines() if x.endswith('.tif')]
            assert process.returncode == 0 and len(tifs) > 0
            assert all(map(Path.exists, tifs))
            self._teardown_files.extend(tifs)
        return all_files_present

    def tearDown(self):
        """Tear down task.

        This removes any decompressed tif files.
        """
        for file in self._teardown_files:
            _logger.debug('Removing %s', file)
            file.unlink()
        return super().tearDown()

    @property
    def signature(self):
        # The number of in and outputs will be dependent on the number of input raw imaging folders and output FOVs
        I = ExpectedDataset.input  # noqa
        signature = {
            'input_files': [('_ibl_rawImagingData.meta.json', self.device_collection, True),
                            I('*.tif', self.device_collection, True) |
                            I('imaging.frames.tar.bz2', self.device_collection, True, unique=False),
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
                             ('mpciROIs.masks.sparse_npz', 'alf/FOV*', True),
                             ('mpciROIs.neuropilMasks.sparse_npz', 'alf/FOV*', True),
                             ('_suite2p_ROIData.raw.zip', 'alf/FOV*', False),
                             ('imaging.frames_motionRegistered.bin', 'suite2p/plane*', False)]
        }
        if not self.overwrite:  # If not forcing re-registration, check whether bin files already exist on disk
            # Including the data.bin in the expected signature ensures raw data files are not needlessly re-downloaded
            # and/or uncompressed during task setup as the local data.bin may be used instead
            # NB: The data.bin file is renamed to imaging.frames_motionRegistered.bin before registration to Alyx
            registered_bin = (I('data.bin', 'suite2p/plane*', True, unique=False) |
                              I('imaging.frames_motionRegistered.bin', 'suite2p/plane*', True, unique=False))
            signature['input_files'][1] = registered_bin | signature['input_files'][1]

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

        This also moves any data.bin and ops.npy files to raw_bin_files for quicker re-runs.

        Parameters
        ----------
        suite2p_dir : pathlib.Path
            The location of the suite2p output (typically session_path/suite2p).
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
        fov_dsets = [d[0] for d in self.signature['output_files'] if d[1].startswith('alf/FOV')]
        for plane_dir in self._get_plane_paths(suite2p_dir):
            # Rename the registered bin file
            if (bin_file := plane_dir.joinpath('data.bin')).exists():
                bin_file.rename(plane_dir.joinpath('imaging.frames_motionRegistered.bin'))
            # Archive the raw suite2p output before renaming
            n = int(plane_dir.name.split('plane')[1])
            fov_dir = self.session_path.joinpath('alf', f'FOV_{n:02}')
            if fov_dir.exists():
                for f in filter(Path.exists, map(fov_dir.joinpath, fov_dsets)):
                    _logger.debug('Removing old file %s', f.relative_to(self.session_path))
                    f.unlink()
            else:
                fov_dir.mkdir(parents=True)
            with ZipFile(fov_dir / '_suite2p_ROIData.raw.zip', 'w', compression=ZIP_DEFLATED) as zf:
                for file in plane_dir.iterdir():
                    if file.suffix != '.bin':
                        zf.write(file, arcname=file.name)
            # save frameQC in each dir (for now, maybe there will be fov specific frame QC eventually)
            if frameQC is not None and len(frameQC) > 0:
                np.save(fov_dir.joinpath('mpci.mpciFrameQC.npy'), frameQC)
                frameQC_names.to_csv(fov_dir.joinpath('mpciFrameQC.names.tsv'), sep='\t', index=False)
            # extract some other data from suite2p outputs
            ops = np.load(plane_dir.joinpath('ops.npy'), allow_pickle=True).item()
            stat = np.load(plane_dir.joinpath('stat.npy'), allow_pickle=True)
            iscell = np.load(plane_dir.joinpath('iscell.npy'))
            # Save suite2p ROI activity outputs in transposed from (n_frames, n_ROI)
            for k, v in rename_dict.items():
                np.save(fov_dir.joinpath(v), np.load(plane_dir.joinpath(k)).T)
            np.save(fov_dir.joinpath('mpci.badFrames.npy'), np.asarray(ops['badframes'], dtype=bool))
            np.save(fov_dir.joinpath('mpciMeanImage.images.npy'), np.asarray(ops['meanImg'], dtype=float))
            np.save(fov_dir.joinpath('mpciROIs.stackPos.npy'), np.asarray([(*s['med'], 0) for s in stat], dtype=int))
            np.save(fov_dir.joinpath('mpciROIs.cellClassifier.npy'), np.asarray(iscell[:, 1], dtype=float))
            np.save(fov_dir.joinpath('mpciROIs.mpciROITypes.npy'), np.asarray(iscell[:, 0], dtype=np.int16))
            # clusters uuids
            uuid_list = ['uuids'] + list(map(str, [uuid.uuid4() for _ in range(len(iscell))]))
            with open(fov_dir.joinpath('mpciROIs.uuids.csv'), 'w+') as fid:
                fid.write('\n'.join(uuid_list))
            (pd.DataFrame([(0, 'no cell'), (1, 'cell')], columns=['roi_values', 'roi_labels'])
             .to_csv(fov_dir.joinpath('mpciROITypes.names.tsv'), sep='\t', index=False))
            # ROI and neuropil masks
            roi_mask, pil_mask = self._masks2sparse(stat, ops)
            with open(fov_dir.joinpath('mpciROIs.masks.sparse_npz'), 'wb') as fp:
                sparse.save_npz(fp, roi_mask)
            with open(fov_dir.joinpath('mpciROIs.neuropilMasks.sparse_npz'), 'wb') as fp:
                sparse.save_npz(fp, pil_mask)
            # Remove the suite2p output files, leaving only the bins and ops
            for path in sorted(plane_dir.iterdir()):
                if path.name != 'ops.npy' and path.suffix != '.bin':
                    path.unlink() if path.is_file() else path.rmdir()
        # Collect all files in those directories
        datasets = self.session_path.joinpath('alf').rglob('FOV_??/*.*.*')
        return sorted(x for x in datasets if x.name in fov_dsets)

    def load_meta_files(self):
        """Load the extracted imaging metadata files.

        Loads and consolidates the imaging data metadata from rawImagingData.meta.json files.
        These files contain ScanImage metadata extracted from the raw tiff headers by the
        function `mesoscopeMetadataExtraction.m` in iblscripts/deploy/mesoscope.

        Returns
        -------
        dict
            Single, consolidated dictionary containing metadata.
        list of dict
            The meta data for each individual imaging bout.
        """
        # Load metadata and make sure all metadata is consistent across FOVs
        meta_files = sorted(self.session_path.glob(f'{self.device_collection}/*rawImagingData.meta.*'))
        collections = sorted(set(f.parts[-2] for f in meta_files))
        # Check there is exactly 1 meta file per collection
        assert len(meta_files) == len(list(self.session_path.glob(self.device_collection))) == len(collections)
        raw_meta = map(alfio.load_file_content, meta_files)
        all_meta = list(map(mesoscope.patch_imaging_meta, raw_meta))
        return self._consolidate_metadata(all_meta) if len(all_meta) > 1 else all_meta[0], all_meta

    @staticmethod
    def _consolidate_metadata(meta_data_all: list) -> dict:
        """
        Check that the metadata is consistent across all raw imaging folders.

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
        for meta in meta_data_all[1:]:
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
        # Create a new enumeration combining all unique QC labels.
        # 'ok' will always have an enum of 0, the rest are determined by order alone
        qc_labels = ['ok']
        frame_qc = []
        for e in exptQC:
            assert e.keys() >= set(['frameQC_names', 'frameQC_frames'])
            # Initialize an NaN array the same size of frameQC_frames to fill with new enum values
            frames = np.full(e['frameQC_frames'].shape, fill_value=np.nan)
            # May be numpy array of str or a single str, in both cases we cast to list of str
            names = list(ensure_list(e['frameQC_names']))
            # For each label for the old enum, populate initialized array with the new one
            for i_old, name in enumerate(names):
                name = name if len(name) else 'unknown'  # handle empty array and empty str
                try:
                    i_new = qc_labels.index(name)
                except ValueError:
                    i_new = len(qc_labels)
                    qc_labels.append(name)
                frames[e['frameQC_frames'] == i_old] = i_new
            frame_qc.append(frames)
        # Concatenate frames
        frame_qc = np.concatenate(frame_qc)
        # If any NaNs left over, assign 'unknown' label
        if (missing_name := np.isnan(frame_qc)).any():
            try:
                i = qc_labels.index('unknown')
            except ValueError:
                i = len(qc_labels)
                qc_labels.append('unknown')
            frame_qc[missing_name] = i
        frame_qc = frame_qc.astype(np.uint32)  # case to uint
        bad_frames, = np.where(frame_qc != 0)
        # Convert labels to value -> label data frame
        frame_qc_names = pd.DataFrame(list(enumerate(qc_labels)), columns=['qc_values', 'qc_labels'])
        return frame_qc, frame_qc_names, bad_frames

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

    def _meta2ops(self, meta):
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
            'save_path0': str(self.session_path),
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
            'dx': dx.tolist(),
            'dy': dy.tolist()
        }

        return db

    @staticmethod
    def _get_plane_paths(path):
        """Return list of sorted suite2p plane folder paths.

        Parameters
        ----------
        path : pathlib.Path
            The path containing plane folders.

        Returns
        -------
        list of pathlib.Path
            The plane folder paths, ordered by number.
        """
        pattern = re.compile(r'(?<=^plane)\d+$')
        return sorted(path.glob('plane?*'), key=lambda x: int(pattern.search(x.name).group()))

    def bin_per_plane(self, metadata, **kwargs):
        """
        Extracts a binary data file of imaging data per imaging plane.

        Parameters
        ----------
        metadata : dict
            A dictionary of extracted metadata.
        save_path0 : str, pathlib.Path
            The root path of the suite2p bin output.
        save_folder : str
            The subfolder within `save_path0` to save the suite2p bin output.
        kwargs
            Other optional arguments to overwrite the defaults for.

        Returns
        -------
        list of pathlib.Path
            Ordered list of output plane folders containing binary data and ops.
        dict
            Suite2p's modified options.
        """
        import suite2p.io

        options = ('nplanes', 'data_path', 'save_path0', 'save_folder', 'fast_disk', 'batch_size',
                   'nchannels', 'keep_movie_raw', 'look_one_level_down', 'lines', 'dx', 'dy', 'force_sktiff',
                   'do_registration', 'slices')
        ops = self._meta2ops(metadata)
        ops['force_sktiff'] = False
        ops['do_registration'] = True
        ops = {k: v for k, v in ops.items() if k in options}
        ops.update(kwargs)
        ops['save_path0'] = str(ops['save_path0'])  # Path objs must be str for suite2p
        # Update the task kwargs attribute as it will be stored in the arguments json field in alyx
        self.kwargs = ops.copy()

        ret = suite2p.io.mesoscan_to_binary(ops.copy())

        # Get ordered list of plane folders
        out_path = Path(ret['save_path0'], ret['save_folder'])
        assert out_path.exists()
        planes = self._get_plane_paths(out_path)
        assert len(planes) == ret['nplanes']

        return planes, ret

    def image_motion_registration(self, ops):
        """Perform motion registration.

        Parameters
        ----------
        ops : dict
            A dict of suite2p options.

        Returns
        -------
        dict
            A dictionary of registration metrics: "regDX" is nPC x 3, where X[:,0]
            is rigid, X[:,1] is average nonrigid, X[:,2] is max nonrigid shifts;
            "regPC" is average of top and bottom frames for each PC; "tPC" is PC
            across time frames; "reg_metrics_avg" is the average of "regDX";
            "reg_metrics_max" is the maximum of "regDX".

        """
        import suite2p
        ops['do_registration'] = True
        ops['do_regmetrics'] = True
        ops['roidetect'] = False
        ret = suite2p.run_plane(ops)
        metrics = {k: ret.get(k, None) for k in ('regDX', 'regPC', 'tPC')}
        has_metrics = ops['do_regmetrics'] and metrics['regDX'] is not None
        metrics['reg_metrics_avg'] = np.mean(metrics['regDX'], axis=0) if has_metrics else None
        metrics['reg_metrics_max'] = np.max(metrics['regDX'], axis=0) if has_metrics else None
        return metrics

    def roi_detection(self, ops):
        """Perform ROI detection.

        Parameters
        ----------
        ops : dict
            A dict of suite2p options.

        Returns
        -------
        dict
            An updated copy of the ops after running ROI detection.
        """
        import suite2p
        ops['do_registration'] = False
        ops['roidetect'] = True
        ret = suite2p.run_plane(ops)
        return ret

    def _run(self, rename_files=True, use_badframes=True, **kwargs):
        """
        Process inputs, run suite2p and make outputs alf compatible.

        The suite2p processing takes place in a 'suite2p' folder within the session path. After running,
        the data.bin files are moved to 'raw_bin_files' and the rest of the folder is zipped up and moved
        to 'alf/

        Parameters
        ----------
        rename_files: bool
            Whether to rename and reorganize the suite2p outputs to be alf compatible. Defaults is True.
        use_badframes: bool
            Whether to exclude bad frames indicated by the experimenter in badframes.mat.
        overwrite : bool
            Whether to re-perform extraction and motion registration.
        do_registration : bool
            Whether to perform motion registration.
        roidetect : bool
            Whether to perform ROI detection.

        Returns
        -------
        list of pathlib.Path
            All files created by the task.

        """

        """ Metadata and parameters """
        overwrite = kwargs.pop('overwrite', self.overwrite)
        # Load and consolidate the image metadata from JSON files
        metadata, all_meta = self.load_meta_files()

        # Create suite2p output folder in root session path
        raw_image_collections = sorted(self.session_path.glob(f'{self.device_collection}'))
        save_path = self.session_path.joinpath(save_folder := 'suite2p')

        # Check for previous intermediate files
        plane_folders = self._get_plane_paths(save_path)
        if len(plane_folders) > 0:
            for bin_file in save_path.glob('plane?*/imaging.frames_motionRegistered.bin'):
                # If there is a previously registered bin file, rename back to data.bin
                # This file name is hard-coded in suite2p but is renamed in _rename_outputs
                # If a data.bin file already exists, we don't want to overwrite it -
                # it may be from a manual registration or incomplete run
                if not bin_file.with_name('data.bin').exists():
                    bin_file.rename(bin_file.with_stem('data'))

        if len(plane_folders) == 0 or overwrite:
            _logger.info('Extracting tif data per plane')
            # Ingest tiff files
            try:
                plane_folders, _ = self.bin_per_plane(metadata, save_folder=save_folder, save_path0=self.session_path, **kwargs)
            except Exception:
                _logger.error('Exception occurred, cleaning up incomplete suite2p folder')
                # NB: Only remove the suite2p folder if there are no unexpected files in there
                # If the extraction failed due to currupted tiffs, the ops file will not have been created
                if save_path.exists() and set(x.name for x in save_path.rglob('*') if x.is_file()) <= {'data.bin'}:
                    shutil.rmtree(save_path, ignore_errors=True)
                raise  # reraise original exception

        """ Bad frames """
        # exptQC.mat contains experimenter QC values that may not affect ROI detection (e.g. noises, pauses)
        qc_datasets = dataset_from_name('exptQC.mat', self.input_files)
        qc_paths = [next(self.session_path.glob(d.glob_pattern), None) for d in qc_datasets]
        qc_paths = sorted(map(str, filter(None, qc_paths)))
        exptQC = [loadmat(p, squeeze_me=True, simplify_cells=True) for p in qc_paths]
        if len(exptQC) > 0:
            frameQC, frameQC_names, _ = self._consolidate_exptQC(exptQC)
        else:
            _logger.warning('No frame QC (exptQC.mat) files found.')
            frameQC = np.array([], dtype='u1')
            frameQC_names = pd.DataFrame(columns=['qc_values', 'qc_labels'])

        # If applicable, save as bad_frames.npy in first raw_imaging_folder for suite2p
        # badframes.mat contains QC values that do affect ROI detection (e.g. no PMT, lens artefacts)
        badframes = np.array([], dtype='uint32')
        total_frames = 0
        # Ensure all indices are relative to total cumulative frames
        for m, collection in zip(all_meta, raw_image_collections):
            badframes_path = self.session_path.joinpath(collection, 'badframes.mat')
            if badframes_path.exists():
                raw_mat = loadmat(badframes_path, squeeze_me=True, simplify_cells=True)
                badframes = np.r_[badframes, raw_mat['badframes'].astype('uint32') + total_frames]
            total_frames += m['nFrames']
        if len(badframes) > 0 and use_badframes is True:
            # The badframes array should always be a subset of the frameQC array
            assert np.max(badframes) < frameQC.size and np.all(frameQC[badframes])
            np.save(raw_image_collections[0].joinpath('bad_frames.npy'), badframes)
            
        """ Suite2p """
        # Create alf if is doesn't exist
        self.session_path.joinpath('alf').mkdir(exist_ok=True)

        # Perform registration
        if kwargs.get('do_registration', True):
            _logger.info('Performing registration')
            for plane in plane_folders:
                ops = np.load(plane.joinpath('ops.npy'), allow_pickle=True).item()
                ops.update(kwargs)
                # (ops['do_registration'], ops['reg_file'], ops['meanImg'])
                _ = self.image_motion_registration(ops)
                # TODO Handle metrics and QC here

        # ROI detection
        if kwargs.get('roidetect', True):
            _logger.info('Performing ROI detection')
            for plane in plane_folders:
                ops = np.load(plane.joinpath('ops.npy'), allow_pickle=True).item()
                ops.update(kwargs)
                self.roi_detection(ops)

        """ Outputs """
        # Save and rename other outputs
        if rename_files:
            self._rename_outputs(save_path, frameQC_names, frameQC)
            # Only return output file that are in the signature (for registration)
            out_files = chain.from_iterable(map(lambda x: x.find_files(self.session_path)[1], self.output_files))
        else:
            out_files = save_path.rglob('*.*')  # Output all files in suite2p folder

        return list(out_files)


class MesoscopeSync(base_tasks.MesoscopeTask):
    """Extract the frame times from the main DAQ."""

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        I = ExpectedDataset.input  # noqa
        signature = {
            'input_files': [I(f'_{self.sync_namespace}_DAQdata.raw.npy', self.sync_collection, True),
                            I(f'_{self.sync_namespace}_DAQdata.timestamps.npy', self.sync_collection, True),
                            I(f'_{self.sync_namespace}_DAQdata.meta.json', self.sync_collection, True),
                            I('_ibl_rawImagingData.meta.json', self.device_collection, True, unique=False),
                            I('rawImagingData.times_scanImage.npy', self.device_collection, True, True, unique=False),
                            I(f'_{self.sync_namespace}_softwareEvents.log.htsv', self.sync_collection, False), ],
            'output_files': [('mpci.times.npy', 'alf/FOV*', True),
                             ('mpciStack.timeshift.npy', 'alf/FOV*', True),]
        }
        return signature

    def _run(self, **kwargs):
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
        all_collections = flatten(map(lambda x: x.identifiers, self.input_files))[::3]
        collections = set(filter(lambda x: fnmatch(x, self.device_collection), all_collections))
        # Load first meta data file to determine the number of FOVs
        # Changing FOV between imaging bouts is not supported currently!
        self.rawImagingData = alfio.load_object(self.session_path / next(iter(collections)), 'rawImagingData')
        self.rawImagingData['meta'] = mesoscope.patch_imaging_meta(self.rawImagingData['meta'])
        n_FOVs = len(self.rawImagingData['meta']['FOV'])
        sync, chmap = self.load_sync()  # Extract sync data from raw DAQ data
        legacy = kwargs.get('legacy', False)  # this option may be removed in the future once fully tested
        mesosync = mesoscope.MesoscopeSyncTimeline(self.session_path, n_FOVs)
        _, out_files = mesosync.extract(
            save=True, sync=sync, chmap=chmap, device_collection=collections, events=events, use_volume_counter=legacy)
        return out_files

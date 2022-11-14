"""The widefield data extraction pipeline.

The widefield pipeline requires task data extraction using the FPGA (ephys_preprocessing),
optogenetics, camera extraction and widefield image data compression, SVD and correction.

Pipeline:
    1. Data renamed to be ALF-compliant and symlinks created with old names for use by wfield
    2. Raw image data is compressed
    3. Renamed and compressed files are registered to Alyx, imaging snapshots attached as Alyx notes
    4. Preprocessing run to produce
"""
import logging
from pathlib import Path

from ibllib.io.extractors.widefield import Widefield as WidefieldExtractor
from ibllib.pipes import base_tasks
from ibllib.io.video import get_video_meta
from ibllib.plots.snapshot import ReportSnapshot

import labcams.io

_logger = logging.getLogger(__name__)


class WidefieldRegisterRaw(base_tasks.WidefieldTask, base_tasks.RegisterRawDataTask):

    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('dorsal_cortex_landmarks.json', self.device_collection, False),
                            ('*.camlog', self.device_collection, True),
                            ('widefield_wiring.htsv', self.device_collection, False)],
            'output_files': [('widefieldLandmarks.dorsalCortex.json', 'alf/widefield', False),
                             ('widefieldEvents.raw.camlog', self.device_collection, True),
                             ('widefieldChannels.wiring.htsv', self.device_collection, False)]
        }
        return signature

    def _run(self, symlink_old=True):
        out_files = super()._run(symlink_old=True)
        self.register_snapshots()
        return out_files

    def register_snapshots(self, unlink=False):
        """
        Register any photos in the snapshots folder to the session. Typically user will take photo of dorsal cortex before
        and after session

        Returns
        -------

        """
        snapshots_path = self.session_path.joinpath('raw_widefield_data', 'snapshots')
        if not snapshots_path.exists():
            return

        eid = self.one.path2eid(self.session_path, query_type='remote')
        if not eid:
            _logger.warning('Failed to upload snapshots: session not found on Alyx')
            return
        note = dict(user=self.one.alyx.user, content_type='session', object_id=eid, text='')

        notes = []
        for snapshot in snapshots_path.glob('*.tif'):
            with open(snapshot, 'rb') as img_file:
                files = {'image': img_file}
                notes.append(self.one.alyx.rest('notes', 'create', data=note, files=files))
            if unlink:
                snapshot.unlink()
        if unlink and next(snapshots_path.rglob('*'), None) is None:
            snapshots_path.rmdir()


class WidefieldCompress(base_tasks.WidefieldTask):

    priority = 90
    job_size = 'large'

    @property
    def signature(self):
        signature = {
            'input_files': [('*.dat', self.device_collection, True)],
            'output_files': [('imaging.frames.mov', self.device_collection, True)]
        }
        return signature

    def _run(self, remove_uncompressed=False, verify_output=True, **kwargs):
        # Find raw data dat file
        filename, collection, _ = self.input_files[0]
        filepath = next(self.session_path.rglob(str(Path(collection).joinpath(filename))))

        # Construct filename for compressed video
        out_name, out_collection, _ = self.output_files[0]
        output_file = self.session_path.joinpath(out_collection, out_name)
        # Compress to mov
        stack = labcams.io.mmap_dat(str(filepath))
        labcams.io.stack_to_mj2_lossless(stack, str(output_file), rate=30)

        assert output_file.exists(), 'Failed to compress data: no output file found'

        if verify_output:
            meta = get_video_meta(output_file)
            assert meta.length > 0 and meta.size > 0, f'Video file empty: {output_file}'

        if remove_uncompressed:
            filepath.unlink()

        return [output_file]


#  level 1
class WidefieldPreprocess(base_tasks.WidefieldTask):

    priority = 80
    job_size = 'large'

    @property
    def signature(self):
        signature = {
            'input_files': [('imaging.frames.*', self.device_collection, True),
                            ('widefieldEvents.raw.*', self.device_collection, True)],
            'output_files': [('widefieldChannels.frameAverage.npy', 'alf/widefield', True),
                             ('widefieldU.images.npy', 'alf/widefield', True),
                             ('widefieldSVT.uncorrected.npy', 'alf/widefield', True),
                             ('widefieldSVT.haemoCorrected.npy', 'alf/widefield', True)]
        }
        return signature

    def _run(self, upload_plots=True, **kwargs):
        self.wf = WidefieldExtractor(self.session_path)
        _, out_files = self.wf.extract(save=True, extract_timestamps=False)

        if upload_plots:
            output_plots = []
            if self.wf.data_path.joinpath('hemodynamic_correction.png').exists():
                output_plots.append(self.wf.data_path.joinpath('hemodynamic_correction.png'))
            if self.wf.data_path.joinpath('motion_correction.png').exists():
                output_plots.append(self.wf.data_path.joinpath('motion_correction.png'))

            if len(output_plots) > 0:
                eid = self.one.path2eid(self.session_path)
                snp = ReportSnapshot(self.session_path, eid, one=self.one)
                snp.outputs = output_plots
                snp.register_images(widths=['orig'], function='wfield')

        return out_files

    def tearDown(self):
        super(WidefieldPreprocess, self).tearDown()
        self.wf.remove_files()


class WidefieldSync(base_tasks.WidefieldTask):

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('imaging.frames.mov', self.device_collection, True),
                            ('widefieldEvents.raw.camlog', self.device_collection, True),
                            (f'_{self.sync_namespace}_sync.channels.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.polarities.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.times.npy', self.sync_collection, True)],
            'output_files': [('imaging.times.npy', 'alf/widefield', True),
                             ('imaging.imagingLightSource.npy', 'alf/widefield', True),
                             ('imagingLightSource.properties.htsv', 'alf/widefield', True)]
        }
        return signature

    def _run(self):

        self.wf = WidefieldExtractor(self.session_path)
        save_paths = [self.session_path.joinpath(sig[1], sig[0]) for sig in self.signature['output_files']]
        out_files = self.wf.sync_timestamps(bin_exists=False, save=True, save_paths=save_paths,
                                            sync_collection=self.sync_collection)

        # TODO QC

        return out_files


class WidefieldFOV(base_tasks.WidefieldTask):

    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('widefieldLandmarks.dorsalCortex.json', 'alf/widefield', True),
                            ('widefieldU.images.npy', 'alf/widefield', True),
                            ('widefieldSVT.haemoCorrected.npy', 'alf/widefield', True)],
            'output_files': [('widefieldU.images_atlasTransformed.npy', 'alf/widefield', True),
                             ('widefieldU.brainLocationIds_ccf_2017.npy', 'alf/widefield', True)]
        }

        return signature

    def _run(self):

        outfiles = []

        # from wfield import load_allen_landmarks, SVDStack, atlas_from_landmarks_file
        # from ibllib.atlas.regions import BrainRegions
        # from iblutil.numerical import ismember
        # import numpy as np
        # U = np.load(self.session_path.joinpath('alf/widefield', 'widefieldU.images.npy'))
        # SVT = np.load(self.session_path.joinpath('alf/widefield', 'widefieldSVT.haemoCorrected.npy'))
        # lmark_file = self.session_path.joinpath('alf/widefield', 'widefieldLandmarks.dorsalCortex.json')
        # landmarks = load_allen_landmarks(lmark_file)
        #
        # br = BrainRegions()
        #
        # stack = SVDStack(U, SVT)
        # stack.set_warped(1, M=landmarks['transform'])
        #
        # atlas, area_names, mask = atlas_from_landmarks_file(lmark_file)
        # atlas = atlas.astype(np.int32)
        # wf_ids = np.array([n[0] for n in area_names])
        # allen_ids = np.array([br.acronym2id(n[1].split('_')[0], mapping='Allen-lr', hemisphere=n[1].split('_')[1])[0]
        #                      for n in area_names])
        #
        # atlas_allen = np.zeros_like(atlas)
        # a, b = ismember(atlas, wf_ids)
        # atlas_allen[a] = allen_ids[b]
        #
        # file_U = self.session_path.joinpath('alf/widefield', 'widefieldU.images_atlasTransformed.npy')
        # np.save(file_U, stack.U_warped)
        # outfiles.append(file_U)
        #
        # # Do we save the mask??
        # file_atlas = self.session_path.joinpath('alf/widefield', 'widefieldU.brainLocationIds_ccf_2017.npy')
        # np.save(file_atlas, atlas_allen)
        # outfiles.append(file_atlas)

        return outfiles

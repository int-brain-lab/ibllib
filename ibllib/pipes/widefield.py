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
from collections import OrderedDict
from pathlib import Path

from ibllib.io.extractors.widefield import Widefield as WidefieldExtractor
from ibllib.pipes import tasks
from ibllib.pipes.ephys_preprocessing import (
    EphysPulses, EphysMtscomp, EphysAudio, EphysVideoCompress, EphysVideoSyncQc, EphysTrials, EphysPassive, EphysDLC,
    EphysPostDLC)
from ibllib.oneibl.registration import register_session_raw_data
from ibllib.io.video import get_video_meta

import labcams.io

_logger = logging.getLogger('ibllib')


class WidefieldRegisterRaw(tasks.Task):
    signature = {
        'input_files': [('dorsal_cortex_landmarks.json', 'raw_widefield_data', False),
                        ('*.camlog', 'raw_widefield_data', True),
                        ('widefield_wiring.csv', 'raw_widefield_data', False)],
        'output_files': [('widefieldLandmarks.dorsalCortex.json', 'alf', True),
                         ('widefieldEvents.raw.camlog', 'raw_widefield_data', True),
                         ('widefieldChannels.wiring.csv', 'raw_widefield_data', False)]
    }
    priority = 100

    def _run(self, overwrite=False):
        out_files, _ = register_session_raw_data(self.session_path, one=self.one, dry=True)
        widefield_out_files = self.rename_files(symlink_old=True)
        if len(widefield_out_files) > 0:
            out_files = out_files + widefield_out_files
        self.register_snapshots()
        return out_files

    def rename_files(self, symlink_old=True):
        """
        Rename the raw widefield data for a given session so that the data can be registered to Alyx.
        Keeping the old file names as symlinks is useful for preprocessing with the `wfield` module.

        Parameters
        ----------
        symlink_old : bool
            If True, create symlinks with the old filenames.

        """
        session_path = Path(self.session_path).joinpath('raw_widefield_data')
        if not session_path.exists():
            _logger.warning(f'Path does not exist: {session_path}')
            return []
        out_files = []
        for before, after in zip(self.input_files, self.output_files):
            old_file, old_collection, required = before
            old_path = self.session_path.rglob(str(Path(old_collection).joinpath(old_file)))
            old_path = next(old_path, None)
            if not old_path and not required:
                continue

            new_file, new_collection, _ = after
            new_path = self.session_path.joinpath(new_collection, new_file)
            new_path.parent.mkdir(parents=True, exist_ok=True)
            old_path.replace(new_path)
            if symlink_old:
                old_path.symlink_to(new_path)
            out_files.append(new_path)

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


class WidefieldCompress(tasks.Task):
    priority = 40
    level = 0
    force = False
    signature = {
        'input_files': [('*.dat', 'raw_widefield_data', True)],
        'output_files': [('widefield.raw.mov', 'raw_widefield_data', True)]
    }

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
class WidefieldPreprocess(tasks.Task):
    priority = 60
    level = 1
    force = False
    signature = {
        'input_files': [('widefield.raw.*', 'raw_widefield_data', True),
                        ('widefieldEvents.raw.*', 'raw_widefield_data', True)],
        'output_files': [('widefieldChannels.frameAverage.npy', 'alf', True),
                         ('widefieldU.images.npy', 'alf', True),
                         ('widefieldSVT.uncorrected.npy', 'alf', True),
                         ('widefieldSVT.haemoCorrected.npy', 'alf', True)]
    }

    def _run(self, **kwargs):
        self.wf = WidefieldExtractor(self.session_path)
        _, out_files = self.wf.extract(save=True, extract_timestamps=False)
        return out_files

    def tearDown(self):
        super(WidefieldPreprocess, self).tearDown()
        self.wf.remove_files()


class WidefieldSync(tasks.Task):
    priority = 60
    level = 1
    force = False
    signature = {
        'input_files': [('widefield.raw.*', 'raw_widefield_data', True),
                        ('widefieldEvents.raw.*', 'raw_widefield_data', True),
                        ('_spikeglx_sync*.npy', 'raw_ephys_data*', True),
                        ('_spikeglx_sync.polarities*.npy', 'raw_ephys_data*', True),
                        ('_spikeglx_sync.times*.npy', 'raw_ephys_data*', True)],
        'output_files': [('widefield.times.npy', 'alf', True),
                         ('widefield.widefieldLightSource.npy', 'alf', True),
                         ('widefieldLightSource.properties.csv', 'alf', True)]
    }

    def _run(self):

        self.wf = WidefieldExtractor(self.session_path)
        save_paths = [self.session_path.joinpath(sig[1], sig[0]) for sig in self.signature['output_files']]
        out_files = self.wf.sync_timestamps(bin_exists=False, save=True, save_paths=save_paths)

        # TODO QC

        return out_files


class WidefieldFOV(tasks.Task):
    priority = 60
    level = 2
    force = False
    signature = {
        'input_files': [('widefieldLandmarks.dorsalCortex.json', 'alf', True),
                        ('widefieldSVT.uncorrected.npy', 'alf', True),
                        ('widefieldSVT.haemoCorrected.npy', 'alf', True)],
        'output_files': []
    }

    def _run(self):
        # TODO make task that computes location

        return []


# pipeline
class WidefieldExtractionPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(WidefieldExtractionPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        for Task in (WidefieldRegisterRaw, WidefieldCompress, EphysMtscomp, EphysPulses, EphysAudio,
                     EphysVideoCompress):
            task = Task(session_path)
            tasks[task.name] = task
        # level 1
        tasks["EphysTrials"] = EphysTrials(self.session_path, parents=[tasks["EphysPulses"]])
        tasks["EphysPassive"] = EphysPassive(self.session_path, parents=[tasks["EphysPulses"]])
        tasks["WidefieldSync"] = WidefieldSync(self.session_path, parents=[tasks["EphysPulses"]])
        tasks["WidefieldPreprocess"] = WidefieldPreprocess(self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysVideoSyncQc"] = EphysVideoSyncQc(
            self.session_path, parents=[tasks["EphysVideoCompress"], tasks["EphysPulses"], tasks["EphysTrials"]])
        tasks["EphysDLC"] = EphysDLC(self.session_path, parents=[tasks["EphysVideoCompress"]])
        tasks['WidefieldFOV'] = WidefieldFOV(self.session_path, parents=[tasks["WidefieldPreprocess"]])
        # level 3
        tasks["EphysPostDLC"] = EphysPostDLC(self.session_path, parents=[tasks["EphysDLC"]])
        self.tasks = tasks

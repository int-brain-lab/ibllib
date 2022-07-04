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
import spikeglx
from ibllib.io.extractors.ephys_fpga import _sync_to_alf
import one.alf.io as alfio

import labcams.io

_logger = logging.getLogger('ibllib')


class WidefieldRegisterRaw(tasks.Task):
    level = 0
    signature = {
        'input_files': [('dorsal_cortex_landmarks.json', 'raw_widefield_data', False),
                        ('*.camlog', 'raw_widefield_data', True),
                        ('widefield_wiring.csv', 'raw_widefield_data', False),
                        ('*.meta', 'raw_widefield_data', False),
                        ('labcams_configuration.json', 'raw_widefield_data', False)],
        'output_files': [('widefieldLandmarks.dorsalCortex.json', 'alf/widefield', True),
                         ('widefieldEvents.raw.camlog', 'raw_widefield_data', True),
                         ('widefieldChannels.wiring.csv', 'raw_widefield_data', False),
                         ('widefield.raw.nidq.meta', 'raw_widefield_data', False),
                         ('widefield.raw.nidq.wiring.json', 'raw_widefield_data', False)]
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


class WidefieldMtscomp(tasks.Task):
    signature = {
        'input_files': [('*nidq.bin', 'raw_widefield_data', True),
                        ('*nidq.meta', 'raw_widefield_data', True), ],
        'output_files': [('widefield.raw.nidq.cbin', 'raw_widefield_data', True),
                         ('widefield.raw.nidq.ch', 'raw_widefield_data', True),
                         ('widefield.raw.nidq.meta', 'raw_widefield_data', True)]
    }

    def _run(self):

        out_files = []
        # search for .bin files in the raw_widefield_data folder
        files = spikeglx.glob_ephys_file(self.session_path.joinpath('raw_widefield_data'))
        assert len(files) == 1
        bin_file = files[0].get('nidq', None)
        # Rename files (only if they haven't already been renamed)
        if 'widefield.raw.nidq' not in bin_file.stem:
            new_bin_file = bin_file.parent.joinpath('widefield.raw.nidq' + bin_file.suffix)
            bin_file.replace(new_bin_file)
            meta_file = bin_file.with_suffix('.meta')
            new_meta_file = meta_file.parent.joinpath('widefield.raw.nidq.meta')
            meta_file.replace(new_meta_file)
        else:
            new_bin_file = bin_file
            new_meta_file = bin_file.with_suffix('.meta')

        sr = spikeglx.Reader(new_bin_file)
        # Compress files (only if they haven't already been compressed)
        if not sr.is_mtscomp:
            cbin_file = sr.compress_file(keep_original=False)
            ch_file = cbin_file.with_suffix('.ch')
        else:
            cbin_file = new_bin_file
            ch_file = cbin_file.with_suffix('.ch')
            assert cbin_file.suffix == '.cbin'

        out_files.append(cbin_file)
        out_files.append(ch_file)
        out_files.append(new_meta_file)

        return out_files


class WidefieldPulses(tasks.Task):
    signature = {
        'input_files': [('widefield.raw.nidq.meta', 'raw_widefield_data', True),
                        ('widefield.raw.nidq.cbin', 'raw_widefield_data', True),
                        ('widefield.raw.nidq.ch', 'raw_widefield_data', True)],
        'output_files': [('_spikeglx_sync.channels.npy', 'raw_widefield_data', True),
                         ('_spikeglx_sync.polarities.npy', 'raw_widefield_data', True),
                         ('_spikeglx_sync.polarities.times', 'raw_widefield_data', True)]
    }

    def _run(self, overwrite=False):

        # TODO this is replicating a lot of ephys_fpga.extract_sync refactor to make generalisable along with Dynamic pipeline
        syncs = []
        outputs = []

        files = spikeglx.glob_ephys_files(self.session_path.joinpath('raw_widefield_data'))
        assert len(files) == 1
        bin_file = files[0].get('nidq', None)
        if not bin_file:
            return []

        alfname = dict(object='sync', namespace='spikeglx')
        file_exists = alfio.exists(bin_file.parent, **alfname)

        if not overwrite and file_exists:
            _logger.warning(f'Skipping raw sync: SGLX sync found for folder {files[0].label}!')
            sync = alfio.load_object(bin_file.parent, **alfname)
            out_files, _ = alfio._ls(bin_file.parent, **alfname)
        else:
            sr = spikeglx.Reader(bin_file)
            sync, out_files = _sync_to_alf(sr, bin_file.parent, save=True)

        outputs.extend(out_files)
        syncs.extend([sync])

        return outputs, syncs


class WidefieldTrials(EphysTrials):
    signature = {
        'input_files': [('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
                        ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True),
                        ('_spikeglx_sync.channels.*', 'raw_widefield_data', True),
                        ('_spikeglx_sync.polarities.*', 'raw_widefield_data', True),
                        ('_spikeglx_sync.times.*', 'raw_widefield_data', True),
                        ('_iblrig_encoderEvents.raw*', 'raw_behavior_data', True),
                        ('_iblrig_encoderPositions.raw*', 'raw_behavior_data', True),
                        ('*wiring.json', 'raw_widefield_data', False),
                        ('*.meta', 'raw_widefield_data', True)],
        'output_files': [('*trials.choice.npy', 'alf', True),
                         ('*trials.contrastLeft.npy', 'alf', True),
                         ('*trials.contrastRight.npy', 'alf', True),
                         ('*trials.feedbackType.npy', 'alf', True),
                         ('*trials.feedback_times.npy', 'alf', True),
                         ('*trials.firstMovement_times.npy', 'alf', True),
                         ('*trials.goCueTrigger_times.npy', 'alf', True),
                         ('*trials.goCue_times.npy', 'alf', True),
                         ('*trials.intervals.npy', 'alf', True),
                         ('*trials.intervals_bpod.npy', 'alf', True),
                         ('*trials.itiDuration.npy', 'alf', False),
                         ('*trials.probabilityLeft.npy', 'alf', True),
                         ('*trials.response_times.npy', 'alf', True),
                         ('*trials.rewardVolume.npy', 'alf', True),
                         ('*trials.stimOff_times.npy', 'alf', True),
                         ('*trials.stimOn_times.npy', 'alf', True),
                         ('*wheel.position.npy', 'alf', True),
                         ('*wheel.timestamps.npy', 'alf', True),
                         ('*wheelMoves.intervals.npy', 'alf', True),
                         ('*wheelMoves.peakAmplitude.npy', 'alf', True)]
    }

    def _extract_behaviour(self):
        raise NotImplementedError
        # return dsets, out_files

    def get_signatures(self, **kwargs):
        self.input_files = self.signature['input_files']
        self.output_files = self.signature['output_files']


class WidefieldCompress(tasks.Task):
    priority = 40
    level = 0
    force = False
    signature = {
        'input_files': [('*.dat', 'raw_widefield_data', True)],
        'output_files': [('imaging.frames.mov', 'raw_widefield_data', True)]
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
        'input_files': [('imaging.frames.*', 'raw_widefield_data', True),
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
        'input_files': [('imaging.frames.*', 'raw_widefield_data', True),
                        ('widefieldEvents.raw.*', 'raw_widefield_data', True),
                        ('_spikeglx_sync*.npy', 'raw_ephys_data*', True),
                        ('_spikeglx_sync.polarities*.npy', 'raw_ephys_data*', True),
                        ('_spikeglx_sync.times*.npy', 'raw_ephys_data*', True)],
        'output_files': [('imaging.times.npy', 'alf/widefield', True),
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

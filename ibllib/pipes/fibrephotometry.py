"""Extraction tasks for fibrephotometry"""

import logging
from collections import OrderedDict
import traceback
from pathlib import Path
import packaging.version

import numpy as np
import pandas as pd
import labcams.io

import one.alf.io as alfio

from ibllib.misc import check_nvidia_driver
from ibllib.ephys import ephysqc, spikes, sync_probes
from ibllib.io import ffmpeg, spikeglx
from ibllib.io.video import label_from_path
from ibllib.io.extractors.widefield import Widefield as WidefieldExtractor
from ibllib.pipes import tasks
from ibllib.pipes.training_preprocessing import TrainingRegisterRaw as EphysRegisterRaw
from ibllib.pipes.ephys_preprocessing import (
    EphysPulses, EphysMtscomp, EphysAudio, EphysVideoCompress, EphysVideoSyncQc, EphysTrials, EphysPassive, EphysDLC, EphysPostDLC
)
from ibllib.oneibl.registration import register_session_raw_data
from ibllib.pipes.misc import create_alyx_probe_insertions
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.qc.task_metrics import TaskQC

_logger = logging.getLogger('ibllib')


class FibrePhotometryRegisterRaw(tasks.Task):
    # TODO
    signature = {
        'input_files': [('dorsal_cortex_landmarks.json', 'raw_fp_data', True),
                        ('*.camlog', 'raw_fp_data', True)],
        'output_files': [('widefieldLandmarks.dorsalCortex.json', 'raw_fp_data', True),
                         ('widefieldEvents.raw.camlog', 'raw_fp_data', True)]
    }
    priority = 100

    def _run(self, overwrite=False):
        self.rename_files(symlink_old=True)
        self.register_snapshots()
        out_files, _ = register_session_raw_data(self.session_path, one=self.one, dry=True)
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
        session_path = Path(self.session_path).joinpath('raw_fp_data')
        if not session_path.exists():
            _logger.warning(f'Path does not exist: {session_path}')
            return
        for before, after in zip(self.input_files, self.output_files):
            old_file, old_collection, required = before
            old_path = self.session_path.rglob(str(Path(old_collection).joinpath(old_file)))
            old_path = next(old_path, None)
            if not old_path and not required:
                continue

            new_file, new_collection, _ = after
            new_path = self.session_path.joinpath(new_collection, new_file)
            old_path.replace(new_path)
            if symlink_old:
                old_path.symlink_to(new_path)

    def register_snapshots(self, unlink=False):
        """
        Register any photos in the snapshots folder to the session.
        Returns
        -------
        """
        snapshots_path = self.session_path.joinpath('raw_fp_data', 'snapshots')
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


#  level 1
class FibrePhotometryPreprocess(tasks.Task):
    priority = 60
    level = 1
    force = False
    signature = {
        'input_files': [('fpData.raw.*', 'raw_fp_data', True),
                        ('fpData.timestamps.*', 'raw_fp_data', True),
                        ('fpData.channels.*', 'raw_fp_data', True)],
        'output_files': [('*trials.choice.npy', 'alf', True), ]  # TODO
    }

    def _run(self, **kwargs):
        dsets, out_files = FibrePhotometryExtractor(self.session_path).extract(save=True)

        # QC could be run here
        return out_files


def _extract_haemo_corrected():
    U = np.load('U.npy')
    SVT = np.load('SVT.npy')

    frame_rate = 30.  # acquisition rate (2 channels)
    output_folder = None  # write to current directory or path

    from wfield.ncaas import dual_color_hemodymamic_correction

    SVTcorr = dual_color_hemodymamic_correction(U, SVTa, SVTb, frame_rate=frame_rate, output_folder=output_folder);

# pipeline
class FibrePhotometryExtractionPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(WidefieldExtractionPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        for Task in (WidefieldRegisterRaw, WidefieldCompress, EphysPulses, WidefieldPreprocess, EphysAudio, EphysVideoCompress):
            task = Task(session_path)
            tasks[task.name] = task
        # level 1
        tasks["EphysTrials"] = EphysTrials(self.session_path, parents=[tasks["EphysPulses"]])
        tasks["EphysPassive"] = EphysPassive(self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysVideoSyncQc"] = EphysVideoSyncQc(
            self.session_path, parents=[tasks["EphysVideoCompress"], tasks["EphysPulses"], tasks["EphysTrials"]])
        # tasks["EphysCellsQc"] = EphysCellsQc(self.session_path, parents=[tasks["SpikeSorting"]])
        tasks["EphysDLC"] = EphysDLC(self.session_path, parents=[tasks["EphysVideoCompress"]])
        # level 3
        tasks["EphysPostDLC"] = EphysPostDLC(self.session_path, parents=[tasks["EphysDLC"]])
        self.tasks = tasks

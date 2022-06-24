import logging

from ibllib.io import ffmpeg
from ibllib.pipes import base_tasks
from ibllib.io.video import label_from_path
from ibllib.io.extractors import camera
from ibllib.qc.camera import run_all_qc as run_camera_qc

_logger = logging.getLogger('ibllib')


class VideoRegisterRaw(base_tasks.VideoTask, base_tasks.RegisterRawDataTask):
    """
    Task to register raw video data. Builds up list of files to register from list of cameras given in session params file
    """
    cpu = 1
    io_charge = 90
    level = 0
    force = False

    def dynamic_signatures(self):
        input_signatures = []
        output_signatures = [(f'_iblrig_{cam}Camera.timestamps*', self.collection, True) for cam in self.cameras] +\
                            [(f'_iblrig_{cam}Camera.GPIO.bin', self.collection, True) for cam in self.cameras] +\
                            [(f'_iblrig_{cam}Camera.frame_counter.bin', self.collection, True) for cam in self.cameras] + \
                            [(f'_iblrig_{cam}Camera.frameData.bin', self.collection, False) for cam in self.cameras] + \
                            [('_iblrig_videoCodeFiles.raw*', self.collection, False)]

        return input_signatures, output_signatures


class VideoCompress(base_tasks.VideoTask):
    """
    Task to compress raw video data from .avi to .mp4 format.
    """
    priority = 90
    level = 0
    force = False

    def dynamic_signatures(self):
        input_signatures = (f'_iblrig_{self.camera}Camera.raw.*', self.collection, True),
        output_signatures = [(f'_iblrig_{cam}Camera.raw.mp4', self.collection, True) for cam in self.cameras]

        return input_signatures, output_signatures

    def _run(self):
        # avi to mp4 compression
        command = ('ffmpeg -i {file_in} -y -nostdin -codec:v libx264 -preset slow -crf 17 '
                   '-loglevel 0 -codec:a copy {file_out}')
        output_files = ffmpeg.iblrig_video_compression(self.session_path, command)

        if len(output_files) == 0:
            _logger.info('No compressed videos found')
            return

        return output_files


class VideoSyncQc(base_tasks.VideoTask):
    """
    Task to sync camera timestamps to main DAQ timestamps
    N.B Signatures only reflect new daq naming convention, non compatible with ephys when not running on server
    """
    priority = 40
    level = 2
    force = True

    def dynamic_signatures(self):
        input_signatures = [(f'_iblrig_{cam}Camera.raw.mp4', self.device_collection, True) for cam in self.cameras] +\
                           [(f'_iblrig_{cam}Camera.timestamps*', self.device_collection, False) for cam in self.cameras] +\
                           [(f'_iblrig_{cam}Camera.GPIO.bin', self.device_collection, False) for cam in self.cameras] +\
                           [(f'_iblrig_{cam}Camera.frame_counter.bin', self.device_collection, False) for cam in self.cameras] + \
                           [(f'_iblrig_{cam}Camera.frameData.bin', self.device_collection, False) for cam in self.cameras] + \
                           [('_iblrig_taskData.raw.*', self.main_task_collection, True),
                            ('_iblrig_taskSettings.raw.*', self.main_task_collection, True),
                            ('*wheel.position.npy', 'alf', False),
                            ('*wheel.timestamps.npy', 'alf', False)]

        if self.sync == 'nidq':
            input_signatures += [('_spikeglx_sync.channels.npy', self.sync_collection, True),
                                 ('_spikeglx_sync.polarities.npy', self.sync_collection, True),
                                 ('_spikeglx_sync.times.npy', self.sync_collection, True),
                                 ('daq.wiring.json', self.sync_collection, True)]

        output_signatures = [(f'_iblrig_{cam}Camera.times.npy', 'alf', True) for cam in self.cameras]

        return input_signatures, output_signatures

    def _run(self, **kwargs):

        mp4_files = self.session_path.joinpath(self.device_collection).rglob('*.mp4')
        labels = [label_from_path(x) for x in mp4_files]

        # Video timestamps extraction
        output_files = []
        data, files = camera.extract_all(self.session_path, sync_type=self.sync, sync_collection=self.sync_collection,
                                         save=True, labels=labels)
        output_files.extend(files)

        # Video QC
        run_camera_qc(self.session_path, update=True, one=self.one, cameras=labels)

        return output_files

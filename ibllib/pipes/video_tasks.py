import logging
import subprocess

from ibllib.io import ffmpeg
from ibllib.pipes import base_tasks
from ibllib.io.video import label_from_path, get_video_meta
from ibllib.io.extractors import camera
from ibllib.qc.camera import run_all_qc as run_camera_qc

_logger = logging.getLogger('ibllib')


class VideoRegisterRaw(base_tasks.VideoTask, base_tasks.RegisterRawDataTask):
    """
    Task to register raw video data. Builds up list of files to register from list of cameras given in session params file
    """

    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [],
            'output_files':
                [(f'_iblrig_{cam}Camera.timestamps*', self.device_collection, False) for cam in self.cameras] +
                [(f'_iblrig_{cam}Camera.GPIO.bin', self.device_collection, False) for cam in self.cameras] +
                [(f'_iblrig_{cam}Camera.frame_counter.bin', self.device_collection, False) for cam in self.cameras] +
                [(f'_iblrig_{cam}Camera.frameData.bin', self.device_collection, False) for cam in self.cameras] +
                [('_iblrig_videoCodeFiles.raw*', self.device_collection, False)]
        }
        return signature

    def assert_expected_outputs(self, raise_error=True):
        """
        frameData replaces the timestamps file. Therefore if frameData is present, timestamps is
        optional and vice versa.
        """
        assert self.status == 0
        _logger.info('Checking output files')
        everything_is_fine, files = self.assert_expected(self.output_files)

        required = any('Camera.frameData' in x or 'Camera.timestamps' in x for x in map(str, files))
        if not (everything_is_fine and required):
            for out in self.outputs:
                _logger.error(f"{out}")
            if raise_error:
                raise FileNotFoundError("Missing outputs after task completion")

        return everything_is_fine, files


class VideoCompress(base_tasks.VideoTask):
    """
    Task to compress raw video data from .avi to .mp4 format.
    """
    priority = 90
    job_size = 'large'

    @property
    def signature(self):
        signature = {
            'input_files': [(f'_iblrig_{cam}Camera.raw.*', self.device_collection, True) for cam in self.cameras],
            'output_files': [(f'_iblrig_{cam}Camera.raw.mp4', self.device_collection, True) for cam in self.cameras]
        }
        return signature

    def _run(self):
        # TODO different compression parameters based on whether it is training or not based on number of cameras?
        # avi to mp4 compression
        if self.sync == 'bpod':
            command = ('ffmpeg -i {file_in} -y -nostdin -codec:v libx264 -preset slow -crf 29 '
                       '-nostats -codec:a copy {file_out}')
        else:
            command = ('ffmpeg -i {file_in} -y -nostdin -codec:v libx264 -preset slow -crf 17 '
                       '-loglevel 0 -codec:a copy {file_out}')

        output_files = ffmpeg.iblrig_video_compression(self.session_path, command)

        if len(output_files) == 0:
            _logger.info('No compressed videos found')
            return

        return output_files


class VideoConvert(base_tasks.VideoTask):
    """
    Task that converts compressed avi to mp4 format and renames video and camlog files. Specific to UCLA widefield implementation
    """
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [(f'{cam}_cam*.avi', self.device_collection, True) for cam in self.cameras] +
                           [(f'{cam}_cam*.camlog', self.device_collection, False) for cam in self.cameras],
            'output_files': [(f'_iblrig_{cam}Camera.raw.mp4', self.device_collection, True) for cam in self.cameras] +
                            [(f'_iblrig_{cam}Camera.raw.camlog', self.device_collection, True) for cam in self.cameras]
        }

        return signature

    def _run(self):
        output_files = []
        for cam in self.cameras:

            # rename and register the camlog files
            camlog_file = next(self.session_path.joinpath(self.device_collection).glob(f'{cam}_cam*.camlog'))
            new_camlog_file = self.session_path.joinpath(self.device_collection, f'_iblrig_{cam}Camera.raw.camlog')
            camlog_file.replace(new_camlog_file)
            output_files.append(new_camlog_file)

            # convert the avi files to mp4
            avi_file = next(self.session_path.joinpath(self.device_collection).glob(f'{cam}_cam*.avi'))
            mp4_file = self.session_path.joinpath(self.device_collection, f'_iblrig_{cam}Camera.raw.mp4')
            command2run = f"ffmpeg -i {str(avi_file)} -c:v copy -c:a copy -y {str(mp4_file)}"

            process = subprocess.Popen(command2run, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            info, error = process.communicate()
            if process.returncode == 0:
                # check the video meta matched and remove the original file
                meta_avi = get_video_meta(avi_file)
                _ = meta_avi.pop('size')
                meta_mp4 = get_video_meta(mp4_file)
                match = True
                for key in meta_avi.keys():
                    if meta_avi[key] != meta_mp4[key]:
                        match = False

                # if all checks out we can remove the original avi
                if match:
                    avi_file.unlink()
                    output_files.append(mp4_file)
                else:
                    _logger.error(f'avi and mp4 meta data do not match for {avi_file}')
            else:
                _logger.error(f'conversion to mp4 failed for {avi_file}: {error}')

        return output_files


class VideoSyncQcCamlog(base_tasks.VideoTask):
    """
    Task to sync camera timestamps to main DAQ timestamps when camlog files are used. Specific to UCLA widefield implementation
    """
    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [(f'_iblrig_{cam}Camera.raw.mp4', self.device_collection, True) for cam in self.cameras] +
                           [(f'_iblrig_{cam}Camera.raw.camlog', self.device_collection, False) for cam in self.cameras] +
                           [(f'_{self.sync_namespace}_sync.channels.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.polarities.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.times.npy', self.sync_collection, True),
                            ('*.wiring.json', self.sync_collection, True),
                            ('*wheel.position.npy', 'alf', False),
                            ('*wheel.timestamps.npy', 'alf', False)],
            'output_files': [(f'_ibl_{cam}Camera.times.npy', 'alf', True) for cam in self.cameras]
        }

        return signature

    def _run(self, qc=True, **kwargs):

        mp4_files = self.session_path.joinpath(self.device_collection).rglob('*.mp4')
        labels = [label_from_path(x) for x in mp4_files]

        # Video timestamps extraction
        output_files = []
        data, files = camera.extract_all(self.session_path, sync_type=self.sync, sync_collection=self.sync_collection,
                                         save=True, labels=labels, camlog=True)
        output_files.extend(files)

        # Video QC
        if qc:
            run_camera_qc(self.session_path, update=True, one=self.one, cameras=labels, camlog=True,
                          sync_collection=self.sync_collection, sync_type=self.sync)

        return output_files


class VideoSyncQcBpod(base_tasks.VideoTask):
    """
    Task to sync camera timestamps to main DAQ timestamps
    N.B Signatures only reflect new daq naming convention, non compatible with ephys when not running on server
    """
    priority = 40
    job_size = 'small'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Task collection (this needs to be specified in the task kwargs)
        self.collection = self.get_task_collection(kwargs.get('collection', None))
        # Task type (protocol)
        self.protocol = self.get_protocol(kwargs.get('protocol', None), task_collection=self.collection)

    @property
    def signature(self):
        signature = {
            'input_files': [(f'_iblrig_{cam}Camera.raw.mp4', self.device_collection, True) for cam in self.cameras] +
                           [(f'_iblrig_{cam}Camera.timestamps*', self.device_collection, False) for cam in self.cameras] +
                           [(f'_iblrig_{cam}Camera.GPIO.bin', self.device_collection, False) for cam in self.cameras] +
                           [(f'_iblrig_{cam}Camera.frame_counter.bin', self.device_collection, False) for cam in self.cameras] +
                           [(f'_iblrig_{cam}Camera.frameData.bin', self.device_collection, False) for cam in self.cameras] +
                           [('_iblrig_taskData.raw.*', self.collection, True),
                            ('_iblrig_taskSettings.raw.*', self.collection, True),
                            ('*wheel.position.npy', 'alf', False),
                            ('*wheel.timestamps.npy', 'alf', False)],
            'output_files': [(f'_ibl_{cam}Camera.times.npy', 'alf', True) for cam in self.cameras]
        }

        return signature

    def _run(self, **kwargs):

        mp4_files = self.session_path.joinpath(self.device_collection).rglob('*.mp4')
        labels = [label_from_path(x) for x in mp4_files]

        # Video timestamps extraction
        output_files = []
        data, files = camera.extract_all(self.session_path, sync_type=self.sync, sync_collection=self.sync_collection,
                                         save=True, labels=labels)
        output_files.extend(files)

        # Video QC
        run_camera_qc(self.session_path, update=True, one=self.one, cameras=labels,
                      sync_collection=self.sync_collection, sync_type=self.sync)

        return output_files


class VideoSyncQcNidq(base_tasks.VideoTask):
    """
    Task to sync camera timestamps to main DAQ timestamps
    N.B Signatures only reflect new daq naming convention, non compatible with ephys when not running on server
    """
    priority = 40
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [(f'_iblrig_{cam}Camera.raw.mp4', self.device_collection, True) for cam in self.cameras] +
                           [(f'_iblrig_{cam}Camera.timestamps*', self.device_collection, False) for cam in self.cameras] +
                           [(f'_iblrig_{cam}Camera.GPIO.bin', self.device_collection, False) for cam in self.cameras] +
                           [(f'_iblrig_{cam}Camera.frame_counter.bin', self.device_collection, False) for cam in self.cameras] +
                           [(f'_iblrig_{cam}Camera.frameData.bin', self.device_collection, False) for cam in self.cameras] +
                           [(f'_{self.sync_namespace}_sync.channels.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.polarities.npy', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.times.npy', self.sync_collection, True),
                            ('*.wiring.json', self.sync_collection, True),
                            ('*wheel.position.npy', 'alf', False),
                            ('*wheel.timestamps.npy', 'alf', False),
                            ('*experiment.description*', '', False)],
            'output_files': [(f'_ibl_{cam}Camera.times.npy', 'alf', True) for cam in self.cameras]
        }

        return signature

    def _run(self, **kwargs):

        mp4_files = self.session_path.joinpath(self.device_collection).glob('*.mp4')
        labels = [label_from_path(x) for x in mp4_files]

        # Video timestamps extraction
        output_files = []
        data, files = camera.extract_all(self.session_path, sync_type=self.sync, sync_collection=self.sync_collection,
                                         save=True, labels=labels)
        output_files.extend(files)

        # Video QC
        run_camera_qc(self.session_path, update=True, one=self.one, cameras=labels,
                      sync_collection=self.sync_collection, sync_type=self.sync)

        return output_files

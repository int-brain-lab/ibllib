import time
import logging
import subprocess
import traceback
import cv2
from pathlib import Path

from ibllib.io import ffmpeg, raw_daq_loaders
from ibllib.pipes import base_tasks
from ibllib.io.video import label_from_path, get_video_meta
from ibllib.io.extractors import camera
from ibllib.qc.camera import run_all_qc as run_camera_qc
from ibllib.misc import check_nvidia_driver

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

        kwargs = {}
        if self.sync_namespace == 'timeline':
            # Load sync from timeline file
            alf_path = self.session_path / self.sync_collection
            kwargs['sync'], kwargs['chmap'] = raw_daq_loaders.load_timeline_sync_and_chmap(alf_path)

        # Video timestamps extraction
        output_files = []
        data, files = camera.extract_all(self.session_path, sync_type=self.sync, sync_collection=self.sync_collection,
                                         save=True, labels=labels, **kwargs)
        output_files.extend(files)

        # Video QC
        run_camera_qc(self.session_path, update=True, one=self.one, cameras=labels,
                      sync_collection=self.sync_collection, sync_type=self.sync)

        return output_files


class LightningPose(base_tasks.VideoTask):
    # TODO: make one task per cam? Or how to figure out the number of cameras (self.cameras)?
    gpu = 1
    io_charge = 100
    level = 2
    force = True
    job_size = 'large'

    env = Path.home().joinpath('Documents', 'PYTHON', 'envs', 'litpose', 'bin', 'activate')
    scripts = Path.home().joinpath('Documents', 'PYTHON', 'iblscripts', 'deploy', 'serverpc', 'litpose')

    @property
    def signature(self):
        signature = {
            'input_files': [(f'_iblrig_{cam}Camera.raw.mp4', self.device_collection, True) for cam in self.cameras],
            'output_files': [(f'_ibl_{cam}Camera.lightningPose.pqt', 'alf', True) for cam in self.cameras]
        }

        return signature

    @staticmethod
    def _video_intact(file_mp4):
        """Checks that the downloaded video can be opened and is not empty"""
        cap = cv2.VideoCapture(str(file_mp4))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        intact = True if frame_count > 0 else False
        cap.release()
        return intact

    def _check_env(self):
        """Check that scripts are present, env can be activated and get iblvideo version"""
        assert len(list(self.scripts.rglob('run_litpose.*'))) == 2, \
            f'Scripts run_litpose.sh and run_litpose.py do not exist in {self.scripts}'
        assert self.env.exists(), f"environment does not exist in assumed location {self.env}"
        command2run = f"source {self.env}; python -c 'import iblvideo; print(iblvideo.__version__)'"
        process = subprocess.Popen(
            command2run,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            executable="/bin/bash"
        )
        info, error = process.communicate()
        if process.returncode != 0:
            raise AssertionError(f"environment check failed\n{error.decode('utf-8')}")
        version = info.decode("utf-8").strip().split('\n')[-1]
        return version

    def _run(self, overwrite=True, **kwargs):

        # Gather video files
        self.session_path = Path(self.session_path)
        mp4_files = [self.session_path.joinpath(self.device_collection) for cam in self.cameras if
                     self.session_path.joinpath(self.device_collection, f'_iblrig_{cam}Camera.raw.mp4').exists()]

        labels = [label_from_path(x) for x in mp4_files]
        _logger.info(f'Running on {labels} videos')

        # Check the environment
        self.version = self._check_env()
        _logger.info(f'iblvideo version {self.version}')

        # If all results exist and overwrite is False, skip computation
        expected_outputs_present, expected_outputs = self.assert_expected(self.output_files, silent=True)
        if overwrite is False and expected_outputs_present is True:
            actual_outputs = expected_outputs
            return actual_outputs

        # Else, loop over videos
        actual_outputs = []
        for label, mp4_file in zip(labels, mp4_files):
            # Catch exceptions so that the other cams can still run but set status to Errored
            try:
                # Check that the GPU is (still) accessible
                check_nvidia_driver()
                # Check that the video can be loaded
                if not self._video_intact(mp4_file):
                    _logger.error(f"Corrupt raw video file {mp4_file}")
                    self.status = -1
                    continue
                t0 = time.time()
                _logger.info(f'Running Ligthning Pose on {label}Camera.')
                command2run = f"{self.scripts.joinpath('run_litpose.sh')} {str(self.env)} {mp4_file} {overwrite}"
                _logger.info(command2run)
                process = subprocess.Popen(
                    command2run,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    executable="/bin/bash",
                )
                info, error = process.communicate()
                if process.returncode != 0:
                    error_str = error.decode("utf-8").strip()
                    _logger.error(f'Lightning pose failed for {label}Camera.\n\n'
                                  f'++++++++ Output of subprocess for debugging ++++++++\n\n'
                                  f'{error_str}\n'
                                  f'++++++++++++++++++++++++++++++++++++++++++++\n')
                    self.status = -1
                    continue
                else:
                    _logger.info(f'{label} camera took {(time.time() - t0)} seconds')
                    result = next(self.session_path.joinpath('alf').glob(f'_ibl_{label}Camera.lightningPose*.pqt'))
                    actual_outputs.append(result)

            except BaseException:
                _logger.error(traceback.format_exc())
                self.status = -1
                continue

        return actual_outputs

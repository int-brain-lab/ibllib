from pathlib import Path
import numpy as np

from ibllib.io import ffmpeg
from ibllib.pipes import tasks


class VideoRegisterRaw(tasks.Task):
    cpu = 1
    io_charge = 90
    level = 0
    force = False
    signature = {
        'input_files': [],
        'output_files': [('_iblrig_*Camera.timestamps*', 'raw_video_data', True),
                         ('_iblrig_*Camera.frameData.bin', 'raw_video_data', False),
                         ('_iblrig_*Camera.GPIO.bin', 'raw_video_data', True),
                         ('_iblrig_*Camera.frame_counter.bin', 'raw_video_data', True),
                         ('_iblrig_videoCodeFiles.raw*', 'raw_video_data', False)]}

    def _run(self):
        out_files = []
        for file_sig in self.output_files:
            file_name, collection, required = file_sig
            file_path = self.session_path.rglob(str(Path(collection).joinpath(file_name)))
            file_path = next(file_path, None)
            if not file_path and not required:
                continue
            out_files.append(file_path)

        return out_files

    def get_signatures(self):

        output_files = Path(self.session_path).joinpath('raw_video_data').glob('*')
        labels = np.unique([label_from_path(x) for x in output_files])

        self.input_files = self.signature['input_files']

        full_output_files = []
        for sig in self.signature['output_files']:
            if 'Camera' in sig[0]:
                for label in labels:
                    full_output_files.append((sig[0].replace('*Camera', f'{label}Camera'), sig[1], sig[2]))

        self.output_files = full_output_files


class VideoCompress(tasks.Task):
    priority = 90
    level = 0
    force = False
    signature = {
        'input_files': [('_iblrig_*Camera.raw.*', 'raw_video_data', True)],
        'output_files': [('_iblrig_*Camera.raw.mp4', 'raw_video_data', True)]
    }

    def _run(self, **kwargs):
        # avi to mp4 compression
        command = ('ffmpeg -i {file_in} -y -nostdin -codec:v libx264 -preset slow -crf 17 '
                   '-loglevel 0 -codec:a copy {file_out}')
        output_files = ffmpeg.iblrig_video_compression(self.session_path, command)

        if len(output_files) == 0:
            _logger.info('No compressed videos found')
            return

        return output_files

    def get_signatures(self, **kwargs):
        # Detect the number of cameras
        output_files = Path(self.session_path).joinpath('raw_video_data').glob('*')
        labels = np.unique([label_from_path(x) for x in output_files])

        full_input_files = []
        for sig in self.signature['input_files']:
            for label in labels:
                full_input_files.append((sig[0].replace('*Camera', f'{label}Camera'), sig[1], sig[2]))

        self.input_files = full_input_files

        full_output_files = []
        for sig in self.signature['output_files']:
            for label in labels:
                full_output_files.append((sig[0].replace('*Camera', f'{label}Camera'), sig[1], sig[2]))

        self.output_files = full_output_files


# TODO # How to deal with these when we don't know what we doing i.e when we don't know what sync we are using
class VideoSyncQc(tasks.Task):
    priority = 40
    level = 2
    force = True
    signature = {
        'input_files': [('_iblrig_*Camera.raw.mp4', 'raw_video_data', True),
                        ('_iblrig_*Camera.timestamps*', 'raw_video_data', False),
                        ('_iblrig_*Camera.frameData.bin', 'raw_video_data', False),
                        ('_iblrig_*Camera.GPIO.bin', 'raw_video_data', False),
                        ('_iblrig_*Camera.frame_counter.bin', 'raw_video_data', False),
                        ('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
                        ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True),
                        ('_spikeglx_sync.channels.*', 'raw_ephys_data*', True),
                        ('_spikeglx_sync.polarities.*', 'raw_ephys_data*', True), # How to deal with these when we don't know what we doing
                        ('_spikeglx_sync.times.*', 'raw_ephys_data*', True),
                        ('*wheel.position.npy', 'alf', False),
                        ('*wheel.timestamps.npy', 'alf', False),
                        ('*wiring.json', 'raw_ephys_data*', False),
                        ('*.meta', 'raw_ephys_data*', True)],

        'output_files': [('_ibl_*Camera.times.npy', 'alf', True)]
    }

    def _run(self, **kwargs):

        mp4_files = self.session_path.joinpath('raw_video_data').rglob('*.mp4')
        labels = [label_from_path(x) for x in mp4_files]

        # Video timestamps extraction
        output_files = []
        data, files = camera.extract_all(self.session_path, save=True, labels=labels)
        output_files.extend(files)

        # Video QC
        run_camera_qc(self.session_path, update=True, one=self.one, cameras=labels)

        return output_files

        # TODO This is what the training had, does the stream matter?
        # Video timestamps extraction
        # data, files = camera.extract_all(self.session_path, save=True, video_path=output_files[0])
        # output_files.extend(files)
#
        # # Video QC
        # CameraQC(self.session_path, 'left', one=self.one, stream=False).run(update=True)
        # return output_files


    def get_signatures(self, **kwargs):
        neuropixel_version = spikeglx.get_neuropixel_version_from_folder(self.session_path)
        probes = spikeglx.get_probes_from_folder(self.session_path)
        # need to detect the number of cameras
        output_files = Path(self.session_path).joinpath('raw_video_data').rglob('*')
        labels = np.unique([label_from_path(x) for x in output_files])

        full_input_files = []
        for sig in self.signature['input_files']:
            if 'raw_ephys_data*' in sig[1]:
                if neuropixel_version != '3A':
                    full_input_files.append((sig[0], 'raw_ephys_data', sig[2]))
                for probe in probes:
                    full_input_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))
            elif 'Camera' in sig[0]:
                for lab in labels:
                    full_input_files.append((sig[0].replace('*Camera', f'{lab}Camera'), sig[1], sig[2]))
            else:
                full_input_files.append((sig[0], sig[1], sig[2]))

        self.input_files = full_input_files

        full_output_files = []
        for sig in self.signature['output_files']:
            if 'raw_ephys_data*' in sig[1]:
                if neuropixel_version != '3A':
                    full_output_files.append((sig[0], 'raw_ephys_data', sig[2]))
                for probe in probes:
                    full_output_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))
            elif 'Camera' in sig[0]:
                for lab in labels:
                    full_output_files.append((sig[0].replace('*Camera', f'{lab}Camera'), sig[1], sig[2]))
            else:
                full_output_files.append((sig[0], sig[1], sig[2]))

        self.output_files = full_output_files


class DLC(tasks.Task):
    """
    This task relies on a correctly installed dlc environment as per
    https://docs.google.com/document/d/1g0scP6_3EmaXCU4SsDNZWwDTaD9MG0es_grLA-d0gh0/edit#

    If your environment is set up otherwise, make sure that you set the respective attributes:
    t = EphysDLC(session_path)
    t.dlcenv = Path('/path/to/your/dlcenv/bin/activate')
    t.scripts = Path('/path/to/your/iblscripts/deploy/serverpc/dlc')
    """
    gpu = 1
    cpu = 4
    io_charge = 90
    level = 2
    force = True

    dlcenv = Path.home().joinpath('Documents', 'PYTHON', 'envs', 'dlcenv', 'bin', 'activate')
    scripts = Path.home().joinpath('Documents', 'PYTHON', 'iblscripts', 'deploy', 'serverpc', 'dlc')
    signature = {
        'input_files': [
            ('_iblrig_*Camera.raw.mp4', 'raw_video_data', True)],
        'output_files': [
            ('_ibl_*Camera.dlc.pqt', 'alf', True),
            ('*Camera.ROIMotionEnergy.npy', 'alf', True),
            ('*ROIMotionEnergy.position.npy', 'alf', True)],
    }

    def _check_dlcenv(self):
        """Check that scripts are present, dlcenv can be activated and get iblvideo version"""
        assert len(list(self.scripts.rglob('run_dlc.*'))) == 2, \
            f'Scripts run_dlc.sh and run_dlc.py do not exist in {self.scripts}'
        assert len(list(self.scripts.rglob('run_motion.*'))) == 2, \
            f'Scripts run_motion.sh and run_motion.py do not exist in {self.scripts}'
        assert self.dlcenv.exists(), f"DLC environment does not exist in assumed location {self.dlcenv}"
        command2run = f"source {self.dlcenv}; python -c 'import iblvideo; print(iblvideo.__version__)'"
        process = subprocess.Popen(
            command2run,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            executable="/bin/bash"
        )
        info, error = process.communicate()
        if process.returncode != 0:
            raise AssertionError(f"DLC environment check failed\n{error.decode('utf-8')}")
        version = info.decode("utf-8").strip().split('\n')[-1]
        return version

    @staticmethod
    def _video_intact(file_mp4):
        """Checks that the downloaded video can be opened and is not empty"""
        cap = cv2.VideoCapture(str(file_mp4))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        intact = True if frame_count > 0 else False
        cap.release()
        return intact

    def _run(self, cams=None, overwrite=False):
        # Default to all three cams
        cams = cams or ['left', 'right', 'body']
        cams = assert_valid_label(cams)
        # Set up
        self.session_id = self.one.path2eid(self.session_path)
        actual_outputs = []

        # Loop through cams
        for cam in cams:
            # Catch exceptions so that following cameras can still run
            try:
                # If all results exist and overwrite is False, skip computation
                expected_outputs_present, expected_outputs = self.assert_expected(self.output_files, silent=True)
                if overwrite is False and expected_outputs_present is True:
                    actual_outputs.extend(expected_outputs)
                    return actual_outputs
                else:
                    file_mp4 = next(self.session_path.joinpath('raw_video_data').glob(f'_iblrig_{cam}Camera.raw*.mp4'))
                    if not file_mp4.exists():
                        # In this case we set the status to Incomplete.
                        _logger.error(f"No raw video file available for {cam}, skipping.")
                        self.status = -3
                        continue
                    if not self._video_intact(file_mp4):
                        _logger.error(f"Corrupt raw video file {file_mp4}")
                        self.status = -1
                        continue
                    # Check that dlc environment is ok, shell scripts exists, and get iblvideo version, GPU addressable
                    self.version = self._check_dlcenv()
                    _logger.info(f'iblvideo version {self.version}')
                    check_nvidia_driver()

                    _logger.info(f'Running DLC on {cam}Camera.')
                    command2run = f"{self.scripts.joinpath('run_dlc.sh')} {str(self.dlcenv)} {file_mp4} {overwrite}"
                    _logger.info(command2run)
                    process = subprocess.Popen(
                        command2run,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        executable="/bin/bash",
                    )
                    info, error = process.communicate()
                    # info_str = info.decode("utf-8").strip()
                    # _logger.info(info_str)
                    if process.returncode != 0:
                        error_str = error.decode("utf-8").strip()
                        _logger.error(f'DLC failed for {cam}Camera.\n\n'
                                      f'++++++++ Output of subprocess for debugging ++++++++\n\n'
                                      f'{error_str}\n'
                                      f'++++++++++++++++++++++++++++++++++++++++++++\n')
                        self.status = -1
                        # We dont' run motion energy, or add any files if dlc failed to run
                        continue
                    dlc_result = next(self.session_path.joinpath('alf').glob(f'_ibl_{cam}Camera.dlc*.pqt'))
                    actual_outputs.append(dlc_result)

                    _logger.info(f'Computing motion energy for {cam}Camera')
                    command2run = f"{self.scripts.joinpath('run_motion.sh')} {str(self.dlcenv)} {file_mp4} {dlc_result}"
                    _logger.info(command2run)
                    process = subprocess.Popen(
                        command2run,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        executable="/bin/bash",
                    )
                    info, error = process.communicate()
                    # info_str = info.decode("utf-8").strip()
                    # _logger.info(info_str)
                    if process.returncode != 0:
                        error_str = error.decode("utf-8").strip()
                        _logger.error(f'Motion energy failed for {cam}Camera.\n\n'
                                      f'++++++++ Output of subprocess for debugging ++++++++\n\n'
                                      f'{error_str}\n'
                                      f'++++++++++++++++++++++++++++++++++++++++++++\n')
                        self.status = -1
                        continue
                    actual_outputs.append(next(self.session_path.joinpath('alf').glob(
                        f'{cam}Camera.ROIMotionEnergy*.npy')))
                    actual_outputs.append(next(self.session_path.joinpath('alf').glob(
                        f'{cam}ROIMotionEnergy.position*.npy')))
            except BaseException:
                _logger.error(traceback.format_exc())
                self.status = -1
                continue
        # If status is Incomplete, check that there is at least one output.
        # # Otherwise make sure it gets set to Empty (outputs = None), and set status to -1 to make sure it doesn't slip
        if self.status == -3 and len(actual_outputs) == 0:
            actual_outputs = None
            self.status = -1
        return actual_outputs

    def get_signatures(self, **kwargs):
        # Detect the number of cameras
        output_files = Path(self.session_path).joinpath('raw_video_data').glob('*')
        labels = np.unique([label_from_path(x) for x in output_files])

        full_input_files = []
        for sig in self.signature['input_files']:
            for label in labels:
                full_input_files.append((sig[0].replace('*Camera', f'{label}Camera'), sig[1], sig[2]))

        self.input_files = full_input_files

        full_output_files = []
        for sig in self.signature['output_files']:
            for label in labels:
                if 'Camera' in sig[0]:
                    full_output_files.append((sig[0].replace('*Camera', f'{label}Camera'), sig[1], sig[2]))
                elif 'ROI' in sig[0]:
                    full_output_files.append((sig[0].replace('*ROI', f'{label}ROI'), sig[1], sig[2]))

        self.output_files = full_output_files


class PostDLC(tasks.Task):
    """
    The post_dlc task takes dlc traces as input and computes useful quantities, as well as qc.
    """
    io_charge = 90
    level = 3
    force = True
    signature = {'input_files': [('_ibl_leftCamera.dlc.pqt', 'alf', True),
                                 ('_ibl_bodyCamera.dlc.pqt', 'alf', True),
                                 ('_ibl_rightCamera.dlc.pqt', 'alf', True),
                                 ('_ibl_rightCamera.times.npy', 'alf', True),
                                 ('_ibl_leftCamera.times.npy', 'alf', True),
                                 ('_ibl_bodyCamera.times.npy', 'alf', True),
                                 # the following are required for the DLC plot only
                                 # they are not strictly required, some plots just might be skipped
                                 # In particular the raw videos don't need to be downloaded as they can be streamed
                                 ('_iblrig_bodyCamera.raw.mp4', 'raw_video_data', True),
                                 ('_iblrig_leftCamera.raw.mp4', 'raw_video_data', True),
                                 ('_iblrig_rightCamera.raw.mp4', 'raw_video_data', True),
                                 ('rightROIMotionEnergy.position.npy', 'alf', True),
                                 ('leftROIMotionEnergy.position.npy', 'alf', True),
                                 ('bodyROIMotionEnergy.position.npy', 'alf', True),
                                 ('_ibl_trials.table.pqt', 'alf', True),
                                 ('_ibl_wheel.position.npy', 'alf', True),
                                 ('_ibl_wheel.timestamps.npy', 'alf', True),
                                 ],
                 # More files are required for all panels of the DLC QC plot to function
                 'output_files': [('_ibl_leftCamera.features.pqt', 'alf', True),
                                  ('_ibl_rightCamera.features.pqt', 'alf', True),
                                  ('licks.times.npy', 'alf', True),
                                  # ('dlc_qc_plot.png', 'snapshot', False)
                                  ]
                 }

    def _run(self, overwrite=True, run_qc=True, plot_qc=True):
        """
        Run the EphysPostDLC task. Returns a list of file locations for the output files in signature. The created plot
        (dlc_qc_plot.png) is not returned, but saved in session_path/snapshots and uploaded to Alyx as a note.

        :param overwrite: bool, whether to recompute existing output files (default is False).
                          Note that the dlc_qc_plot will be (re-)computed even if overwrite = False
        :param run_qc: bool, whether to run the DLC QC (default is True)
        :param plot_qc: book, whether to create the dlc_qc_plot (default is True)

        """
        # Check if output files exist locally
        exist, output_files = self.assert_expected(self.signature['output_files'], silent=True)
        if exist and not overwrite:
            _logger.warning('EphysPostDLC outputs exist and overwrite=False, skipping computations of outputs.')
        else:
            if exist and overwrite:
                _logger.warning('EphysPostDLC outputs exist and overwrite=True, overwriting existing outputs.')
            # Find all available dlc files
            dlc_files = list(Path(self.session_path).joinpath('alf').glob('_ibl_*Camera.dlc.*'))
            for dlc_file in dlc_files:
                _logger.debug(dlc_file)
            output_files = []
            combined_licks = []

            for dlc_file in dlc_files:
                # Catch unforeseen exceptions and move on to next cam
                try:
                    cam = label_from_path(dlc_file)
                    # load dlc trace and camera times
                    dlc = pd.read_parquet(dlc_file)
                    dlc_thresh = likelihood_threshold(dlc, 0.9)
                    # try to load respective camera times
                    try:
                        dlc_t = np.load(next(Path(self.session_path).joinpath('alf').glob(f'_ibl_{cam}Camera.times.*npy')))
                        times = True
                        if dlc_t.shape[0] == 0:
                            _logger.error(f'camera.times empty for {cam} camera. '
                                          f'Computations using camera.times will be skipped')
                            self.status = -1
                            times = False
                        elif dlc_t.shape[0] < len(dlc_thresh):
                            _logger.error(f'Camera times shorter than DLC traces for {cam} camera. '
                                          f'Computations using camera.times will be skipped')
                            self.status = -1
                            times = 'short'
                    except StopIteration:
                        self.status = -1
                        times = False
                        _logger.error(f'No camera.times for {cam} camera. '
                                      f'Computations using camera.times will be skipped')
                    # These features are only computed from left and right cam
                    if cam in ('left', 'right'):
                        features = pd.DataFrame()
                        # If camera times are available, get the lick time stamps for combined array
                        if times is True:
                            _logger.info(f"Computing lick times for {cam} camera.")
                            combined_licks.append(get_licks(dlc_thresh, dlc_t))
                        elif times is False:
                            _logger.warning(f"Skipping lick times for {cam} camera as no camera.times available")
                        elif times == 'short':
                            _logger.warning(f"Skipping lick times for {cam} camera as camera.times are too short")
                        # Compute pupil diameter, raw and smoothed
                        _logger.info(f"Computing raw pupil diameter for {cam} camera.")
                        features['pupilDiameter_raw'] = get_pupil_diameter(dlc_thresh)
                        try:
                            _logger.info(f"Computing smooth pupil diameter for {cam} camera.")
                            features['pupilDiameter_smooth'] = get_smooth_pupil_diameter(features['pupilDiameter_raw'],
                                                                                         cam)
                        except BaseException:
                            _logger.error(f"Computing smooth pupil diameter for {cam} camera failed, saving all NaNs.")
                            _logger.error(traceback.format_exc())
                            features['pupilDiameter_smooth'] = np.nan
                        # Safe to pqt
                        features_file = Path(self.session_path).joinpath('alf', f'_ibl_{cam}Camera.features.pqt')
                        features.to_parquet(features_file)
                        output_files.append(features_file)

                    # For all cams, compute DLC qc if times available
                    if run_qc is True and times in [True, 'short']:
                        # Setting download_data to False because at this point the data should be there
                        qc = DlcQC(self.session_path, side=cam, one=self.one, download_data=False)
                        qc.run(update=True)
                    else:
                        if times is False:
                            _logger.warning(f"Skipping QC for {cam} camera as no camera.times available")
                        if not run_qc:
                            _logger.warning(f"Skipping QC for {cam} camera as run_qc=False")

                except BaseException:
                    _logger.error(traceback.format_exc())
                    self.status = -1
                    continue

            # Combined lick times
            if len(combined_licks) > 0:
                lick_times_file = Path(self.session_path).joinpath('alf', 'licks.times.npy')
                np.save(lick_times_file, sorted(np.concatenate(combined_licks)))
                output_files.append(lick_times_file)
            else:
                _logger.warning("No lick times computed for this session.")

        if plot_qc:
            _logger.info("Creating DLC QC plot")
            try:
                session_id = self.one.path2eid(self.session_path)
                fig_path = self.session_path.joinpath('snapshot', 'dlc_qc_plot.png')
                if not fig_path.parent.exists():
                    fig_path.parent.mkdir(parents=True, exist_ok=True)
                fig = dlc_qc_plot(self.session_path, one=self.one)
                fig.savefig(fig_path)
                fig.clf()
                snp = ReportSnapshot(self.session_path, session_id, one=self.one)
                snp.outputs = [fig_path]
                snp.register_images(widths=['orig'],
                                    function=str(dlc_qc_plot.__module__) + '.' + str(dlc_qc_plot.__name__))
            except BaseException:
                _logger.error('Could not create and/or upload DLC QC Plot')
                _logger.error(traceback.format_exc())
                self.status = -1

        return output_files
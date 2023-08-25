"""Standard task protocol extractor dynamic pipeline tasks."""
import logging
import traceback

from pkg_resources import parse_version
import one.alf.io as alfio
from one.alf.files import session_path_parts
from one.api import ONE

from ibllib.oneibl.registration import get_lab
from ibllib.pipes import base_tasks
from ibllib.io.raw_data_loaders import load_settings
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.qc.task_metrics import HabituationQC, TaskQC
from ibllib.io.extractors.ephys_passive import PassiveChoiceWorld
from ibllib.io.extractors import bpod_trials
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io.extractors.bpod_trials import get_bpod_extractor
from ibllib.io.extractors.ephys_fpga import extract_all
from ibllib.io.extractors.mesoscope import TimelineTrials
from ibllib.pipes import training_status
from ibllib.plots.figures import BehaviourPlots

_logger = logging.getLogger('ibllib')


class HabituationRegisterRaw(base_tasks.RegisterRawDataTask, base_tasks.BehaviourTask):
    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [],
            'output_files': [
                ('_iblrig_taskData.raw.*', self.collection, True),
                ('_iblrig_taskSettings.raw.*', self.collection, True),
                ('_iblrig_encoderEvents.raw*', self.collection, False),
                ('_iblrig_encoderPositions.raw*', self.collection, False),
                ('_iblrig_encoderTrialInfo.raw*', self.collection, False),
                ('_iblrig_stimPositionScreen.raw*', self.collection, False),
                ('_iblrig_syncSquareUpdate.raw*', self.collection, False),
                ('_iblrig_ambientSensorData.raw*', self.collection, False)
            ]
        }
        return signature


class HabituationTrialsBpod(base_tasks.BehaviourTask):
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [
                ('_iblrig_taskData.raw.*', self.collection, True),
                ('_iblrig_taskSettings.raw.*', self.collection, True),
            ],
            'output_files': [
                ('*trials.contrastLeft.npy', self.output_collection, True),
                ('*trials.contrastRight.npy', self.output_collection, True),
                ('*trials.feedback_times.npy', self.output_collection, True),
                ('*trials.feedbackType.npy', self.output_collection, True),
                ('*trials.goCue_times.npy', self.output_collection, True),
                ('*trials.goCueTrigger_times.npy', self.output_collection, True),
                ('*trials.intervals.npy', self.output_collection, True),
                ('*trials.rewardVolume.npy', self.output_collection, True),
                ('*trials.stimOff_times.npy', self.output_collection, True),
                ('*trials.stimOn_times.npy', self.output_collection, True),
                ('*trials.stimOnTrigger_times.npy', self.output_collection, True),
            ]
        }
        return signature

    def _run(self, update=True):
        """
        Extracts an iblrig training session
        """
        extractor = bpod_trials.get_bpod_extractor(self.session_path, task_collection=self.collection)
        trials, output_files = extractor.extract(task_collection=self.collection, save=True)

        if trials is None:
            return None
        if self.one is None or self.one.offline:
            return output_files
        # Run the task QC
        # Compile task data for QC
        qc = HabituationQC(self.session_path, one=self.one)
        qc.extractor = TaskQCExtractor(self.session_path, sync_collection=self.sync_collection,
                                       one=self.one, sync_type=self.sync, task_collection=self.collection)
        namespace = 'task' if self.protocol_number is None else f'task_{self.protocol_number:02}'
        qc.run(update=update, namespace=namespace)
        return output_files


class TrialRegisterRaw(base_tasks.RegisterRawDataTask, base_tasks.BehaviourTask):
    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [],
            'output_files': [
                ('_iblrig_taskData.raw.*', self.collection, True),
                ('_iblrig_taskSettings.raw.*', self.collection, True),
                ('_iblrig_encoderEvents.raw*', self.collection, False),
                ('_iblrig_encoderPositions.raw*', self.collection, False),
                ('_iblrig_encoderTrialInfo.raw*', self.collection, False),
                ('_iblrig_stimPositionScreen.raw*', self.collection, False),
                ('_iblrig_syncSquareUpdate.raw*', self.collection, False),
                ('_iblrig_ambientSensorData.raw*', self.collection, False)
            ]
        }
        return signature


class PassiveRegisterRaw(base_tasks.RegisterRawDataTask, base_tasks.BehaviourTask):
    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [],
            'output_files': [('_iblrig_taskSettings.raw.*', self.collection, True),
                             ('_iblrig_encoderEvents.raw*', self.collection, True),
                             ('_iblrig_encoderPositions.raw*', self.collection, True),
                             ('_iblrig_encoderTrialInfo.raw*', self.collection, True),
                             ('_iblrig_stimPositionScreen.raw*', self.collection, True),
                             ('_iblrig_syncSquareUpdate.raw*', self.collection, True),
                             ('_iblrig_RFMapStim.raw*', self.collection, True)]
        }
        return signature


class PassiveTask(base_tasks.BehaviourTask):
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('_iblrig_taskSettings.raw*', self.collection, True),
                            ('_iblrig_RFMapStim.raw*', self.collection, True),
                            (f'_{self.sync_namespace}_sync.channels.*', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.polarities.*', self.sync_collection, True),
                            (f'_{self.sync_namespace}_sync.times.*', self.sync_collection, True),
                            ('*.wiring.json', self.sync_collection, False),
                            ('*.meta', self.sync_collection, False)],
            'output_files': [('_ibl_passiveGabor.table.csv', self.output_collection, True),
                             ('_ibl_passivePeriods.intervalsTable.csv', self.output_collection, True),
                             ('_ibl_passiveRFM.times.npy', self.output_collection, True),
                             ('_ibl_passiveStims.table.csv', self.output_collection, True)]
        }
        return signature

    def _run(self, **kwargs):
        """returns a list of pathlib.Paths. """
        data, paths = PassiveChoiceWorld(self.session_path).extract(
            sync_collection=self.sync_collection, task_collection=self.collection, save=True,
            path_out=self.session_path.joinpath(self.output_collection), protocol_number=self.protocol_number)

        if any(x is None for x in paths):
            self.status = -1

        return paths


class PassiveTaskTimeline(base_tasks.BehaviourTask, base_tasks.MesoscopeTask):
    """TODO should be mesoscope invariant, using wiring file"""
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('_iblrig_taskSettings.raw*', self.collection, True),
                            ('_iblrig_RFMapStim.raw*', self.collection, True),
                            (f'_{self.sync_namespace}_sync.channels.*', self.sync_collection, False),
                            (f'_{self.sync_namespace}_sync.polarities.*', self.sync_collection, False),
                            (f'_{self.sync_namespace}_sync.times.*', self.sync_collection, False)],
            'output_files': [('_ibl_passiveGabor.table.csv', self.output_collection, True),
                             ('_ibl_passivePeriods.intervalsTable.csv', self.output_collection, True),
                             ('_ibl_passiveRFM.times.npy', self.output_collection, True),
                             ('_ibl_passiveStims.table.csv', self.output_collection, True)]
        }
        return signature

    def _run(self, **kwargs):
        """returns a list of pathlib.Paths.
        This class exists to load the sync file and set the protocol_number to None
        """
        settings = load_settings(self.session_path, self.collection)
        version = settings.get('IBLRIG_VERSION_TAG', '100.0.0')
        if version == '100.0.0' or parse_version(version) <= parse_version('7.1.0'):
            _logger.warning('Protocol spacers not supported; setting protocol_number to None')
            self.protocol_number = None

        sync, chmap = self.load_sync()
        data, paths = PassiveChoiceWorld(self.session_path).extract(
            sync_collection=self.sync_collection, task_collection=self.collection, save=True,
            path_out=self.session_path.joinpath(self.output_collection),
            protocol_number=self.protocol_number, sync=sync, sync_map=chmap)

        if any(x is None for x in paths):
            self.status = -1

        return paths


class ChoiceWorldTrialsBpod(base_tasks.BehaviourTask):
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [
                ('_iblrig_taskData.raw.*', self.collection, True),
                ('_iblrig_taskSettings.raw.*', self.collection, True),
                ('_iblrig_encoderEvents.raw*', self.collection, True),
                ('_iblrig_encoderPositions.raw*', self.collection, True)],
            'output_files': [
                ('*trials.goCueTrigger_times.npy', self.output_collection, True),
                ('*trials.stimOnTrigger_times.npy', self.output_collection, False),
                ('*trials.table.pqt', self.output_collection, True),
                ('*wheel.position.npy', self.output_collection, True),
                ('*wheel.timestamps.npy', self.output_collection, True),
                ('*wheelMoves.intervals.npy', self.output_collection, True),
                ('*wheelMoves.peakAmplitude.npy', self.output_collection, True)
            ]
        }
        return signature

    def _run(self, update=True):
        """
        Extracts an iblrig training session
        """
        extractor = bpod_trials.get_bpod_extractor(self.session_path, task_collection=self.collection)
        extractor.default_path = self.output_collection
        trials, output_files = extractor.extract(task_collection=self.collection, save=True)
        if trials is None:
            return None
        if self.one is None or self.one.offline:
            return output_files
        # Run the task QC
        # Compile task data for QC
        type = get_session_extractor_type(self.session_path, task_collection=self.collection)
        # FIXME Task data should not need re-extracting
        if type == 'habituation':
            qc = HabituationQC(self.session_path, one=self.one)
            qc.extractor = TaskQCExtractor(self.session_path, one=self.one, sync_collection=self.sync_collection,
                                           sync_type=self.sync, task_collection=self.collection)
        else:  # Update wheel data
            qc = TaskQC(self.session_path, one=self.one)
            qc.extractor = TaskQCExtractor(self.session_path, one=self.one, sync_collection=self.sync_collection,
                                           sync_type=self.sync, task_collection=self.collection)
            qc.extractor.wheel_encoding = 'X1'
        # Aggregate and update Alyx QC fields
        namespace = 'task' if self.protocol_number is None else f'task_{self.protocol_number:02}'
        qc.run(update=update, namespace=namespace)

        return output_files


class ChoiceWorldTrialsNidq(base_tasks.BehaviourTask):
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [
                ('_iblrig_taskData.raw.*', self.collection, True),
                ('_iblrig_taskSettings.raw.*', self.collection, True),
                ('_iblrig_encoderEvents.raw*', self.collection, True),
                ('_iblrig_encoderPositions.raw*', self.collection, True),
                (f'_{self.sync_namespace}_sync.channels.npy', self.sync_collection, True),
                (f'_{self.sync_namespace}_sync.polarities.npy', self.sync_collection, True),
                (f'_{self.sync_namespace}_sync.times.npy', self.sync_collection, True),
                ('*wiring.json', self.sync_collection, False),
                ('*.meta', self.sync_collection, True)],
            'output_files': [
                ('*trials.goCueTrigger_times.npy', self.output_collection, True),
                ('*trials.intervals_bpod.npy', self.output_collection, False),
                ('*trials.stimOff_times.npy', self.output_collection, False),
                ('*trials.table.pqt', self.output_collection, True),
                ('*wheel.position.npy', self.output_collection, True),
                ('*wheel.timestamps.npy', self.output_collection, True),
                ('*wheelMoves.intervals.npy', self.output_collection, True),
                ('*wheelMoves.peakAmplitude.npy', self.output_collection, True)
            ]
        }
        return signature

    def _behaviour_criterion(self, update=True):
        """
        Computes and update the behaviour criterion on Alyx
        """
        from brainbox.behavior import training

        trials = alfio.load_object(self.session_path.joinpath(self.output_collection), 'trials')
        good_enough = training.criterion_delay(
            n_trials=trials["intervals"].shape[0],
            perf_easy=training.compute_performance_easy(trials),
        )
        if update:
            eid = self.one.path2eid(self.session_path, query_type='remote')
            self.one.alyx.json_field_update(
                "sessions", eid, "extended_qc", {"behavior": int(good_enough)}
            )

    def _extract_behaviour(self):
        dsets, out_files = extract_all(self.session_path, self.sync_collection, task_collection=self.collection,
                                       save_path=self.session_path.joinpath(self.output_collection),
                                       protocol_number=self.protocol_number, save=True)

        return dsets, out_files

    def _run_qc(self, trials_data, update=True, plot_qc=True):
        # Run the task QC
        qc = TaskQC(self.session_path, one=self.one, log=_logger)
        qc.extractor = TaskQCExtractor(self.session_path, lazy=True, one=qc.one, sync_collection=self.sync_collection,
                                       sync_type=self.sync, task_collection=self.collection)
        # Extract extra datasets required for QC
        qc.extractor.data = trials_data  # FIXME This line is pointless
        qc.extractor.extract_data()

        # Aggregate and update Alyx QC fields
        namespace = 'task' if self.protocol_number is None else f'task_{self.protocol_number:02}'
        qc.run(update=update, namespace=namespace)

        if plot_qc:
            _logger.info('Creating Trials QC plots')
            try:
                # TODO needs to be adapted for chained protocols
                session_id = self.one.path2eid(self.session_path)
                plot_task = BehaviourPlots(session_id, self.session_path, one=self.one)
                _ = plot_task.run()
                self.plot_tasks.append(plot_task)

            except Exception:
                _logger.error('Could not create Trials QC Plot')
                _logger.error(traceback.format_exc())
                self.status = -1

    def _run(self, update=True, plot_qc=True):
        dsets, out_files = self._extract_behaviour()

        if not self.one or self.one.offline:
            return out_files

        self._behaviour_criterion(update=update)
        self._run_qc(dsets, update=update, plot_qc=plot_qc)
        return out_files


class ChoiceWorldTrialsTimeline(ChoiceWorldTrialsNidq):
    """Behaviour task extractor with DAQdata.raw NPY datasets."""
    @property
    def signature(self):
        signature = super().signature
        signature['input_files'] = [
            ('_iblrig_taskData.raw.*', self.collection, True),
            ('_iblrig_taskSettings.raw.*', self.collection, True),
            ('_iblrig_encoderEvents.raw*', self.collection, True),
            ('_iblrig_encoderPositions.raw*', self.collection, True),
            (f'_{self.sync_namespace}_DAQdata.raw.npy', self.sync_collection, True),
            (f'_{self.sync_namespace}_DAQdata.timestamps.npy', self.sync_collection, True),
            (f'_{self.sync_namespace}_DAQdata.meta.json', self.sync_collection, True),
        ]
        if self.protocol:
            extractor = get_bpod_extractor(self.session_path, protocol=self.protocol)
            if extractor.save_names:
                signature['output_files'] = [(fn, self.output_collection, True)
                                             for fn in filter(None, extractor.save_names)]
        return signature

    def _extract_behaviour(self):
        """Extract the Bpod trials data and Timeline acquired signals."""
        # First determine the extractor from the task protocol
        extractor = get_bpod_extractor(self.session_path, self.protocol, self.collection)
        ret, _ = extractor.extract(save=False, task_collection=self.collection)
        bpod_trials = {k: v for k, v in zip(extractor.var_names, ret)}

        trials = TimelineTrials(self.session_path, bpod_trials=bpod_trials)
        save_path = self.session_path / self.output_collection
        if not self._spacer_support(extractor.settings):
            _logger.warning('Protocol spacers not supported; setting protocol_number to None')
            self.protocol_number = None
        dsets, out_files = trials.extract(
            save=True, path_out=save_path, sync_collection=self.sync_collection,
            task_collection=self.collection, protocol_number=self.protocol_number)

        if not isinstance(dsets, dict):
            dsets = {k: v for k, v in zip(trials.var_names, dsets)}

        self.timeline = trials.timeline  # Store for QC later
        self.frame2ttl = trials.frame2ttl
        self.audio = trials.audio

        return dsets, out_files

    def _run_qc(self, trials_data, update=True, **kwargs):
        """
        Run the task QC and update Alyx with results.

        Parameters
        ----------
        trials_data : dict
            The extracted trials data.
        update : bool
            If true, update Alyx with the result.

        Notes
        -----
        - Unlike the super class, currently the QC plots are not generated.
        - Expects the frame2ttl and audio attributes to be set from running _extract_behaviour.
        """
        # TODO Task QC extractor for Timeline
        qc = TaskQC(self.session_path, one=self.one, log=_logger)
        qc.extractor = TaskQCExtractor(self.session_path, lazy=True, one=qc.one, sync_collection=self.sync_collection,
                                       sync_type=self.sync, task_collection=self.collection)
        # Extract extra datasets required for QC
        qc.extractor.data = TaskQCExtractor.rename_data(trials_data.copy())
        qc.extractor.load_raw_data()

        qc.extractor.frame_ttls = self.frame2ttl
        qc.extractor.audio_ttls = self.audio
        # qc.extractor.bpod_ttls = channel_events('bpod')

        # Aggregate and update Alyx QC fields
        namespace = 'task' if self.protocol_number is None else f'task_{self.protocol_number:02}'
        qc.run(update=update, namespace=namespace)


class TrainingStatus(base_tasks.BehaviourTask):
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [
                ('_iblrig_taskData.raw.*', self.collection, True),
                ('_iblrig_taskSettings.raw.*', self.collection, True),
                ('*trials.table.pqt', self.output_collection, True)],
            'output_files': []
        }
        return signature

    def _run(self, upload=True):
        """
        Extracts training status for subject
        """

        lab = get_lab(self.session_path, self.one.alyx)
        if lab == 'cortexlab':
            one = ONE(base_url='https://alyx.internationalbrainlab.org')
        else:
            one = self.one

        df = training_status.get_latest_training_information(self.session_path, one)
        if df is not None:
            training_status.make_plots(
                self.session_path, self.one, df=df, save=True, upload=upload, task_collection=self.collection)
            # Update status map in JSON field of subjects endpoint
            if self.one and not self.one.offline:
                _logger.debug('Updating JSON field of subjects endpoint')
                status = (df.set_index('date')[['training_status', 'session_path']].drop_duplicates(
                    subset='training_status', keep='first').to_dict())
                date, sess = status.items()
                data = {'trained_criteria': {v.replace(' ', '_'): (k, self.one.path2eid(sess[1][k]))
                                             for k, v in date[1].items()}}
                _, subject, *_ = session_path_parts(self.session_path)
                self.one.alyx.json_field_update('subjects', subject, data=data)
        output_files = []
        return output_files

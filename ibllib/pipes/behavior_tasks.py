from ibllib.pipes import base_tasks
from ibllib.io.extractors.ephys_passive import PassiveChoiceWorld
from ibllib.io.extractors import bpod_trials
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.qc.task_metrics import HabituationQC, TaskQC
from ibllib.io.extractors.base import get_session_extractor_type
from ibllib.io.extractors.ephys_fpga import extract_all
from ibllib.pipes import training_status

import one.alf.io as alfio
from ibllib.plots.figures import BehaviourPlots
import logging
import traceback

_logger = logging.getLogger('ibllib')


class HabituationRegisterRaw(base_tasks.RegisterRawDataTask):
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


class HabituationTrialsBpod(base_tasks.DynamicTask):
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
                ('*trials.contrastLeft.npy', 'alf', True),
                ('*trials.contrastRight.npy', 'alf', True),
                ('*trials.feedback_times.npy', 'alf', True),
                ('*trials.feedbackType.npy', 'alf', True),
                ('*trials.goCue_times.npy', 'alf', True),
                ('*trials.goCueTrigger_times.npy', 'alf', True),
                ('*trials.intervals.npy', 'alf', True),
                ('*trials.rewardVolume.npy', 'alf', True),
                ('*trials.stimOff_times.npy', 'alf', True),
                ('*trials.stimOn_times.npy', 'alf', True),
                ('*trials.stimOnTrigger_times.npy', 'alf', True),
            ]
        }
        return signature

    def _run(self, update=True):
        """
        Extracts an iblrig training session
        """
        # TODO this doesn't use the self.collection in any way, always assumes data in raw_behavior_data, needs to be changed
        trials, wheel, output_files = bpod_trials.extract_all(self.session_path, save=True)
        if trials is None:
            return None
        if self.one is None or self.one.offline:
            return output_files
        # Run the task QC
        # Compile task data for QC
        qc = HabituationQC(self.session_path, one=self.one)
        qc.extractor = TaskQCExtractor(self.session_path, one=self.one)
        qc.run(update=update)
        return output_files


class TrialRegisterRaw(base_tasks.RegisterRawDataTask):
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


class PassiveRegisterRaw(base_tasks.RegisterRawDataTask):
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


class PassiveTask(base_tasks.DynamicTask):
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
            'output_files': [('_ibl_passiveGabor.table.csv', 'alf', True),
                             ('_ibl_passivePeriods.intervalsTable.csv', 'alf', True),
                             ('_ibl_passiveRFM.times.npy', 'alf', True),
                             ('_ibl_passiveStims.table.csv', 'alf', True)]
        }
        return signature

    def _run(self):
        """returns a list of pathlib.Paths. """
        data, paths = PassiveChoiceWorld(self.session_path).extract(
            sync_collection=self.sync_collection, task_collection=self.collection, save=True)

        if any([x is None for x in paths]):
            self.status = -1

        return paths


class ChoiceWorldTrialsBpod(base_tasks.DynamicTask):
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
                ('*trials.goCueTrigger_times.npy', 'alf', True),
                ('*trials.stimOnTrigger_times.npy', 'alf', False),
                ('*trials.table.pqt', 'alf', True),
                ('*wheel.position.npy', 'alf', True),
                ('*wheel.timestamps.npy', 'alf', True),
                ('*wheelMoves.intervals.npy', 'alf', True),
                ('*wheelMoves.peakAmplitude.npy', 'alf', True)
            ]
        }
        return signature

    def _run(self, update=True):
        """
        Extracts an iblrig training session
        """
        # TODO this doesn't use the self.collection in any way, always assumes data in raw_behavior_data, needs to be changed
        trials, wheel, output_files = bpod_trials.extract_all(self.session_path, save=True)
        if trials is None:
            return None
        if self.one is None or self.one.offline:
            return output_files
        # Run the task QC
        # Compile task data for QC
        type = get_session_extractor_type(self.session_path)
        if type == 'habituation':
            qc = HabituationQC(self.session_path, one=self.one)
            qc.extractor = TaskQCExtractor(self.session_path, one=self.one)
        else:  # Update wheel data
            qc = TaskQC(self.session_path, one=self.one)
            qc.extractor = TaskQCExtractor(self.session_path, one=self.one)
            qc.extractor.wheel_encoding = 'X1'
        # Aggregate and update Alyx QC fields
        qc.run(update=update)
        return output_files


class ChoiceWorldTrialsNidq(base_tasks.DynamicTask):
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
                ('*trials.goCueTrigger_times.npy', 'alf', True),
                ('*trials.intervals_bpod.npy', 'alf', False),
                ('*trials.stimOff_times.npy', 'alf', False),
                ('*trials.table.pqt', 'alf', True),
                ('*wheel.position.npy', 'alf', True),
                ('*wheel.timestamps.npy', 'alf', True),
                ('*wheelMoves.intervals.npy', 'alf', True),
                ('*wheelMoves.peakAmplitude.npy', 'alf', True)
            ]
        }
        return signature

    def _behaviour_criterion(self, update=True):
        """
        Computes and update the behaviour criterion on Alyx
        """
        from brainbox.behavior import training

        trials = alfio.load_object(self.session_path.joinpath("alf"), "trials")
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
        dsets, out_files = extract_all(self.session_path, self.sync_collection, save=True)

        return dsets, out_files

    def _run(self, update=True, plot_qc=True):
        dsets, out_files = self._extract_behaviour()

        if not self.one or self.one.offline:
            return out_files

        self._behaviour_criterion(update=update)
        # Run the task QC
        # TODO this doesn't use the self.collection in any way, always assumes data in raw_behavior_data, needs to be changed
        qc = TaskQC(self.session_path, one=self.one, log=_logger)
        qc.extractor = TaskQCExtractor(self.session_path, lazy=True, one=qc.one)
        # Extract extra datasets required for QC
        qc.extractor.data = dsets
        qc.extractor.extract_data()
        # Aggregate and update Alyx QC fields
        qc.run(update=update)

        if plot_qc:
            _logger.info("Creating Trials QC plots")
            try:
                session_id = self.one.path2eid(self.session_path)
                plot_task = BehaviourPlots(session_id, self.session_path, one=self.one)
                _ = plot_task.run()
                self.plot_tasks.append(plot_task)

            except Exception:
                _logger.error('Could not create Trials QC Plot')
                _logger.error(traceback.format_exc())
                self.status = -1

        return out_files


class TrainingStatus(base_tasks.DynamicTask):
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [
                ('_iblrig_taskData.raw.*', self.collection, True),
                ('_iblrig_taskSettings.raw.*', self.collection, True),
                ('*trials.table.pqt', 'alf', True)],
            'output_files': []
        }
        return signature

    def _run(self, upload=True):
        """
        Extracts training status for subject
        """
        df = training_status.get_latest_training_information(self.session_path, self.one)
        if df is not None:
            training_status.make_plots(self.session_path, self.one, df=df, save=True, upload=upload)
        output_files = []
        return output_files

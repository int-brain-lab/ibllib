from pathlib import Path
from ibllib.pipes import tasks


class TaskRegisterRaw(tasks.Task):
    cpu = 1
    io_charge = 90
    level = 0
    force = False
    signature = {
        'input_files': [],
        'output_files': [('_iblrig_taskData.raw.*', 'raw_XX_data', True),
                         ('_iblrig_taskSettings.raw.*', 'raw_XX_data', True),
                         ('_iblrig_encoderEvents.raw*', 'raw_XX_data', True),
                         ('_iblrig_encoderPositions.raw*', 'raw_XX_data', True)]}

    def _run(self):

        collection = self.runtime_args.get('protocol_collection', 'raw_behavior_data')
        out_files = []
        for file_sig in self.output_files:
            file_name, _, required = file_sig
            file_path = self.session_path.rglob(str(Path(collection).joinpath(file_name)))
            file_path = next(file_path, None)
            if not file_path and not required:
                continue
            out_files.append(file_path)

        return out_files

    def get_signatures(self, collection=None):

        collection = self.runtime_args.get('protocol_collection', 'raw_behavior_data')

        input_files = []
        for sig in self.signature['input_files']:
            input_files.append((sig[0], collection, sig[2]))
        self.input_files = input_files

        output_files = []
        for sig in self.signature['output_files']:
            output_files.append((sig[0], collection, sig[2]))
        self.output_files = output_files


class PassiveRegisterRaw(tasks.Task):
    cpu = 1
    io_charge = 90
    level = 0
    force = False
    signature = {
        'input_files': [],
        'output_files': [('_iblrig_taskData.raw.*', 'raw_XX_data', True),
                         ('_iblrig_taskSettings.raw.*', 'raw_XX_data', True),
                         ('_iblrig_encoderEvents.raw*', 'raw_XX_data', True),
                         ('_iblrig_encoderPositions.raw*', 'raw_XX_data', True),
                         ('_iblrig_RFMapStim.raw*', 'raw_XX_data', True)]}

    def _run(self):

        collection = self.runtime_args.get('protocol_collection', 'raw_passive_data')

        out_files = []
        for file_sig in self.output_files:
            file_name, _, required = file_sig
            file_path = self.session_path.rglob(str(Path(collection).joinpath(file_name)))
            file_path = next(file_path, None)
            if not file_path and not required:
                continue
            out_files.append(file_path)

        return out_files

    def get_signatures(self, collection=None):

        collection = self.runtime_args.get('protocol_collection', 'raw_passive_data')

        input_files = []
        for sig in self.signature['input_files']:
            input_files.append((sig[0], collection, sig[2]))
        self.input_files = input_files

        output_files = []
        for sig in self.signature['output_files']:
            output_files.append((sig[0], collection, sig[2]))
        self.output_files = output_files



# TODO make generic task
class TrainingTrialsBpod(tasks.Task):
    priority = 90
    level = 0
    force = False
    signature = {
        'input_files': [('_iblrig_taskData.raw.*', 'raw_XX_data', True),
                        ('_iblrig_taskSettings.raw.*', 'raw_XX_data', True),
                        ('_iblrig_encoderEvents.raw*', 'raw_XX_data', True),
                        ('_iblrig_encoderPositions.raw*', 'raw_XX_data', True)],
        'output_files': [('*trials.goCueTrigger_times.npy', 'alf', True),
                         ('*trials.itiDuration.npy', 'alf', False),
                         ('*trials.table.pqt', 'alf', True),
                         ('*wheel.position.npy', 'alf', True),
                         ('*wheel.timestamps.npy', 'alf', True),
                         ('*wheelMoves.intervals.npy', 'alf', True),
                         ('*wheelMoves.peakAmplitude.npy', 'alf', True)]
    }

    def _run(self):
        """
        Extracts an iblrig training session
        """
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
        qc.run(update=True)
        return output_files

    def get_signatures(self, **kwargs):
        collection = self.runtime_args.get('protocol_collection', 'raw_behavior_data')

        input_files = []
        for sig in self.signature['input_files']:
            input_files.append((sig[0], collection, sig[2]))
        self.input_files = input_files

        self.output_files = self.signature['output_files']


class TrainingTrialsFPGA(tasks.Task):
    priority = 90
    level = 1
    force = False
    signature = {
        'input_files': [('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
                        ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True),
                        ('_spikeglx_sync.channels.*', 'raw_ephys_data*', True),
                        ('_spikeglx_sync.polarities.*', 'raw_ephys_data*', True),
                        ('_spikeglx_sync.times.*', 'raw_ephys_data*', True),
                        ('_iblrig_encoderEvents.raw*', 'raw_behavior_data', True),
                        ('_iblrig_encoderPositions.raw*', 'raw_behavior_data', True),
                        ('*wiring.json', 'raw_ephys_data*', False),
                        ('*.meta', 'raw_ephys_data*', True)],
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



    def _behaviour_criterion(self):
        """
        Computes and update the behaviour criterion on Alyx
        """
        from brainbox.behavior import training

        trials = alfio.load_object(self.session_path.joinpath("alf"), "trials")
        good_enough = training.criterion_delay(
            n_trials=trials["intervals"].shape[0],
            perf_easy=training.compute_performance_easy(trials),
        )
        eid = self.one.path2eid(self.session_path, query_type='remote')
        self.one.alyx.json_field_update(
            "sessions", eid, "extended_qc", {"behavior": int(good_enough)}
        )

    def _extract_behaviour(self):
        sync_collection = self.runtime_args.get('sync_collection', 'raw_ephys_data')
        protocol_collection = self.runtime_args.get('protocol_collection', 'raw_behavior_data')
        dsets, out_files = ephys_fpga.extract_all(self.session_path, sync_collection, save=True)

        return dsets, out_files


    def _run(self, plot_qc=True):
        dsets, out_files = self._extract_behaviour()

        if not self.one or self.one.offline:
            return out_files

        self._behaviour_criterion()
        # Run the task QC
        qc = TaskQC(self.session_path, one=self.one, log=_logger)
        qc.extractor = TaskQCExtractor(self.session_path, lazy=True, one=qc.one)
        # Extract extra datasets required for QC
        qc.extractor.data = dsets
        qc.extractor.extract_data()
        # Aggregate and update Alyx QC fields
        qc.run(update=True)

        if plot_qc:
            _logger.info("Creating Trials QC plots")
            try:
                session_id = self.one.path2eid(self.session_path)
                plot_task = BehaviourPlots(session_id, self.session_path, one=self.one)
                _ = plot_task.run()
                self.plot_tasks.append(plot_task)

            except BaseException:
                _logger.error('Could not create Trials QC Plot')
                _logger.error(traceback.format_exc())
                self.status = -1

        return out_files

    def get_signatures(self, **kwargs):
        neuropixel_version = spikeglx.get_neuropixel_version_from_folder(self.session_path)
        probes = spikeglx.get_probes_from_folder(self.session_path)

        full_input_files = []
        for sig in self.signature['input_files']:
            if 'raw_ephys_data*' in sig[1]:
                if neuropixel_version != '3A':
                    full_input_files.append((sig[0], 'raw_ephys_data', sig[2]))
                for probe in probes:
                    full_input_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))
            else:
                full_input_files.append(sig)

        self.input_files = full_input_files

        self.output_files = self.signature['output_files']



class PassiveTask(tasks.Task):
    cpu = 1
    io_charge = 90
    level = 1
    force = False
    signature = {
        'input_files': [('_iblrig_taskSettings.raw*', 'raw_behavior_data', True),
                        ('_spikeglx_sync.channels.*', 'raw_ephys_data*', True),
                        ('_spikeglx_sync.polarities.*', 'raw_ephys_data*', True),
                        ('_spikeglx_sync.times.*', 'raw_ephys_data*', True),
                        ('*.meta', 'raw_ephys_data*', True),
                        ('*wiring.json', 'raw_ephys_data*', False),
                        ('_iblrig_RFMapStim.raw*', 'raw_passive_data', True)],
        'output_files': [('_ibl_passiveGabor.table.csv', 'alf', True),
                         ('_ibl_passivePeriods.intervalsTable.csv', 'alf', True),
                         ('_ibl_passiveRFM.times.npy', 'alf', True),
                         ('_ibl_passiveStims.table.csv', 'alf', True)]}

    def _run(self):
        """returns a list of pathlib.Paths. """
        data, paths = ephys_passive.PassiveChoiceWorld(self.session_path).extract(save=True)
        if any([x is None for x in paths]):
            self.status = -1
        # Register?
        return paths

    def get_signatures(self, **kwargs):
        neuropixel_version = spikeglx.get_neuropixel_version_from_folder(self.session_path)
        probes = spikeglx.get_probes_from_folder(self.session_path)

        full_input_files = []
        for sig in self.signature['input_files']:
            if 'raw_ephys_data*' in sig[1]:
                if neuropixel_version != '3A':
                    full_input_files.append((sig[0], 'raw_ephys_data', sig[2]))
                for probe in probes:
                    full_input_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))
            else:
                full_input_files.append(sig)

        self.input_files = full_input_files

        self.output_files = self.signature['output_files']




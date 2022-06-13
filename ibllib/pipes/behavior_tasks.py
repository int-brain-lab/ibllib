# whatever the folder needs to be input as an argument to the task
from pathlib import Path
from ibllib.pipes import tasks


class TaskRegisterRaw(tasks.Task):
    cpu = 1
    io_charge = 90
    level = 0
    force = False
    signature = {
        'input_files': [],
        'output_files': [('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
                         ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True),
                         ('_iblrig_encoderEvents.raw*', 'raw_behavior_data', True),
                         ('_iblrig_encoderPositions.raw*', 'raw_behavior_data', True)]}

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

    def get_signatures(self, collection=None):

        self.input_files = self.signature['input_files']
        if collection is None:
            self.output_files = self.signature['output_files']
        else:
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
        'output_files': [('_iblrig_taskData.raw.*', 'raw_passive_data', True),
                         ('_iblrig_taskSettings.raw.*', 'raw_passive_data', True),
                         ('_iblrig_encoderEvents.raw*', 'raw_passive_data', True),
                         ('_iblrig_encoderPositions.raw*', 'raw_passive_data', True),
                         ('_iblrig_RFMapStim.raw*', 'raw_passive_data', True)]}

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

    def get_signatures(self, collection=None):

        self.input_files = self.signature['input_files']

        if collection is None:
            self.output_files = self.signature['output_files']
        else:
            output_files = []
            for sig in self.signature['output_files']:
                output_files.append((sig[0], collection, sig[2]))
            self.output_files = output_files



class TrainingTrials(tasks.Task):

# needs to be based on the protocol which task to use
# map from protocol to task extractor

# submodules that everything needs to be extacted from
# different task based on different extractor?







    pass






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


def get_collection_from

def get_registerraw_task(protocol):
    if 'passive' in protocol:
        return PassiveRegisterRaw
    else
        return TaskRegisterRaw

def get_trials_task(protocol, sync):
    # try the default ones
    # otherwise try the project ones

def get_behavior_subpipe(session_path, protocol, collection, sync, sync_task):

    tasks = OrderedDict()
    tasks['BehaviorRegisterRaw'] = get_registerraw_task(protocol)(session_path, collection)
    tasks['BehaviorTrials'] = get_trials_task(protocol, sync)(session_path, collection, parents=[sync_task])

    return tasks

# pipes need a way to use the arg when rerunning the task


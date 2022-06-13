# These all need to take in a probe argument apart from the Ephys register raw


# EphysPulses

# EphysMtsComp # across probes

# RawEphysQC # per probe

class RawEphysQC(tasks.Task):
    """
    Computes raw electrophysiology QC
    """
    cpu = 2
    io_charge = 30  # this jobs reads raw ap files
    priority = 10  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything
    force = False
    signature = {
        'input_files': [('*ap.meta', 'raw_ephys_data/probe*', True),
                        ('*lf.meta', 'raw_ephys_data/probe*', True),  # not necessary to run task as optional computation
                        ('*lf.ch', 'raw_ephys_data/probe*', False),  # not required it .bin file
                        ('*lf.*bin', 'raw_ephys_data/probe*', True)],  # not necessary to run task as optional computation
        'output_files': [('_iblqc_ephysChannels.apRMS.npy', 'raw_ephys_data/probe*', True),
                         ('_iblqc_ephysChannels.rawSpikeRates.npy', 'raw_ephys_data/probe*', True),
                         ('_iblqc_ephysChannels.labels.npy', 'raw_ephys_data/probe*', True),
                         ('_iblqc_ephysSpectralDensityLF.freqs.npy', 'raw_ephys_data/probe*', True),
                         ('_iblqc_ephysSpectralDensityLF.power.npy', 'raw_ephys_data/probe*', True),
                         ('_iblqc_ephysSpectralDensityAP.freqs.npy', 'raw_ephys_data/probe*', True),
                         ('_iblqc_ephysSpectralDensityAP.power.npy', 'raw_ephys_data/probe*', True),
                         ('_iblqc_ephysTimeRmsLF.rms.npy', 'raw_ephys_data/probe*', True),
                         ('_iblqc_ephysTimeRmsLF.timestamps.npy', 'raw_ephys_data/probe*', True)]
    }

    def _run(self, overwrite=False):
        eid = self.one.path2eid(self.session_path)
        probes = [(x['id'], x['name']) for x in self.one.alyx.rest('insertions', 'list', session=eid)]
        # Usually there should be two probes, if there are less, check if all probes are registered
        if len(probes) < 2:
            _logger.warning(f"{len(probes)} probes registered for session {eid}, trying to register from local data")
            probes = [(p['id'], p['name']) for p in create_alyx_probe_insertions(self.session_path, one=self.one)]
        qc_files = []
        for pid, pname in probes:
            _logger.info(f"\nRunning QC for probe insertion {pname}")
            try:
                eqc = ephysqc.EphysQC(pid, session_path=self.session_path, one=self.one)
                qc_files.extend(eqc.run(update=True, overwrite=overwrite))
                _logger.info("Creating LFP QC plots")
                plot_task = LfpPlots(pid, session_path=self.session_path, one=self.one)
                _ = plot_task.run()
                self.plot_tasks.append(plot_task)
                plot_task = BadChannelsAp(pid, session_path=self.session_path, one=self.one)
                _ = plot_task.run()
                self.plot_tasks.append(plot_task)

            except AssertionError:
                _logger.error(traceback.format_exc())
                self.status = -1
                continue
        return qc_files

    def get_signatures(self, **kwargs):
        probes = spikeglx.get_probes_from_folder(self.session_path)

        full_output_files = []
        for sig in self.signature['output_files']:
            if 'raw_ephys_data/probe*' in sig[1]:
                for probe in probes:
                    full_output_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))

        self.output_files = full_output_files

        # input lf signature required or not required status is going to depend on the output we have, need to be agile here to
        # avoid unnecessary downloading of lf.cbin files
        expected_count = 0
        count = 0
        # check to see if we have lfp qc datasets
        for expected_file in full_output_files:
            if 'LF' in expected_file[0]:
                expected_count += 1
                actual_files = list(Path(self.session_path).rglob(str(Path(expected_file[1]).joinpath(expected_file[0]))))
                if len(actual_files) == 1:
                    count += 1

        lf_required = False if count == expected_count else True

        full_input_files = []
        for sig in self.signature['input_files']:
            if 'raw_ephys_data/probe*' in sig[1]:
                for probe in probes:
                    if 'lf' in sig[0]:
                        full_input_files.append((sig[0], f'raw_ephys_data/{probe}', lf_required if sig[2] else sig[2]))
                    else:
                        full_input_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))

        self.input_files = full_input_files





# SpikeSorting # per probe # chronic and split

# ChronicSpikeSorting

# EphysPulses


# EphysSync

# EphysCellQC
def detect_probes(session_path, register=False):
    # detect from the raw_ephys_data and register the probes to alyx
    # if np2 we need 4 probes
    # otherwise we need just one
    # try except so this doesn't stop the pipeline from being generated only if register flag is True


def get_ephys_pipeline(session_path, sync_task):

    tasks = OrderedDict()
    tasks['EphysMtscomp'] = EphysMtscomp(session_path)
    probes = detect_probes(session_path, register=True)

    for probe in probes:
        tasks[f'RawEphysQC_{probe}'] = RawEphysQC(session_path, probe)
        tasks[f'SpikeSorting_{probe}'] = RawEphysQC(session_path, probe)
        tasks[f'EphysCellQc_{probe}'] = RawEphysQC(session_path, probe)




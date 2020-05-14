import logging

from ibllib.io.extractors import ephys_fpga, ephys_trials
import ibllib.io.raw_data_loaders as rawio
from ibllib.pipes import jobs

_logger = logging.getLogger('ibllib')


class EphysPulses(jobs.Job):
    cpu = 2
    io_charge = 20  # this jobs reads raw ap files
    priority = 90  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything

    def _run(self, overwrite=False):
        syncs, out_files = ephys_fpga.extract_sync(self.session_path, overwrite=overwrite)
        return out_files


class EphysTrials(jobs.Job):
    priority = 90
    level = 1

    def _run(self):
        data = rawio.load_data(self.session_path)
        _logger.info('extract BPOD for ephys session')
        ephys_trials.extract_all(self.session_path, data=data, save=True, return_files=True)
        _logger.info('extract FPGA information for ephys session')
        tmax = data[-1]['behavior_data']['States timestamps']['exit_state'][0][-1] + 60
        ephys_fpga.extract_all(self.session_path, save=True, tmax=tmax)


class EphysExtractionPipeline(jobs.Pipeline):
    label = __name__

    def __init__(self, session_path, **kwargs):
        super(EphysExtractionPipeline, self).__init__(session_path, **kwargs)
        jobs = {}
        self.session_path = session_path
        jobs['EphysPulses'] = EphysPulses(self.session_path)
        jobs['EphysTrials'] = EphysTrials(self.session_path, parents=[jobs['EphysPulses']])

        self.jobs = jobs

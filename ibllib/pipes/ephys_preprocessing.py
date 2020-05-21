import logging

from ibllib.io import ffmpeg
from ibllib.io.extractors import ephys_fpga
from ibllib.pipes import jobs
from ibllib.ephys import ephysqc

_logger = logging.getLogger('ibllib')


#  level 0
class EphysPulses(jobs.Job):
    cpu = 2
    io_charge = 30  # this jobs reads raw ap files
    priority = 90  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything

    def _run(self, overwrite=False):
        syncs, out_files = ephys_fpga.extract_sync(self.session_path, overwrite=overwrite)
        return out_files


class RawEphysQC(jobs.Job):
    """
    Computes raw electrophysiology QC
    """
    cpu = 2
    io_charge = 30  # this jobs reads raw ap files
    priority = 10  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything

    def _run(self, overwrite=False):
        qc_files = ephysqc.raw_qc_session(self.session_path, overwrite=overwrite)
        return qc_files


class EphysAudio(jobs.Job):
    """
    Computes raw electrophysiology QC
    """
    cpu = 2
    priority = 10  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything

    def _run(self, overwrite=False):
        command = 'ffmpeg -i {file_in} -c:a flac -nostats {file_out}'
        file_in = next(self.session_path.rglob('_iblrig_micData.raw.wav'), None)
        if file_in is None:
            return
        file_out = file_in.with_suffix('.flac')
        status, output_files = ffmpeg.compress(file_in=file_in, file_out=file_out, command=command)
        return output_files


class SpikeSorting(jobs.Job):
    """
    Computes raw electrophysiology QC
    """
    gpu = 1
    io_charge = 70  # this jobs reads raw ap files
    priority = 60
    level = 0  # this job doesn't depend on anything

    def _run(self, overwrite=False):
        pass


#  level 1
class EphysTrials(jobs.Job):
    priority = 90
    level = 1

    def _run(self):
        dsets, out_files = ephys_fpga.extract_all(self.session_path, save=True)
        return out_files


class EphysExtractionPipeline(jobs.Pipeline):
    label = __name__

    def __init__(self, session_path, **kwargs):
        super(EphysExtractionPipeline, self).__init__(session_path, **kwargs)
        jobs = {}
        self.session_path = session_path
        # level 0
        jobs['EphysPulses'] = EphysPulses(self.session_path)
        jobs['EphysRawQC'] = RawEphysQC(self.session_path)
        jobs['EphysAudio'] = EphysAudio(self.session_path)
        jobs['SpikeSorting'] = SpikeSorting(self.session_path)
        jobs['EphysTrials'] = EphysTrials(self.session_path, parents=[jobs['EphysPulses']])

        self.jobs = jobs

import re
import logging
from collections import OrderedDict

from ibllib.io import ffmpeg, spikeglx
from ibllib.io.extractors import ephys_fpga
from ibllib.pipes import tasks
from ibllib.ephys import ephysqc, sync_probes, spikes
from ibllib.pipes.training_preprocessing import TrainingRegisterRaw as EphysRegisterRaw

_logger = logging.getLogger('ibllib')


#  level 0
class EphysPulses(tasks.Task):
    cpu = 2
    io_charge = 30  # this jobs reads raw ap files
    priority = 90  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything

    def _run(self, overwrite=False):
        syncs, out_files = ephys_fpga.extract_sync(self.session_path, overwrite=overwrite)
        for out_file in out_files:
            _logger.info(f"extracted pulses for {out_file}")
        return out_files


class RawEphysQC(tasks.Task):
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


class EphysAudio(tasks.Task):
    """
    Computes raw electrophysiology QC
    """
    cpu = 2
    priority = 10  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything

    def _run(self, overwrite=False):
        command = 'ffmpeg -i {file_in} -y -c:a flac -nostats {file_out}'
        file_in = next(self.session_path.rglob('_iblrig_micData.raw.wav'), None)
        if file_in is None:
            return
        file_out = file_in.with_suffix('.flac')
        status, output_file = ffmpeg.compress(file_in=file_in, file_out=file_out, command=command)
        return [output_file]


class SpikeSorting_KS2_Matlab(tasks.Task):
    """
    Computes raw electrophysiology QC
    """
    gpu = 1
    io_charge = 70  # this jobs reads raw ap files
    priority = 60
    level = 0  # this job doesn't depend on anything

    def _run(self, overwrite=False):
        efiles = spikeglx.glob_ephys_files(self.session_path)
        apfiles = [ef.get('ap') for ef in efiles if 'ap' in ef.keys()]
        for apfile in apfiles:
            ks2log = apfile.parent.joinpath('spike_sorting_ks2.log')
            if not ks2log.exists():
                # this will label the job with "empty" status in the database
                return None
            with open(ks2log) as fid:
                line = fid.readline()
            self.version = re.compile("[a-f0-9]{36}").findall(line)[0]
            return []  # the job will be labeled as complete with empty string


class EphysVideoCompress(tasks.Task):
    priority = 40
    level = 1

    def _run(self, **kwargs):
        # avi to mp4 compression
        command = ('ffmpeg -i {file_in} -y -codec:v libx264 -preset slow -crf 17 '
                   '-nostats -loglevel 0 -codec:a copy {file_out}')
        output_files = ffmpeg.iblrig_video_compression(self.session_path, command)
        if len(output_files) == 0:
            self.session_path.joinpath('')
        return output_files


#  level 1
class EphysTrials(tasks.Task):
    priority = 90
    level = 1

    def _run(self):
        dsets, out_files = ephys_fpga.extract_all(self.session_path, save=True)
        return out_files


class EphysSyncSpikeSorting(tasks.Task):
    priority = 90
    level = 1

    def _run(self):
        """
        Post spike-sorting processing:
        - synchronization of probes
        - ks2 to ALF conversion for each probes in alf/probeXX folder
        - computes spike sorting QC
        - creates probes object in alf folder
        To start the job for a session, all electrophysiology ap files from session need to be
        associated with a `sync_merge_ephys.flag` file
        Outputs individual probes
        """
        # first sync the probes
        status, sync_files = sync_probes.sync(self.session_path)
        # then convert ks2 to ALF and resync spike sorting data
        alf_files = spikes.sync_spike_sortings(self.session_path)
        # outputs the probes object in the ALF folder
        probe_files = spikes.probes_description(self.session_path)
        return sync_files + alf_files + probe_files


class EphysMtscomp(tasks.Task):
    priority = 50  # ideally after spike sorting
    level = 0

    def _run(self):
        """
        Compress ephys files looking for `compress_ephys.flag` whithin the probes folder
        Original bin file will be removed
        The registration flag created contains targeted file names at the root of the session
        """
        ephys_files = spikeglx.glob_ephys_files(self.session_path)
        out_files = []
        for ef in ephys_files:
            for typ in ['ap', 'lf', 'nidq']:
                bin_file = ef.get(typ)
                if not bin_file:
                    continue
                sr = spikeglx.Reader(bin_file)
                if sr.is_mtscomp:
                    out_files.append(bin_file)
                else:
                    _logger.info(f"Compressing binary file {bin_file}")
                    out_files.append(sr.compress_file(keep_original=False))
        return out_files


class EphysDLC(tasks.Task):
    gpu = 1
    cpu = 4
    io_charge = 90
    level = 1

    def _run(self):
        """empty placeholder for job creation only"""
        pass


class EphysExtractionPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(EphysExtractionPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks['EphysRegisterRaw'] = EphysRegisterRaw(self.session_path)
        tasks['EphysPulses'] = EphysPulses(self.session_path)
        tasks['EphysRawQC'] = RawEphysQC(self.session_path)
        tasks['EphysAudio'] = EphysAudio(self.session_path)
        tasks['SpikeSorting'] = SpikeSorting_KS2_Matlab(self.session_path)
        tasks['EphysVideoCompress'] = EphysVideoCompress(self.session_path)
        tasks['EphysMtscomp'] = EphysMtscomp(self.session_path)
        # level 1
        tasks['EphysSyncSpikeSorting'] = EphysSyncSpikeSorting(self.session_path, parents=[
            tasks['SpikeSorting'], tasks['EphysPulses']])
        tasks['EphysTrials'] = EphysTrials(self.session_path, parents=[tasks['EphysPulses']])
        tasks['EphysDLC'] = EphysDLC(self.session_path, parents=[tasks['EphysVideoCompress']])
        self.tasks = tasks

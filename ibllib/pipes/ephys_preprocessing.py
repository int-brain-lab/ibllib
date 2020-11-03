import re
from pathlib import Path
import logging
from collections import OrderedDict
import subprocess
import shutil

import mtscomp

from ibllib.io import ffmpeg, spikeglx
from ibllib.io.extractors import ephys_fpga
from ibllib.pipes import tasks
from ibllib.ephys import ephysqc, sync_probes, spikes
from ibllib.pipes.training_preprocessing import TrainingRegisterRaw as EphysRegisterRaw
from ibllib.qc.task_metrics import TaskQC
from ibllib.qc.task_extractors import TaskQCExtractor


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
        command = 'ffmpeg -i {file_in} -y -nostdin -c:a flac -nostats {file_out}'
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
    level = 1  # this job doesn't depend on anything

    @staticmethod
    def _fetch_ks2_commit_hash():
        command2run = 'git --git-dir ~/Documents/MATLAB/Kilosort2/.git rev-parse --verify HEAD'
        process = subprocess.Popen(command2run, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        info, error = process.communicate()
        if process.returncode != 0:
            _logger.error(f"Can't fetch matlab ks2 commit hash, will still attempt to run \n"
                          f"Error: {error.decode('utf-8')}")
            return ''
        return info.decode('utf-8').strip()

    def _run(self, overwrite=False):

        efiles = spikeglx.glob_ephys_files(self.session_path)

        apfiles = [(ef.get('ap'), ef.get('label')) for ef in efiles if 'ap' in ef.keys()]
        for ap_file, label in apfiles:
            # check for pre-existing spike-sorting
            # the spike sorting output can either be with the probe (<1.5.5) or in the
            # session_path/spike_sorters/ks2_matlab/probeXX folder
            ks2_dir = self.session_path.joinpath('spike_sorters', 'ks2_matlab', label)
            if ap_file.parent.joinpath('spike_sorting_ks2.log').exists():
                _logger.info(f'Already ran: spike_sorting_ks2.log found for {ap_file}, skipping.')
                continue  # this will label the job with ok status in the database
            if ks2_dir.joinpath('spike_sorting_ks2.log').exists():
                _logger.info(f'Already ran: spike_sorting_ks2.log found in {ks2_dir}, skipping.')
                continue
            # get the scratch drive from the shell script
            SHELL_SCRIPT = Path.home().joinpath(
                "Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/task_ks2_matlab.sh")
            with open(SHELL_SCRIPT) as fid:
                lines = fid.readlines()
            line = [line for line in lines if line.startswith('SCRATCH_DRIVE=')][0]
            m = re.search(r"\=(.*?)(\#|\n)", line)[0]
            scratch_drive = Path(m[1:-1].strip())
            assert(scratch_drive.exists())

            # clean up and create directory, this also checks write permissions
            # scratch dir has the following shape: ks2m/ZM_3003_2020-07-29_001_probe00
            # first makes sure the tmp dir is clean
            shutil.rmtree(scratch_drive.joinpath('ks2m'), ignore_errors=True)
            scratch_dir = scratch_drive.joinpath(
                'ks2m', '_'.join(list(self.session_path.parts[-3:]) + [label]))
            if scratch_dir.exists():
                shutil.rmtree(scratch_dir, ignore_errors=True)
            scratch_dir.mkdir(parents=True, exist_ok=True)

            # decompresses using mtscomp
            tmp_ap_file = scratch_dir.joinpath(ap_file.name).with_suffix('.bin')
            mtscomp.decompress(cdata=ap_file, out=tmp_ap_file)

            # run matlab spike sorting: with R2019a, it would be much easier to run with
            # -batch option as matlab errors are redirected to stderr automatically
            command2run = f"{SHELL_SCRIPT} {scratch_dir}"
            _logger.info(command2run)
            process = subprocess.Popen(command2run, shell=True, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, executable="/bin/bash")
            info, error = process.communicate()
            info_str = info.decode('utf-8').strip()
            if process.returncode != 0:
                raise RuntimeError(error.decode('utf-8'))
            elif 'run_ks2_ibl.m failed' in info_str:
                raise RuntimeError('Matlab error ks2 log below:')
                _logger.info(info_str)

            # clean up and copy: output to session/spike_sorters/ks2_matlab/probeXX (ks2_dir)
            tmp_ap_file.unlink()  # remove the uncompressed temp binary file
            scratch_dir.joinpath('temp_wh.dat').unlink()  # remove the memmapped pre-processed file
            shutil.move(scratch_dir, ks2_dir)

            self.version = self._fetch_ks2_commit_hash()
        return []  # the job will be labeled as complete with empty string


class EphysVideoCompress(tasks.Task):
    priority = 40
    level = 1

    def _run(self, **kwargs):
        # avi to mp4 compression
        command = ('ffmpeg -i {file_in} -y -nostdin -codec:v libx264 -preset slow -crf 17 '
                   '-loglevel 0 -codec:a copy {file_out}')
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

        # Run the task QC
        qc = TaskQC(self.session_path, one=self.one, log=_logger)
        qc.extractor = TaskQCExtractor(self.session_path, lazy=True, one=qc.one)
        # Extract extra datasets required for QC
        qc.extractor.data = dsets
        qc.extractor.extract_data(partial=True)

        # Aggregate and update Alyx QC fields
        qc.run(update=True)

        return out_files


class EphysSyncSpikeSorting(tasks.Task):
    priority = 90
    level = 2

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
        probe_files = spikes.probes_description(self.session_path, one=self.one)
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
                    out_files.append(bin_file.with_suffix('.ch'))
                else:
                    _logger.info(f"Compressing binary file {bin_file}")
                    out_files.append(sr.compress_file(keep_original=False))
                    out_files.append(bin_file.with_suffix('.ch'))
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
        tasks['EphysVideoCompress'] = EphysVideoCompress(self.session_path)
        tasks['EphysMtscomp'] = EphysMtscomp(self.session_path)
        # level 1
        tasks['SpikeSorting'] = SpikeSorting_KS2_Matlab(self.session_path,
                                                        parents=[tasks['EphysMtscomp']])
        tasks['EphysTrials'] = EphysTrials(self.session_path, parents=[tasks['EphysPulses']])
        tasks['EphysDLC'] = EphysDLC(self.session_path, parents=[tasks['EphysVideoCompress']])
        # level 2
        tasks['EphysSyncSpikeSorting'] = EphysSyncSpikeSorting(self.session_path, parents=[
            tasks['SpikeSorting'], tasks['EphysPulses']])
        self.tasks = tasks

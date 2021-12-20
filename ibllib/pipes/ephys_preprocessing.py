import logging
import re
import shutil
import subprocess
from collections import OrderedDict
import traceback
from pathlib import Path
import packaging.version

import numpy as np
import pandas as pd

import one.alf.io as alfio

from ibllib.misc import check_nvidia_driver
from ibllib.ephys import ephysqc, spikes, sync_probes
from ibllib.io import ffmpeg, spikeglx
from ibllib.io.video import label_from_path
from ibllib.io.extractors import ephys_fpga, ephys_passive, camera
from ibllib.pipes import tasks
from ibllib.pipes.training_preprocessing import TrainingRegisterRaw as EphysRegisterRaw
from ibllib.pipes.misc import create_alyx_probe_insertions
from ibllib.qc.alignment_qc import get_aligned_channels
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.qc.task_metrics import TaskQC
from ibllib.qc.camera import run_all_qc as run_camera_qc
from ibllib.qc.dlc import DlcQC
from ibllib.dsp import rms
from ibllib.plots.figures import dlc_qc_plot
from ibllib.plots.snapshot import ReportSnapshot
from brainbox.behavior.dlc import likelihood_threshold, get_licks, get_pupil_diameter, get_smooth_pupil_diameter

_logger = logging.getLogger("ibllib")


#  level 0
class EphysPulses(tasks.Task):
    """
    Extract Pulses from raw electrophysiology data into numpy arrays
    Perform the probes synchronisation with nidq (3B) or main probe (3A)
    """
    cpu = 2
    io_charge = 30  # this jobs reads raw ap files
    priority = 90  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything
    force = False  # whether or not to force download of missing data on local server if outputs already exist
    signature = {
        'input_files': [('*ap.meta', 'raw_ephys_data/probe*', True),
                        ('*ap.ch', 'raw_ephys_data/probe*', False),  # not necessary when we have .bin file
                        ('*ap.*bin', 'raw_ephys_data/probe*', True),
                        ('*nidq.meta', 'raw_ephys_data', True),
                        ('*nidq.ch', 'raw_ephys_data', False),  # not necessary when we have .bin file
                        ('*nidq.*bin', 'raw_ephys_data', True)],
        'output_files': [('_spikeglx_sync*.npy', 'raw_ephys_data*', True),
                         ('_spikeglx_sync.polarities*.npy', 'raw_ephys_data*', True),
                         ('_spikeglx_sync.times*.npy', 'raw_ephys_data*', True)]
    }

    def get_signatures(self, **kwargs):
        """
        Find the input and output signatures specific for local filesystem
        :return:
        """
        neuropixel_version = spikeglx.get_neuropixel_version_from_folder(self.session_path)
        probes = spikeglx.get_probes_from_folder(self.session_path)

        full_input_files = []
        for sig in self.signature['input_files']:
            if 'raw_ephys_data/probe*' in sig[1]:
                for probe in probes:
                    full_input_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))
            else:
                if neuropixel_version != '3A':
                    full_input_files.append((sig[0], sig[1], sig[2]))

        self.input_files = full_input_files

        full_output_files = []
        for sig in self.signature['output_files']:
            if neuropixel_version != '3A':
                full_output_files.append((sig[0], 'raw_ephys_data', sig[2]))
            for probe in probes:
                full_output_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))

        self.output_files = full_output_files

    def _run(self, overwrite=False):
        # outputs numpy
        syncs, out_files = ephys_fpga.extract_sync(self.session_path, overwrite=overwrite)
        for out_file in out_files:
            _logger.info(f"extracted pulses for {out_file}")

        status, sync_files = sync_probes.sync(self.session_path)
        return out_files + sync_files


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


class EphysAudio(tasks.Task):
    """
    Compresses the microphone wav file in a lossless flac file
    """
    # TESTS DONE
    cpu = 2
    priority = 10  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything
    force = False
    signature = {
        'input_files': [('_iblrig_micData.raw.wav', 'raw_behavior_data', True)],
        'output_files': [('_iblrig_micData.raw.flac', 'raw_behavior_data', True)],
    }

    def _run(self, overwrite=False):
        command = "ffmpeg -i {file_in} -y -nostdin -c:a flac -nostats {file_out}"
        file_in = next(self.session_path.rglob("_iblrig_micData.raw.wav"), None)
        if file_in is None:
            return
        file_out = file_in.with_suffix(".flac")
        status, output_file = ffmpeg.compress(file_in=file_in, file_out=file_out, command=command)
        return [output_file]


class SpikeSorting(tasks.Task):
    """
    Pykilosort 2.5 pipeline
    """
    gpu = 1
    io_charge = 70  # this jobs reads raw ap files
    priority = 60
    level = 1  # this job doesn't depend on anything
    force = True
    SHELL_SCRIPT = Path.home().joinpath(
        "Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/run_pykilosort.sh"
    )
    SPIKE_SORTER_NAME = 'pykilosort'
    PYKILOSORT_REPO = Path.home().joinpath('Documents/PYTHON/SPIKE_SORTING/pykilosort')
    signature = {
        'input_files': [],  # see setUp method for declaration of inputs
        'output_files': []  # see setUp method for declaration of inputs
    }

    @staticmethod
    def spike_sorting_signature(pname=None):
        pname = pname if pname is not None else "probe*"
        input_signature = [('*ap.meta', f'raw_ephys_data/{pname}', True),
                           ('*ap.ch', f'raw_ephys_data/{pname}', True),
                           ('*ap.cbin', f'raw_ephys_data/{pname}', True),
                           ('*nidq.meta', 'raw_ephys_data', False),
                           ('_spikeglx_sync.channels.*', 'raw_ephys_data*', True),
                           ('_spikeglx_sync.polarities.*', 'raw_ephys_data*', True),
                           ('_spikeglx_sync.times.*', 'raw_ephys_data*', True),
                           ('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
                           ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True)]
        output_signature = [('spike_sorting_pykilosort.log', f'spike_sorters/pykilosort/{pname}', True),
                            ('_iblqc_ephysTimeRmsAP.rms.npy', f'raw_ephys_data/{pname}', True),  # new ibllib 2.5
                            ('_iblqc_ephysTimeRmsAP.timestamps.npy', f'raw_ephys_data/{pname}', True)]  # new ibllib 2.5
        return input_signature, output_signature

    @staticmethod
    def _sample2v(ap_file):
        md = spikeglx.read_meta_data(ap_file.with_suffix(".meta"))
        s2v = spikeglx._conversion_sample2v_from_meta(md)
        return s2v["ap"][0]

    @staticmethod
    def _fetch_pykilosort_version(repo_path):
        init_file = Path(repo_path).joinpath('pykilosort', '__init__.py')
        version = SpikeSorting._fetch_ks2_commit_hash(repo_path)  # default
        try:
            with open(init_file) as fid:
                lines = fid.readlines()
                for line in lines:
                    if line.startswith("__version__ = "):
                        version = line.split('=')[-1].strip().replace('"', '').replace("'", '')
        except Exception:
            pass
        return f"pykilosort_{version}"

    @staticmethod
    def _fetch_pykilosort_run_version(log_file):
        with open(log_file) as fid:
            line = fid.readline()
        version = re.search('version (.*), output', line).group(1)

        return f"pykilosort_{version}"

    @staticmethod
    def _fetch_ks2_commit_hash(repo_path):
        command2run = f"git --git-dir {repo_path}/.git rev-parse --verify HEAD"
        process = subprocess.Popen(
            command2run, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        info, error = process.communicate()
        if process.returncode != 0:
            _logger.error(
                f"Can't fetch pykilsort commit hash, will still attempt to run \n"
                f"Error: {error.decode('utf-8')}"
            )
            return ""
        return info.decode("utf-8").strip()

    def setUp(self, probes=None):
        """
        Overwrite setup method to allow inputs and outputs to be only one probe
        :param probes: list of probes e.g ['probe00']
        :return:
        """
        if not probes or len(probes) == 2:
            self.signature['input_files'], self.signature['output_files'] = self.spike_sorting_signature()
        else:
            self.signature['input_files'], self.signature['output_files'] = self.spike_sorting_signature(probes[0])

        return super().setUp(probes=probes)

    def _run_pykilosort(self, ap_file):
        """
        Runs the ks2 matlab spike sorting for one probe dataset
        the raw spike sorting output is in session_path/spike_sorters/{self.SPIKE_SORTER_NAME}/probeXX folder
        (discontinued support for old spike sortings in the probe folder <1.5.5)
        :return: path of the folder containing ks2 spike sorting output
        """
        self.version = self._fetch_pykilosort_version(self.PYKILOSORT_REPO)
        label = ap_file.parts[-2]  # this is usually the probe name
        sorter_dir = self.session_path.joinpath("spike_sorters", self.SPIKE_SORTER_NAME, label)
        self.FORCE_RERUN = False
        if not self.FORCE_RERUN:
            log_file = sorter_dir.joinpath(f"spike_sorting_{self.SPIKE_SORTER_NAME}.log")
            if log_file.exists():
                run_version = self._fetch_pykilosort_run_version(log_file)
                if packaging.version.parse(run_version) > packaging.version.parse('pykilosort_ibl_1.1.0'):
                    _logger.info(f"Already ran: spike_sorting_{self.SPIKE_SORTER_NAME}.log"
                                 f" found in {sorter_dir}, skipping.")
                    return sorter_dir
                else:
                    self.FORCE_RERUN = True

        print(sorter_dir.joinpath(f"spike_sorting_{self.SPIKE_SORTER_NAME}.log"))
        # get the scratch drive from the shell script
        with open(self.SHELL_SCRIPT) as fid:
            lines = fid.readlines()
        line = [line for line in lines if line.startswith("SCRATCH_DRIVE=")][0]
        m = re.search(r"\=(.*?)(\#|\n)", line)[0]
        scratch_drive = Path(m[1:-1].strip())
        assert scratch_drive.exists()

        # clean up and create directory, this also checks write permissions
        # temp_dir has the following shape: pykilosort/ZM_3003_2020-07-29_001_probe00
        # first makes sure the tmp dir is clean
        shutil.rmtree(scratch_drive.joinpath(self.SPIKE_SORTER_NAME), ignore_errors=True)
        temp_dir = scratch_drive.joinpath(
            self.SPIKE_SORTER_NAME, "_".join(list(self.session_path.parts[-3:]) + [label])
        )
        if temp_dir.exists():  # hmmm this has to be decided, we may want to restart ?
            # But failed sessions may then clog the scratch dir and have users run out of space
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        check_nvidia_driver()
        command2run = f"{self.SHELL_SCRIPT} {ap_file} {temp_dir}"
        _logger.info(command2run)
        process = subprocess.Popen(
            command2run,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            executable="/bin/bash",
        )
        info, error = process.communicate()
        info_str = info.decode("utf-8").strip()
        _logger.info(info_str)
        if process.returncode != 0:
            error_str = error.decode("utf-8").strip()
            # try and get the kilosort log if any
            for log_file in temp_dir.rglob('*_kilosort.log'):
                with open(log_file) as fid:
                    log = fid.read()
                    _logger.error(log)
                break
            raise RuntimeError(f"{self.SPIKE_SORTER_NAME} {info_str}, {error_str}")

        shutil.copytree(temp_dir.joinpath('output'), sorter_dir, dirs_exist_ok=True)
        shutil.rmtree(temp_dir, ignore_errors=True)
        return sorter_dir

    def _run(self, probes=None):
        """
        Multiple steps. For each probe:
        - Runs ks2 (skips if it already ran)
        - synchronize the spike sorting
        - output the probe description files
        :param probes: (list of str) if provided, will only run spike sorting for specified probe names
        :return: list of files to be registered on database
        """
        efiles = spikeglx.glob_ephys_files(self.session_path)
        ap_files = [(ef.get("ap"), ef.get("label")) for ef in efiles if "ap" in ef.keys()]
        out_files = []
        for ap_file, label in ap_files:
            if isinstance(probes, list) and label not in probes:
                continue
            try:
                # if the file is part of  a sequence, handles the run accordingly
                sequence_file = ap_file.parent.joinpath(ap_file.stem.replace('ap', 'sequence.json'))
                # temporary just skips for now
                if sequence_file.exists():
                    continue
                ks2_dir = self._run_pykilosort(ap_file)  # runs ks2, skips if it already ran
                probe_out_path = self.session_path.joinpath("alf", label, self.SPIKE_SORTER_NAME)
                shutil.rmtree(probe_out_path, ignore_errors=True)
                probe_out_path.mkdir(parents=True, exist_ok=True)
                spikes.ks2_to_alf(
                    ks2_dir,
                    bin_path=ap_file.parent,
                    out_path=probe_out_path,
                    bin_file=ap_file,
                    ampfactor=self._sample2v(ap_file),
                )
                logfile = ks2_dir.joinpath(f"spike_sorting_{self.SPIKE_SORTER_NAME}.log")
                if logfile.exists():
                    shutil.copyfile(logfile, probe_out_path.joinpath(
                        f"_ibl_log.info_{self.SPIKE_SORTER_NAME}.log"))
                out, _ = spikes.sync_spike_sorting(ap_file=ap_file, out_path=probe_out_path)
                out_files.extend(out)
                # convert ks2_output into tar file and also register
                # Make this in case spike sorting is in old raw_ephys_data folders, for new
                # sessions it should already exist
                tar_dir = self.session_path.joinpath(
                    'spike_sorters', self.SPIKE_SORTER_NAME, label)
                tar_dir.mkdir(parents=True, exist_ok=True)
                out = spikes.ks2_to_tar(ks2_dir, tar_dir, force=self.FORCE_RERUN)
                out_files.extend(out)

                if self.one:
                    eid = self.one.path2eid(self.session_path, query_type='remote')
                    ins = self.one.alyx.rest('insertions', 'list', session=eid, name=label)
                    if len(ins) != 0:
                        resolved = ins[0].get('json', {'temp': 0}).get('extended_qc', {'temp': 0}). \
                            get('alignment_resolved', False)
                        if resolved:
                            chns = np.load(probe_out_path.joinpath('channels.localCoordinates.npy'))
                            out = get_aligned_channels(ins[0], chns, one=self.one, save_dir=probe_out_path)
                            out_files.extend(out)

            except BaseException:
                _logger.error(traceback.format_exc())
                self.status = -1
                continue
        probe_files = spikes.probes_description(self.session_path, one=self.one)
        return out_files + probe_files

    def get_signatures(self, probes=None, **kwargs):
        """
        This transforms all wildcards in collection to exact match
        :param probes:
        :return:
        """

        neuropixel_version = spikeglx.get_neuropixel_version_from_folder(self.session_path)
        probes = probes or spikeglx.get_probes_from_folder(self.session_path)

        full_input_files = []
        for sig in self.signature['input_files']:
            if 'raw_ephys_data*' in sig[1]:
                if neuropixel_version != '3A':
                    full_input_files.append((sig[0], 'raw_ephys_data', sig[2]))
                for probe in probes:
                    full_input_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))

            elif 'raw_ephys_data/probe*' in sig[1]:
                for probe in probes:
                    full_input_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))
            elif 'raw_ephys_data' in sig[1]:
                if neuropixel_version != '3A':
                    full_input_files.append((sig[0], 'raw_ephys_data', sig[2]))
            else:
                full_input_files.append(sig)

        self.input_files = full_input_files

        full_output_files = []
        for sig in self.signature['output_files']:
            if 'probe*' in sig[1]:
                for probe in probes:
                    col = sig[1].split('/')[:-1] + [probe]
                    full_output_files.append((sig[0], '/'.join(col), sig[2]))
            else:
                full_input_files.append(sig)

        self.output_files = full_output_files


class EphysVideoCompress(tasks.Task):
    priority = 40
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
        # need to detect the number of cameras
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


class EphysVideoSyncQc(tasks.Task):
    priority = 40
    level = 2
    force = True
    signature = {
        'input_files': [('_iblrig_*Camera.raw.mp4', 'raw_video_data', True),
                        ('_iblrig_*Camera.timestamps.npy', 'raw_video_data', False),
                        ('_iblrig_*Camera.frameData.bin', 'raw_video_data', False),
                        ('_iblrig_*Camera.GPIO.bin', 'raw_video_data', False),
                        ('_iblrig_*Camera.frame_counter.bin', 'raw_video_data', False),
                        ('_iblrig_taskData.raw.*', 'raw_behavior_data', True),
                        ('_iblrig_taskSettings.raw.*', 'raw_behavior_data', True),
                        ('_spikeglx_sync.channels.*', 'raw_ephys_data*', True),
                        ('_spikeglx_sync.polarities.*', 'raw_ephys_data*', True),
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


#  level 1
class EphysTrials(tasks.Task):
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

    def _run(self):
        dsets, out_files = ephys_fpga.extract_all(self.session_path, save=True)

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


class EphysCellsQc(tasks.Task):
    priority = 90
    level = 3
    force = False

    signature = {
        'input_files': [('spikes.times.npy', 'alf/probe*', True),
                        ('spikes.clusters.npy', 'alf/probe*', True),
                        ('spikes.amps.npy', 'alf/probe*', True),
                        ('spikes.depths.npy', 'alf/probe*', True),
                        ('clusters.channels.npy', 'alf/probe*', True)],
        'output_files': [('clusters.metrics.pqt', 'alf/probe*', True)]
    }

    def _compute_cell_qc(self, folder_probe):
        """
        Computes the cell QC given an extracted probe alf path
        :param folder_probe: folder
        :return:
        """
        # compute the straight qc
        _logger.info(f"Computing cluster qc for {folder_probe}")
        spikes = alfio.load_object(folder_probe, 'spikes')
        clusters = alfio.load_object(folder_probe, 'clusters')
        df_units, drift = ephysqc.spike_sorting_metrics(
            spikes.times, spikes.clusters, spikes.amps, spikes.depths,
            cluster_ids=np.arange(clusters.channels.size))
        # if the ks2 labels file exist, load them and add the column
        file_labels = folder_probe.joinpath('cluster_KSLabel.tsv')
        if file_labels.exists():
            ks2_labels = pd.read_csv(file_labels, sep='\t')
            ks2_labels.rename(columns={'KSLabel': 'ks2_label'}, inplace=True)
            df_units = pd.concat(
                [df_units, ks2_labels['ks2_label'].reindex(df_units.index)], axis=1)
        # save as parquet file
        df_units.to_parquet(folder_probe.joinpath("clusters.metrics.pqt"))
        return folder_probe.joinpath("clusters.metrics.pqt"), df_units, drift

    def _label_probe_qc(self, folder_probe, df_units, drift):
        """
        Labels the json field of the alyx corresponding probe insertion
        :param folder_probe:
        :param df_units:
        :param drift:
        :return:
        """
        eid = self.one.path2eid(self.session_path, query_type='remote')
        # the probe name is the first folder after alf: {session_path}/alf/{probe_name}/{spike_sorter_name}
        probe_name = Path(folder_probe).relative_to(self.session_path.joinpath('alf')).parts[0]
        pdict = self.one.alyx.rest('insertions', 'list', session=eid, name=probe_name, no_cache=True)
        if len(pdict) != 1:
            _logger.warning(f'No probe found for probe name: {probe_name}')
            return
        isok = df_units['label'] == 1
        qcdict = {'n_units': int(df_units.shape[0]),
                  'n_units_qc_pass': int(np.sum(isok)),
                  'firing_rate_max': np.max(df_units['firing_rate'][isok]),
                  'firing_rate_median': np.median(df_units['firing_rate'][isok]),
                  'amplitude_max_uV': np.max(df_units['amp_max'][isok]) * 1e6,
                  'amplitude_median_uV': np.max(df_units['amp_median'][isok]) * 1e6,
                  'drift_rms_um': rms(drift['drift_um']),
                  }
        file_wm = folder_probe.joinpath('_kilosort_whitening.matrix.npy')
        if file_wm.exists():
            wm = np.load(file_wm)
            qcdict['whitening_matrix_conditioning'] = np.linalg.cond(wm)
        # groom qc dict (this function will eventually go directly into the json field update)
        for k in qcdict:
            if isinstance(qcdict[k], np.int64):
                qcdict[k] = int(qcdict[k])
            elif isinstance(qcdict[k], float):
                qcdict[k] = np.round(qcdict[k], 2)
        self.one.alyx.json_field_update("insertions", pdict[0]["id"], "json", qcdict)

    def _run(self):
        """
        Post spike-sorting quality control at the cluster level.
        Outputs a QC table in the clusters ALF object and labels corresponding probes in Alyx
        """
        files_spikes = Path(self.session_path).joinpath('alf').rglob('spikes.times.npy')
        folder_probes = [f.parent for f in files_spikes]
        out_files = []
        for folder_probe in folder_probes:
            try:
                qc_file, df_units, drift = self._compute_cell_qc(folder_probe)
                out_files.append(qc_file)
                self._label_probe_qc(folder_probe, df_units, drift)
            except Exception:
                _logger.error(traceback.format_exc())
                self.status = -1
                continue
        return out_files

    def get_signatures(self, **kwargs):
        files_spikes = Path(self.session_path).joinpath('alf').rglob('spikes.times.npy')
        folder_probes = [f.parent for f in files_spikes]

        full_input_files = []
        for sig in self.signature['input_files']:
            for folder in folder_probes:
                full_input_files.append((sig[0], str(folder.relative_to(self.session_path)), sig[2]))

        self.input_files = full_input_files

        full_output_files = []
        for sig in self.signature['output_files']:
            for folder in folder_probes:
                full_output_files.append((sig[0], str(folder.relative_to(self.session_path)), sig[2]))

        self.output_files = full_output_files


class EphysMtscomp(tasks.Task):
    priority = 50  # ideally after spike sorting
    level = 0
    force = False
    signature = {
        'input_files': [('*ap.meta', 'raw_ephys_data/probe*', True),
                        ('*ap.*bin', 'raw_ephys_data/probe*', True),
                        ('*lf.meta', 'raw_ephys_data/probe*', False),  # NP2 doesn't have lf files
                        ('*lf.*bin', 'raw_ephys_data/probe*', False),  # NP2 doesn't have lf files
                        ('*nidq.meta', 'raw_ephys_data', True),
                        ('*nidq.*bin', 'raw_ephys_data', True)],
        'output_files': [('*ap.meta', 'raw_ephys_data/probe*', True),
                         ('*ap.cbin', 'raw_ephys_data/probe*', False),  # may not be present on local server anymore
                         ('*ap.ch', 'raw_ephys_data/probe*', True),
                         ('*lf.meta', 'raw_ephys_data/probe*', False),  # NP2 doesn't have lf files # TODO detect from meta
                         ('*lf.cbin', 'raw_ephys_data/probe*', False),  # may not be present on local server anymore
                         ('*lf.ch', 'raw_ephys_data/probe*', False),
                         ('*nidq.meta', 'raw_ephys_data', True),
                         ('*nidq.cbin', 'raw_ephys_data', False),  # may not be present on local server anymore
                         ('*nidq.ch', 'raw_ephys_data', True)]
    }

    def _run(self):
        """
        Compress ephys files looking for `compress_ephys.flag` within the probes folder
        Original bin file will be removed
        The registration flag created contains targeted file names at the root of the session
        """

        out_files = []
        ephys_files = spikeglx.glob_ephys_files(self.session_path)
        ephys_files += spikeglx.glob_ephys_files(self.session_path, ext="ch")
        ephys_files += spikeglx.glob_ephys_files(self.session_path, ext="meta")

        for ef in ephys_files:
            for typ in ["ap", "lf", "nidq"]:
                bin_file = ef.get(typ)
                if not bin_file:
                    continue
                if bin_file.suffix.find("bin") == 1:
                    with spikeglx.Reader(bin_file) as sr:
                        if sr.is_mtscomp:
                            out_files.append(bin_file)
                        else:
                            _logger.info(f"Compressing binary file {bin_file}")
                            out_files.append(sr.compress_file(keep_original=False))
                            out_files.append(bin_file.with_suffix('.ch'))
                else:
                    out_files.append(bin_file)

        return out_files

    def get_signatures(self, **kwargs):
        neuropixel_version = spikeglx.get_neuropixel_version_from_folder(self.session_path)
        probes = spikeglx.get_probes_from_folder(self.session_path)

        full_input_files = []
        for sig in self.signature['input_files']:
            if 'raw_ephys_data/probe*' in sig[1]:
                for probe in probes:
                    full_input_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))
            else:
                if neuropixel_version != '3A':
                    full_input_files.append((sig[0], sig[1], sig[2]))

        self.input_files = full_input_files

        full_output_files = []
        for sig in self.signature['output_files']:
            if 'raw_ephys_data/probe*' in sig[1]:
                for probe in probes:
                    full_output_files.append((sig[0], f'raw_ephys_data/{probe}', sig[2]))
            else:
                if neuropixel_version != '3A':
                    full_output_files.append((sig[0], sig[1], sig[2]))

        self.output_files = full_output_files


class EphysDLC(tasks.Task):
    gpu = 1
    cpu = 4
    io_charge = 90
    level = 2

    def _run(self):
        """empty placeholder for job creation only"""
        pass


class EphysPostDLC(tasks.Task):
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
                                 ('_ibl_bodyCamera.times.npy', 'alf', True)],
                 # More files are required for all panels of the DLC QC plot to function
                 'output_files': [('_ibl_leftCamera.features.pqt', 'alf', True),
                                  ('_ibl_rightCamera.features.pqt', 'alf', True),
                                  ('licks.times.npy', 'alf', True)]
                 }

    def _run(self, overwrite=False, run_qc=True, plot_qc=True):
        # Check if output files exist locally
        exist, output_files = self.assert_expected(self.signature['output_files'], silent=True)
        if exist and not overwrite:
            _logger.warning('EphysPostDLC outputs exist and overwrite=False, skipping computations of outputs.')
        else:
            if exist and overwrite:
                _logger.warning('EphysPostDLC outputs exist and overwrite=True, overwriting existing outputs.')
            # Find all available dlc traces and dlc times
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
                    except StopIteration:
                        _logger.error(f'No camera.times found for {cam} camera. '
                                      f'Computations using camera.times will be skipped')
                        self.status = -1
                        times = False

                    # These features are only computed from left and right cam
                    if cam in ('left', 'right'):
                        features = pd.DataFrame()
                        # If camera times are available, get the lick time stamps for combined array
                        if times:
                            _logger.info(f"Computing lick times for {cam} camera.")
                            combined_licks.append(get_licks(dlc_thresh, dlc_t))
                        else:
                            _logger.warning(f"Skipping lick times for {cam} camera as no camera.times available.")
                        # Compute pupil diameter, raw and smoothed
                        _logger.info(f"Computing raw pupil diameter for {cam} camera.")
                        features['pupilDiameter_raw'] = get_pupil_diameter(dlc_thresh)
                        _logger.info(f"Computing smooth pupil diameter for {cam} camera.")
                        features['pupilDiameter_smooth'] = get_smooth_pupil_diameter(features['pupilDiameter_raw'], cam)
                        # Safe to pqt
                        features_file = Path(self.session_path).joinpath('alf', f'_ibl_{cam}Camera.features.pqt')
                        features.to_parquet(features_file)
                        output_files.append(features_file)

                    # For all cams, compute DLC qc if times available
                    if times and run_qc:
                        # Setting download_data to False because at this point the data should be there
                        qc = DlcQC(self.session_path, side=cam, one=self.one, download_data=False)
                        qc.run(update=True)
                    else:
                        if not times:
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
                fig = dlc_qc_plot(self.one.path2eid(self.session_path), one=self.one)
                fig.savefig(fig_path)
                snp = ReportSnapshot(self.session_path, session_id, one=self.one)
                snp.outputs = [fig_path]
                snp.register_images(widths=['orig'],
                                    function=str(dlc_qc_plot.__module__) + '.' + str(dlc_qc_plot.__name__))
            except BaseException:
                _logger.error('Could not create and/or upload DLC QC Plot')
                _logger.error(traceback.format_exc())
                self.status = -1

        return output_files


class EphysPassive(tasks.Task):
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


class EphysExtractionPipeline(tasks.Pipeline):
    label = __name__

    def __init__(self, session_path=None, **kwargs):
        super(EphysExtractionPipeline, self).__init__(session_path, **kwargs)
        tasks = OrderedDict()
        self.session_path = session_path
        # level 0
        tasks["EphysRegisterRaw"] = EphysRegisterRaw(self.session_path)
        tasks["EphysPulses"] = EphysPulses(self.session_path)
        tasks["EphysRawQC"] = RawEphysQC(self.session_path)
        tasks["EphysAudio"] = EphysAudio(self.session_path)
        tasks["EphysMtscomp"] = EphysMtscomp(self.session_path)
        tasks['EphysVideoCompress'] = EphysVideoCompress(self.session_path)
        # level 1
        tasks["SpikeSorting"] = SpikeSorting(
            self.session_path, parents=[tasks["EphysMtscomp"], tasks["EphysPulses"]])
        tasks["EphysTrials"] = EphysTrials(self.session_path, parents=[tasks["EphysPulses"]])

        tasks["EphysPassive"] = EphysPassive(self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysVideoSyncQc"] = EphysVideoSyncQc(
            self.session_path, parents=[tasks["EphysVideoCompress"], tasks["EphysPulses"], tasks["EphysTrials"]])
        tasks["EphysCellsQc"] = EphysCellsQc(self.session_path, parents=[tasks["SpikeSorting"]])
        tasks["EphysDLC"] = EphysDLC(self.session_path, parents=[tasks["EphysVideoCompress"]])
        # level 3
        tasks["EphysPostDLC"] = EphysPostDLC(self.session_path, parents=[tasks["EphysDLC"]])
        self.tasks = tasks

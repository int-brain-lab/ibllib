# These all need to take in a probe argument apart from the Ephys register raw
import logging
from pathlib import Path
import subprocess
import re
import packaging
import shutil
import numpy as np

import spikeglx
import neuropixel

from ibllib.misc import check_nvidia_driver
from ibllib.pipes import base_tasks
from ibllib.ephys import ephysqc, spikes

from ibllib.plots.figures import LfpPlots, ApPlots, BadChannelsAp
from ibllib.plots.figures import SpikeSorting as SpikeSortingPlots


import traceback

_logger = logging.getLogger("ibllib")
# EphysPulses

class EphysRegisterRaw(base_tasks.EphysTask):
    """
    Task to register probe desription file and also make the alyx probe insertions
    """
    probe_files = spikes.probes_description(self.session_path, one=self.one)

class EphysCompressNP1(base_tasks.EphysTask):
    def dynamic_signatures(self):
        input_signatures = [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*ap.*bin', f'{self.device_collection}/{self.pname}', True),
                            ('*lf.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*lf.*bin', f'{self.device_collection}/{self.pname}', True)]

        output_signatures = [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*ap.cbin', f'{self.device_collection}/{self.pname}', True),
                             ('*ap.ch', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.cbin', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.ch', f'{self.device_collection}/{self.pname}', True)]

        return input_signatures, output_signatures

    def _run(self):

        # TODO test
        out_files = []
        ephys_files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname))

        for ef in ephys_files:
            for typ in ["ap", "lf"]:
                bin_file = ef.get(typ)
                if not bin_file:
                    continue
                if bin_file.suffix.find("bin") == 1:
                    sr = spikeglx.Reader(bin_file)
                    if sr.is_mtscomp:
                        out_files.append(bin_file)
                    else:
                        _logger.info(f"Compressing binary file {bin_file}")
                        out_files.append(sr.compress_file(keep_original=False))
                else:
                    out_files.append(bin_file)

            out_files.append(bin_file.with_suffix('.ch'))
            out_files.append(bin_file.with_suffix('.meta'))

        return out_files


class EphysCompressNP21(base_tasks.EphysTask):
    def dynamic_signatures(self):
        input_signatures = [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*ap.*bin', f'{self.device_collection}/{self.pname}', True),
                            ('*lf.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*lf.*bin', f'{self.device_collection}/{self.pname}', True)]

        output_signatures = [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*ap.cbin', f'{self.device_collection}/{self.pname}', True),
                             ('*ap.ch', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.cbin', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.ch', f'{self.device_collection}/{self.pname}', True)]

        return input_signatures, output_signatures

    def _run(self):

        files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname))
        assert len(files) == 1  # This will fail if the session is split
        bin_file = files[0].get('ap', None)

        np_conv = neuropixel.NP2Converter(bin_file, compress=True)
        np_conv_status = np_conv.process()
        out_files = np_conv.get_processed_files()

        if np_conv_status == 1:
            return out_files
        else:
            return


class EphysCompressNP24(base_tasks.EphysTask):
    def dynamic_signatures(self):
        input_signatures = [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*ap.*bin', f'{self.device_collection}/{self.pname}', True),
                            ('*lf.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*lf.*bin', f'{self.device_collection}/{self.pname}', True)]

        pextra = ['a', 'b', 'c', 'd']
        output_signatures = [('*ap.meta', f'{self.device_collection}/{self.pname}_{pext}', True) for pext in pextra] + \
                            [('*ap.cbin', f'{self.device_collection}/{self.pname}_{pext}', True) for pext in pextra] + \
                            [('*ap.ch', f'{self.device_collection}/{self.pname}_{pext}', True) for pext in pextra] + \
                            [('*lf.meta', f'{self.device_collection}/{self.pname}_{pext}', True) for pext in pextra] + \
                            [('*lf.cbin', f'{self.device_collection}/{self.pname}_{pext}', True) for pext in pextra] + \
                            [('*lf.ch', f'{self.device_collection}/{self.pname}_{pext}', True) for pext in pextra]

        return input_signatures, output_signatures

    def _run(self):

        files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname))
        assert len(files) == 1  # This will fail if the session is split
        bin_file = files[0].get('ap', None)

        np_conv = neuropixel.NP2Converter(bin_file, post_check=True, compress=True, delete_original=False) # TODO once we happy change this to True
        # TODO delete original deletes the whole folder of bin_file (I don't think we want to do that)
        np_conv_status = np_conv.process()
        out_files = np_conv.get_processed_files()

        if np_conv_status == 1:
            return out_files
        else:
            return


class EphysPulses(base_tasks.EphysTask):
    # For now just have this as the same task as before
    # TODO Need to register the wiring file for each probe

    # TODO needs some love


class RawEphysQC(base_tasks.EphysTask):

    cpu = 2
    io_charge = 30  # this jobs reads raw ap files
    priority = 10  # a lot of jobs depend on this one
    level = 0  # this job doesn't depend on anything
    force = False
    def dynamic_signatures(self):

        input_signatures =  [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.ch', f'{self.device_collection}/{self.pname}', False),
                             ('*lf.*bin', f'{self.device_collection}/{self.pname}', False)]

        output_signatures = [('_iblqc_ephysChannels.apRMS.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysChannels.rawSpikeRates.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysChannels.labels.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysSpectralDensityLF.freqs.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysSpectralDensityLF.power.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysSpectralDensityAP.freqs.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysSpectralDensityAP.power.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysTimeRmsLF.rms.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysTimeRmsLF.timestamps.npy', f'{self.device_collection}/{self.pname}', True)]

        return input_signatures, output_signatures
    # TODO make sure this works with NP2 probes (at the moment not sure it will due to raiseError mapping)
    def _run(self, overwrite=False):

        eid = self.one.path2eid(self.session_path)
        probe = self.one.alyx.rest('insertions', 'list', session=eid, probe=self.pname)

        # We expect there to only be one probe
        if len(probe) != 1:
            _logger.warning(f"{self.pname} for {eid} not found")
            self.status = -1
            return

        pid = probe[0]['id']
        qc_files = []
        _logger.info(f"\nRunning QC for probe insertion {self.pname}")
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

        return qc_files

class SpikeSorting(base_tasks.EphysTask):
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

    def dynamic_signatures(self):

        input_signatures =  [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*ap.cbin', f'{self.device_collection}/{self.pname}', True),
                             ('*ap.ch', f'{self.device_collection}/{self.pname}', True),
                             ('*sync.npy', f'{self.device_collection}/{self.pname}', True )]

        output_signatures = [('spike_sorting_pykilosort.log', f'spike_sorters/pykilosort/{self.pname}', True),
                            ('_iblqc_ephysTimeRmsAP.rms.npy', f'{self.device_collection}/{self.pname}', True),
                            ('_iblqc_ephysTimeRmsAP.timestamps.npy', f'{self.device_collection}/{self.pname}', True)]

        return input_signatures, output_signatures

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

        # TODO need to figure this out
        for qcfile in temp_dir.glob('_iblqc_*AP*'):
            shutil.move(qcfile, ap_file.parent.joinpath(qcfile.name))

        return sorter_dir

    def _run(self):
        """
        Multiple steps. For each probe:
        - Runs ks2 (skips if it already ran)
        - synchronize the spike sorting
        - output the probe description files
        :param probes: (list of str) if provided, will only run spike sorting for specified probe names
        :return: list of files to be registered on database
        """
        efiles = spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname))
        ap_files = [(ef.get("ap"), ef.get("label")) for ef in efiles if "ap" in ef.keys()]
        out_files = []
        for ap_file, label in ap_files:

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
                # For now leave the syncing here
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
                    ins = self.one.alyx.rest('insertions', 'list', session=eid, name=label, query_type='remote')
                    if len(ins) != 0:
                        _logger.info("Creating SpikeSorting QC plots")
                        plot_task = ApPlots(ins[0]['id'], session_path=self.session_path, one=self.one)
                        _ = plot_task.run()
                        self.plot_tasks.append(plot_task)

                        plot_task = SpikeSortingPlots(ins[0]['id'], session_path=self.session_path, one=self.one)
                        _ = plot_task.run(collection=str(probe_out_path.relative_to(self.session_path)))
                        self.plot_tasks.append(plot_task)

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

        return out_files







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




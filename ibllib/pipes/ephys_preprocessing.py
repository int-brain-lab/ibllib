import logging
import re
import shutil
import subprocess
from collections import OrderedDict
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import mtscomp
import alf.io
from ibllib.ephys import ephysqc, spikes, sync_probes
from ibllib.io import ffmpeg, spikeglx
from ibllib.io.extractors import ephys_fpga, ephys_passive
from ibllib.pipes import tasks
from ibllib.pipes.training_preprocessing import TrainingRegisterRaw as EphysRegisterRaw
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.qc.task_metrics import TaskQC
from ibllib.dsp import rms

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
        command = "ffmpeg -i {file_in} -y -nostdin -c:a flac -nostats {file_out}"
        file_in = next(self.session_path.rglob("_iblrig_micData.raw.wav"), None)
        if file_in is None:
            return
        file_out = file_in.with_suffix(".flac")
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
    def _sample2v(ap_file):
        md = spikeglx.read_meta_data(ap_file.with_suffix(".meta"))
        s2v = spikeglx._conversion_sample2v_from_meta(md)
        return s2v["ap"][0]

    @staticmethod
    def _fetch_ks2_commit_hash():
        command2run = "git --git-dir ~/Documents/MATLAB/Kilosort2/.git rev-parse --verify HEAD"
        process = subprocess.Popen(
            command2run, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        info, error = process.communicate()
        if process.returncode != 0:
            _logger.error(
                f"Can't fetch matlab ks2 commit hash, will still attempt to run \n"
                f"Error: {error.decode('utf-8')}"
            )
            return ""
        return info.decode("utf-8").strip()

    def _run_ks2(self, ap_file):
        """
        Runs the ks2 matlab spike sorting for one probe dataset
        the spike sorting output can either be with the probe (<1.5.5) or in the
        session_path/spike_sorters/ks2_matlab/probeXX folder
        :return: path of the folder containing ks2 spike sorting output
        """
        label = ap_file.parts[-2]
        if ap_file.parent.joinpath("spike_sorting_ks2.log").exists():
            _logger.info(f"Already ran: spike_sorting_ks2.log found for {ap_file}, skipping.")
            return ap_file.parent
        ks2_dir = self.session_path.joinpath("spike_sorters", "ks2_matlab", label)
        if ks2_dir.joinpath("spike_sorting_ks2.log").exists():
            _logger.info(f"Already ran: spike_sorting_ks2.log found in {ks2_dir}, skipping.")
            return ks2_dir
        # get the scratch drive from the shell script
        SHELL_SCRIPT = Path.home().joinpath(
            "Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/task_ks2_matlab.sh"
        )
        with open(SHELL_SCRIPT) as fid:
            lines = fid.readlines()
        line = [line for line in lines if line.startswith("SCRATCH_DRIVE=")][0]
        m = re.search(r"\=(.*?)(\#|\n)", line)[0]
        scratch_drive = Path(m[1:-1].strip())
        assert scratch_drive.exists()

        # clean up and create directory, this also checks write permissions
        # scratch dir has the following shape: ks2m/ZM_3003_2020-07-29_001_probe00
        # first makes sure the tmp dir is clean
        shutil.rmtree(scratch_drive.joinpath("ks2m"), ignore_errors=True)
        scratch_dir = scratch_drive.joinpath(
            "ks2m", "_".join(list(self.session_path.parts[-3:]) + [label])
        )
        if scratch_dir.exists():
            shutil.rmtree(scratch_dir, ignore_errors=True)
        scratch_dir.mkdir(parents=True, exist_ok=True)

        # decompresses using mtscomp
        tmp_ap_file = scratch_dir.joinpath(ap_file.name).with_suffix(".bin")
        mtscomp.decompress(cdata=ap_file, out=tmp_ap_file)

        # run matlab spike sorting: with R2019a, it would be much easier to run with
        # -batch option as matlab errors are redirected to stderr automatically
        command2run = f"{SHELL_SCRIPT} {scratch_dir}"
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
        if process.returncode != 0:
            raise RuntimeError(error.decode("utf-8"))
        elif "run_ks2_ibl.m failed" in info_str:
            raise RuntimeError("Matlab error ks2 log below:")
            _logger.info(info_str)

        # clean up and copy: output to session/spike_sorters/ks2_matlab/probeXX (ks2_dir)
        tmp_ap_file.unlink()  # remove the uncompressed temp binary file
        scratch_dir.joinpath("temp_wh.dat").unlink()  # remove the memmapped pre-processed file
        shutil.move(scratch_dir, ks2_dir)

        self.version = self._fetch_ks2_commit_hash()
        return ks2_dir

    def _run(self, overwrite=False):
        """
        Multiple steps. For each probe:
        - Runs ks2 (skips if it already ran)
        - synchronize the spike sorting
        - output the probe description files
        :param overwrite:
        :return: list of files to be registered on database
        """
        efiles = spikeglx.glob_ephys_files(self.session_path)
        ap_files = [(ef.get("ap"), ef.get("label")) for ef in efiles if "ap" in ef.keys()]
        out_files = []
        for ap_file, label in ap_files:
            try:
                ks2_dir = self._run_ks2(ap_file)  # runs ks2, skips if it already ran
                probe_out_path = self.session_path.joinpath("alf", label)
                probe_out_path.mkdir(parents=True, exist_ok=True)
                spikes.ks2_to_alf(
                    ks2_dir,
                    bin_path=ap_file.parent,
                    out_path=probe_out_path,
                    bin_file=ap_file,
                    ampfactor=self._sample2v(ap_file),
                )
                out, _ = spikes.sync_spike_sorting(ap_file=ap_file, out_path=probe_out_path)
                out_files.extend(out)
                # convert ks2_output into tar file and also register
                # Make this in case spike sorting is in old raw_ephys_data folders, for new
                # sessions it should already exist
                tar_dir = self.session_path.joinpath('spike_sorters', 'ks2_matlab', label)
                tar_dir.mkdir(parents=True, exist_ok=True)
                out = spikes.ks2_to_tar(ks2_dir, tar_dir)
                out_files.extend(out)
            except BaseException:
                _logger.error(traceback.format_exc())
                self.status = -1
                continue
        probe_files = spikes.probes_description(self.session_path, one=self.one)
        return out_files + probe_files


class EphysVideoCompress(tasks.Task):
    priority = 40
    level = 1

    def _run(self, **kwargs):
        # avi to mp4 compression
        command = (
            "ffmpeg -i {file_in} -y -nostdin -codec:v libx264 -preset slow -crf 17 "
            "-loglevel 0 -codec:a copy {file_out}"
        )
        output_files = ffmpeg.iblrig_video_compression(self.session_path, command)
        if len(output_files) == 0:
            output_files = None  # labels the task as empty if no output
        return output_files


#  level 1
class EphysTrials(tasks.Task):
    priority = 90
    level = 1

    def _behaviour_criterion(self):
        """
        Computes and update the behaviour criterion on Alyx
        """
        import alf.io
        from brainbox.behavior import training

        if self.one is None:  # if no instance of Alyx is provided, do not touch any database
            return
        trials = alf.io.load_object(self.session_path.joinpath("alf"), "trials")
        good_enough = training.criterion_delay(
            n_trials=trials["intervals"].shape[0],
            perf_easy=training.compute_performance_easy(trials),
        )
        eid = self.one.eid_from_path(self.session_path)
        self.one.alyx.json_field_update(
            "sessions", eid, "extended_qc", {"behavior": int(good_enough)}
        )

    def _run(self):
        dsets, out_files = ephys_fpga.extract_all(self.session_path, save=True)
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


class EphysCellsQc(tasks.Task):
    priority = 90
    level = 3

    def _compute_cell_qc(self, folder_probe):
        """
        Computes the cell QC given an extracted probe alf path
        :param folder_probe: folder
        :return:
        """
        # compute the straight qc
        _logger.info(f"Computing cluster qc for {folder_probe}")
        spikes = alf.io.load_object(folder_probe, 'spikes')
        clusters = alf.io.load_object(folder_probe, 'clusters')
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
        eid = self.one.eid_from_path(self.session_path)
        pdict = self.one.alyx.rest('insertions', 'list', session=eid, name=folder_probe.parts[-1])
        if len(pdict) != 1:
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
            except BaseException:
                _logger.error(traceback.format_exc())
                self.status = -1
                continue
        return out_files


class EphysMtscomp(tasks.Task):
    priority = 50  # ideally after spike sorting
    level = 0

    def _run(self):
        """
        Compress ephys files looking for `compress_ephys.flag` whithin the probes folder
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
                    sr = spikeglx.Reader(bin_file)
                    if sr.is_mtscomp:
                        out_files.append(bin_file)
                    else:
                        _logger.info(f"Compressing binary file {bin_file}")
                        out_files.append(sr.compress_file(keep_original=False))
                        out_files.append(bin_file.with_suffix('.ch'))
                else:
                    out_files.append(bin_file)

        return out_files


class EphysDLC(tasks.Task):
    gpu = 1
    cpu = 4
    io_charge = 90
    level = 1

    def _run(self):
        """empty placeholder for job creation only"""
        pass


class EphysPassive(tasks.Task):
    cpu = 1
    io_charge = 90
    level = 1

    def _run(self):
        """returns a list of pathlib.Paths. """
        data, paths = ephys_passive.PassiveChoiceWorld(self.session_path).extract(save=True)
        if any([x is None for x in paths]):
            self.status = -1
        # Register?
        return paths


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
        tasks["EphysVideoCompress"] = EphysVideoCompress(self.session_path)
        tasks["EphysMtscomp"] = EphysMtscomp(self.session_path)
        # level 1
        tasks['SpikeSorting'] = SpikeSorting_KS2_Matlab(
            self.session_path, parents=[tasks['EphysMtscomp'], tasks['EphysPulses']])
        tasks['EphysTrials'] = EphysTrials(self.session_path, parents=[tasks['EphysPulses']])
        tasks['EphysDLC'] = EphysDLC(self.session_path, parents=[tasks['EphysVideoCompress']])
        tasks["EphysPassive"] = EphysPassive(self.session_path, parents=[tasks["EphysPulses"]])
        # level 2
        tasks["EphysCellsQc"] = EphysCellsQc(self.session_path, parents=[tasks["SpikeSorting"]])
        self.tasks = tasks

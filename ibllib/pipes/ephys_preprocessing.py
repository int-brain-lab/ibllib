"""(Deprecated) Electrophysiology data preprocessing tasks.

These tasks are part of the old pipeline. This module has been replaced by the `ephys_tasks` module
and the dynamic pipeline.
"""
import logging
import subprocess
from collections import OrderedDict
import traceback
from pathlib import Path
import warnings

import cv2
import numpy as np
import pandas as pd

import one.alf.io as alfio
from ibldsp.utils import rms
import spikeglx

from ibllib.misc import check_nvidia_driver
from ibllib.ephys import ephysqc, sync_probes
from ibllib.io import ffmpeg
from ibllib.io.video import label_from_path, assert_valid_label
from ibllib.io.extractors import ephys_fpga, ephys_passive, camera
from ibllib.pipes import tasks, base_tasks
import ibllib.pipes.training_preprocessing as tpp
from ibllib.pipes.misc import create_alyx_probe_insertions
from ibllib.pipes.ephys_tasks import SpikeSorting
from ibllib.qc.task_extractors import TaskQCExtractor
from ibllib.qc.task_metrics import TaskQC
from ibllib.qc.camera import run_all_qc as run_camera_qc
from ibllib.qc.dlc import DlcQC
from ibllib.plots.figures import dlc_qc_plot, BehaviourPlots, LfpPlots, BadChannelsAp
from ibllib.plots.snapshot import ReportSnapshot
from brainbox.behavior.dlc import likelihood_threshold, get_licks, get_pupil_diameter, get_smooth_pupil_diameter

_logger = logging.getLogger('ibllib')
warnings.warn('`pipes.ephys_preprocessing` to be removed in favour of dynamic pipeline', FutureWarning)


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
    force = False  # whether to force download of missing data on local server if outputs already exist
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


class EphysVideoCompress(tasks.Task):
    priority = 90
    level = 0
    force = False
    job_size = 'large'
    io_charge = 100

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
        labels = {label_from_path(x) for x in output_files}

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
                        ('_iblrig_*Camera.timestamps.ssv', 'raw_video_data', False),
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
        'output_files': [('*trials.table.pqt', 'alf', True),
                         ('*trials.goCueTrigger_times.npy', 'alf', True),
                         ('*trials.intervals_bpod.npy', 'alf', True),
                         ('*trials.stimOff_times.npy', 'alf', True),
                         ('*trials.quiescencePeriod.npy', 'alf', True),
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

    def extract_behaviour(self, save=True):
        dsets, out_files, self.extractor = ephys_fpga.extract_all(
            self.session_path, save=save, return_extractor=True)

        return dsets, out_files

    def run_qc(self, trials_data=None, update=True, plot_qc=False):
        if trials_data is None:
            trials_data, _ = self.extract_behaviour(save=False)
        if not trials_data:
            raise ValueError('No trials data found')

        # Run the task QC
        qc = TaskQC(self.session_path, one=self.one, log=_logger)
        qc.extractor = TaskQCExtractor(self.session_path, lazy=True, one=qc.one)
        # Extract extra datasets required for QC
        qc.extractor.data = qc.extractor.rename_data(trials_data)
        wheel_ts_bpod = self.extractor.bpod2fpga(self.extractor.bpod_trials['wheel_timestamps'])
        qc.extractor.data['wheel_timestamps_bpod'] = wheel_ts_bpod
        qc.extractor.data['wheel_position_bpod'] = self.extractor.bpod_trials['wheel_position']
        qc.extractor.wheel_encoding = 'X4'
        qc.extractor.settings = self.extractor.settings
        qc.extractor.frame_ttls = self.extractor.frame2ttl
        qc.extractor.audio_ttls = self.extractor.audio
        qc.extractor.bpod_ttls = self.extractor.bpod

        # Aggregate and update Alyx QC fields
        qc.run(update=update)

        if plot_qc:
            _logger.info("Creating Trials QC plots")
            try:
                session_id = self.one.path2eid(self.session_path)
                plot_task = BehaviourPlots(session_id, self.session_path, one=self.one)
                _ = plot_task.run()
                self.plot_tasks.append(plot_task)
            except Exception:
                _logger.error('Could not create Trials QC Plot')
                _logger.error(traceback.format_exc())
                self.status = -1
        return qc

    def _run(self, plot_qc=True):
        dsets, out_files = self.extract_behaviour()

        if self.one and not self.one.offline:
            self._behaviour_criterion()
            self.run_qc(trials_data=dsets, update=True, plot_qc=plot_qc)
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


class LaserTrialsLegacy(EphysTrials):
    """This is the legacy extractor for Guido's ephys optogenetic stimulation protocol.

    This is legacy because personal project extractors should be in a separate repository.
    """
    def extract_behaviour(self):
        dsets, out_files = super().extract_behaviour()

        # Re-extract the laser datasets as the above default extractor discards them
        from ibllib.io.extractors import opto_trials
        laser = opto_trials.LaserBool(self.session_path)
        dsets_laser, out_files_laser = laser.extract(save=True)
        dsets.update({k: v for k, v in zip(laser.var_names, dsets_laser)})
        out_files.extend(out_files_laser)
        return dsets, out_files


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
    """
    This task relies on a correctly installed dlc environment as per
    https://docs.google.com/document/d/1g0scP6_3EmaXCU4SsDNZWwDTaD9MG0es_grLA-d0gh0/edit#

    If your environment is set up otherwise, make sure that you set the respective attributes:
    t = EphysDLC(session_path)
    t.dlcenv = Path('/path/to/your/dlcenv/bin/activate')
    t.scripts = Path('/path/to/your/iblscripts/deploy/serverpc/dlc')
    """
    gpu = 1
    cpu = 4
    io_charge = 100
    level = 2
    force = True
    job_size = 'large'

    dlcenv = Path.home().joinpath('Documents', 'PYTHON', 'envs', 'dlcenv', 'bin', 'activate')
    scripts = Path.home().joinpath('Documents', 'PYTHON', 'iblscripts', 'deploy', 'serverpc', 'dlc')
    signature = {
        'input_files': [
            ('_iblrig_leftCamera.raw.mp4', 'raw_video_data', True),
            ('_iblrig_rightCamera.raw.mp4', 'raw_video_data', True),
            ('_iblrig_bodyCamera.raw.mp4', 'raw_video_data', True),
        ],
        'output_files': [
            ('_ibl_leftCamera.dlc.pqt', 'alf', True),
            ('_ibl_rightCamera.dlc.pqt', 'alf', True),
            ('_ibl_bodyCamera.dlc.pqt', 'alf', True),
            ('leftCamera.ROIMotionEnergy.npy', 'alf', True),
            ('rightCamera.ROIMotionEnergy.npy', 'alf', True),
            ('bodyCamera.ROIMotionEnergy.npy', 'alf', True),
            ('leftROIMotionEnergy.position.npy', 'alf', True),
            ('rightROIMotionEnergy.position.npy', 'alf', True),
            ('bodyROIMotionEnergy.position.npy', 'alf', True),
        ],
    }

    def _check_dlcenv(self):
        """Check that scripts are present, dlcenv can be activated and get iblvideo version"""
        assert len(list(self.scripts.rglob('run_dlc.*'))) == 2, \
            f'Scripts run_dlc.sh and run_dlc.py do not exist in {self.scripts}'
        assert len(list(self.scripts.rglob('run_motion.*'))) == 2, \
            f'Scripts run_motion.sh and run_motion.py do not exist in {self.scripts}'
        assert self.dlcenv.exists(), f"DLC environment does not exist in assumed location {self.dlcenv}"
        command2run = f"source {self.dlcenv}; python -c 'import iblvideo; print(iblvideo.__version__)'"
        process = subprocess.Popen(
            command2run,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            executable="/bin/bash"
        )
        info, error = process.communicate()
        if process.returncode != 0:
            raise AssertionError(f"DLC environment check failed\n{error.decode('utf-8')}")
        version = info.decode("utf-8").strip().split('\n')[-1]
        return version

    @staticmethod
    def _video_intact(file_mp4):
        """Checks that the downloaded video can be opened and is not empty"""
        cap = cv2.VideoCapture(str(file_mp4))
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        intact = True if frame_count > 0 else False
        cap.release()
        return intact

    def _run(self, cams=None, overwrite=False):
        # Default to all three cams
        cams = cams or ['left', 'right', 'body']
        cams = assert_valid_label(cams)
        # Set up
        self.session_id = self.one.path2eid(self.session_path)
        actual_outputs = []

        # Loop through cams
        for cam in cams:
            # Catch exceptions so that following cameras can still run
            try:
                # If all results exist and overwrite is False, skip computation
                expected_outputs_present, expected_outputs = self.assert_expected(self.output_files, silent=True)
                if overwrite is False and expected_outputs_present is True:
                    actual_outputs.extend(expected_outputs)
                    return actual_outputs
                else:
                    file_mp4 = next(self.session_path.joinpath('raw_video_data').glob(f'_iblrig_{cam}Camera.raw*.mp4'))
                    if not file_mp4.exists():
                        # In this case we set the status to Incomplete.
                        _logger.error(f"No raw video file available for {cam}, skipping.")
                        self.status = -3
                        continue
                    if not self._video_intact(file_mp4):
                        _logger.error(f"Corrupt raw video file {file_mp4}")
                        self.status = -1
                        continue
                    # Check that dlc environment is ok, shell scripts exists, and get iblvideo version, GPU addressable
                    self.version = self._check_dlcenv()
                    _logger.info(f'iblvideo version {self.version}')
                    check_nvidia_driver()

                    _logger.info(f'Running DLC on {cam}Camera.')
                    command2run = f"{self.scripts.joinpath('run_dlc.sh')} {str(self.dlcenv)} {file_mp4} {overwrite}"
                    _logger.info(command2run)
                    process = subprocess.Popen(
                        command2run,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        executable="/bin/bash",
                    )
                    info, error = process.communicate()
                    # info_str = info.decode("utf-8").strip()
                    # _logger.info(info_str)
                    if process.returncode != 0:
                        error_str = error.decode("utf-8").strip()
                        _logger.error(f'DLC failed for {cam}Camera.\n\n'
                                      f'++++++++ Output of subprocess for debugging ++++++++\n\n'
                                      f'{error_str}\n'
                                      f'++++++++++++++++++++++++++++++++++++++++++++\n')
                        self.status = -1
                        # We dont' run motion energy, or add any files if dlc failed to run
                        continue
                    dlc_result = next(self.session_path.joinpath('alf').glob(f'_ibl_{cam}Camera.dlc*.pqt'))
                    actual_outputs.append(dlc_result)

                    _logger.info(f'Computing motion energy for {cam}Camera')
                    command2run = f"{self.scripts.joinpath('run_motion.sh')} {str(self.dlcenv)} {file_mp4} {dlc_result}"
                    _logger.info(command2run)
                    process = subprocess.Popen(
                        command2run,
                        shell=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        executable="/bin/bash",
                    )
                    info, error = process.communicate()
                    # info_str = info.decode("utf-8").strip()
                    # _logger.info(info_str)
                    if process.returncode != 0:
                        error_str = error.decode("utf-8").strip()
                        _logger.error(f'Motion energy failed for {cam}Camera.\n\n'
                                      f'++++++++ Output of subprocess for debugging ++++++++\n\n'
                                      f'{error_str}\n'
                                      f'++++++++++++++++++++++++++++++++++++++++++++\n')
                        self.status = -1
                        continue
                    actual_outputs.append(next(self.session_path.joinpath('alf').glob(
                        f'{cam}Camera.ROIMotionEnergy*.npy')))
                    actual_outputs.append(next(self.session_path.joinpath('alf').glob(
                        f'{cam}ROIMotionEnergy.position*.npy')))
            except BaseException:
                _logger.error(traceback.format_exc())
                self.status = -1
                continue
        # If status is Incomplete, check that there is at least one output.
        # # Otherwise make sure it gets set to Empty (outputs = None), and set status to -1 to make sure it doesn't slip
        if self.status == -3 and len(actual_outputs) == 0:
            actual_outputs = None
            self.status = -1
        return actual_outputs


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
                                 ('_ibl_bodyCamera.times.npy', 'alf', True),
                                 # the following are required for the DLC plot only
                                 # they are not strictly required, some plots just might be skipped
                                 # In particular the raw videos don't need to be downloaded as they can be streamed
                                 ('_iblrig_bodyCamera.raw.mp4', 'raw_video_data', True),
                                 ('_iblrig_leftCamera.raw.mp4', 'raw_video_data', True),
                                 ('_iblrig_rightCamera.raw.mp4', 'raw_video_data', True),
                                 ('rightROIMotionEnergy.position.npy', 'alf', False),
                                 ('leftROIMotionEnergy.position.npy', 'alf', False),
                                 ('bodyROIMotionEnergy.position.npy', 'alf', False),
                                 ('_ibl_trials.table.pqt', 'alf', True),
                                 ('_ibl_wheel.position.npy', 'alf', True),
                                 ('_ibl_wheel.timestamps.npy', 'alf', True),
                                 ],
                 # More files are required for all panels of the DLC QC plot to function
                 'output_files': [('_ibl_leftCamera.features.pqt', 'alf', True),
                                  ('_ibl_rightCamera.features.pqt', 'alf', True),
                                  ('licks.times.npy', 'alf', True),
                                  # ('dlc_qc_plot.png', 'snapshot', False)
                                  ]
                 }

    def _run(self, overwrite=True, run_qc=True, plot_qc=True):
        """
        Run the EphysPostDLC task. Returns a list of file locations for the output files in signature. The created plot
        (dlc_qc_plot.png) is not returned, but saved in session_path/snapshots and uploaded to Alyx as a note.

        :param overwrite: bool, whether to recompute existing output files (default is False).
                          Note that the dlc_qc_plot will be (re-)computed even if overwrite = False
        :param run_qc: bool, whether to run the DLC QC (default is True)
        :param plot_qc: book, whether to create the dlc_qc_plot (default is True)

        """
        # Check if output files exist locally
        exist, output_files = self.assert_expected(self.signature['output_files'], silent=True)
        if exist and not overwrite:
            _logger.warning('EphysPostDLC outputs exist and overwrite=False, skipping computations of outputs.')
        else:
            if exist and overwrite:
                _logger.warning('EphysPostDLC outputs exist and overwrite=True, overwriting existing outputs.')
            # Find all available dlc files
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
                        if dlc_t.shape[0] == 0:
                            _logger.error(f'camera.times empty for {cam} camera. '
                                          f'Computations using camera.times will be skipped')
                            self.status = -1
                            times = False
                        elif dlc_t.shape[0] < len(dlc_thresh):
                            _logger.error(f'Camera times shorter than DLC traces for {cam} camera. '
                                          f'Computations using camera.times will be skipped')
                            self.status = -1
                            times = 'short'
                    except StopIteration:
                        self.status = -1
                        times = False
                        _logger.error(f'No camera.times for {cam} camera. '
                                      f'Computations using camera.times will be skipped')
                    # These features are only computed from left and right cam
                    if cam in ('left', 'right'):
                        features = pd.DataFrame()
                        # If camera times are available, get the lick time stamps for combined array
                        if times is True:
                            _logger.info(f"Computing lick times for {cam} camera.")
                            combined_licks.append(get_licks(dlc_thresh, dlc_t))
                        elif times is False:
                            _logger.warning(f"Skipping lick times for {cam} camera as no camera.times available")
                        elif times == 'short':
                            _logger.warning(f"Skipping lick times for {cam} camera as camera.times are too short")
                        # Compute pupil diameter, raw and smoothed
                        _logger.info(f"Computing raw pupil diameter for {cam} camera.")
                        features['pupilDiameter_raw'] = get_pupil_diameter(dlc_thresh)
                        try:
                            _logger.info(f"Computing smooth pupil diameter for {cam} camera.")
                            features['pupilDiameter_smooth'] = get_smooth_pupil_diameter(features['pupilDiameter_raw'],
                                                                                         cam)
                        except BaseException:
                            _logger.error(f"Computing smooth pupil diameter for {cam} camera failed, saving all NaNs.")
                            _logger.error(traceback.format_exc())
                            features['pupilDiameter_smooth'] = np.nan
                        # Safe to pqt
                        features_file = Path(self.session_path).joinpath('alf', f'_ibl_{cam}Camera.features.pqt')
                        features.to_parquet(features_file)
                        output_files.append(features_file)

                    # For all cams, compute DLC qc if times available
                    if run_qc is True and times in [True, 'short']:
                        # Setting download_data to False because at this point the data should be there
                        qc = DlcQC(self.session_path, side=cam, one=self.one, download_data=False)
                        qc.run(update=True)
                    else:
                        if times is False:
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
                fig = dlc_qc_plot(self.session_path, one=self.one)
                fig.savefig(fig_path)
                fig.clf()
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
        tasks['ExperimentDescriptionRegisterRaw'] = base_tasks.ExperimentDescriptionRegisterRaw(self.session_path)
        tasks["EphysRegisterRaw"] = tpp.TrainingRegisterRaw(self.session_path)
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
        tasks['EphysTrainingStatus'] = tpp.TrainingStatus(self.session_path, parents=[tasks["EphysTrials"]])
        # level 3
        tasks["EphysPostDLC"] = EphysPostDLC(self.session_path, parents=[tasks["EphysDLC"], tasks["EphysTrials"],
                                                                         tasks["EphysVideoSyncQc"]])
        self.tasks = tasks

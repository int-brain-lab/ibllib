import logging
import traceback
from pathlib import Path
import subprocess
import re
import packaging
import shutil
import numpy as np
import pandas as pd

import spikeglx
import neuropixel
from neurodsp.utils import rms

import one.alf.io as alfio

from ibllib.misc import check_nvidia_driver
from ibllib.pipes import base_tasks
from ibllib.pipes.sync_tasks import SyncPulses
from ibllib.ephys import ephysqc, spikes
from ibllib.qc.alignment_qc import get_aligned_channels
from ibllib.plots.figures import LfpPlots, ApPlots, BadChannelsAp
from ibllib.plots.figures import SpikeSorting as SpikeSortingPlots
from ibllib.io.extractors.ephys_fpga import extract_sync
from ibllib.ephys.spikes import sync_probes


_logger = logging.getLogger("ibllib")


class EphysRegisterRaw(base_tasks.DynamicTask):
    """
    Creates the probe insertions and uploads the probe descriptions file, also compresses the nidq files and uploads
    """

    priority = 100
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [],
            'output_files': [('probes.description.json', 'alf', True)]
        }
        return signature

    def _run(self):

        out_files = spikes.probes_description(self.session_path, self.one)

        return out_files


class EphysSyncRegisterRaw(base_tasks.DynamicTask):
    """
    Task to rename, compress and register raw daq data with .bin format collected using NIDAQ
    """

    priority = 90
    cpu = 2
    io_charge = 30  # this jobs reads raw ap files
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('*.*bin', self.sync_collection, True),
                            ('*.meta', self.sync_collection, True),
                            ('*.wiring.json', self.sync_collection, True)],
            'output_files': [('*nidq.cbin', self.sync_collection, True),
                             ('*nidq.ch', self.sync_collection, True),
                             ('*nidq.meta', self.sync_collection, True),
                             ('*nidq.wiring.json', self.sync_collection, True)]
        }
        return signature

    def _run(self):

        out_files = []

        # Detect the wiring file
        wiring_file = next(self.session_path.joinpath(self.sync_collection).glob('*.wiring.json'), None)
        if wiring_file is not None:
            out_files.append(wiring_file)

        # Search for .bin files in the sync_collection folder
        files = list(self.session_path.joinpath(self.sync_collection).glob('*nidq.*bin'))
        bin_file = files[0] if len(files) == 1 else None

        # If we don't have a .bin/ .cbin file anymore see if we can still find the .ch and .meta files
        if bin_file is None:
            for ext in ['ch', 'meta']:
                files = list(self.session_path.joinpath(self.sync_collection).glob(f'*nidq.{ext}'))
                ext_file = files[0] if len(files) == 1 else None
                if ext_file is not None:
                    out_files.append(ext_file)

            return out_files if len(out_files) > 0 else None

        # If we do find the .bin file, compress files (only if they haven't already been compressed)
        sr = spikeglx.Reader(bin_file)
        if sr.is_mtscomp:
            sr.close()
            cbin_file = bin_file
            assert cbin_file.suffix == '.cbin'
        else:
            cbin_file = sr.compress_file()
            sr.close()
            bin_file.unlink()

        meta_file = cbin_file.with_suffix('.meta')
        ch_file = cbin_file.with_suffix('.ch')

        out_files.append(cbin_file)
        out_files.append(ch_file)
        out_files.append(meta_file)

        return out_files


class EphysCompressNP1(base_tasks.EphysTask):
    priority = 90
    cpu = 2
    io_charge = 100  # this jobs reads raw ap files
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*ap.*bin', f'{self.device_collection}/{self.pname}', True),
                            ('*lf.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*lf.*bin', f'{self.device_collection}/{self.pname}', True),
                            ('*wiring.json', f'{self.device_collection}/{self.pname}', False)],
            'output_files': [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*ap.cbin', f'{self.device_collection}/{self.pname}', True),
                             ('*ap.ch', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.cbin', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.ch', f'{self.device_collection}/{self.pname}', True),
                             ('*wiring.json', f'{self.device_collection}/{self.pname}', False)]
        }
        return signature

    def _run(self):

        out_files = []

        # Detect and upload the wiring file
        wiring_file = next(self.session_path.joinpath(self.device_collection, self.pname).glob('*wiring.json'), None)
        if wiring_file is not None:
            out_files.append(wiring_file)

        ephys_files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname))
        ephys_files += spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname), ext="ch")
        ephys_files += spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname), ext="meta")

        for ef in ephys_files:
            for typ in ["ap", "lf"]:
                bin_file = ef.get(typ)
                if not bin_file:
                    continue
                if bin_file.suffix.find("bin") == 1:
                    with spikeglx.Reader(bin_file) as sr:
                        if sr.is_mtscomp:
                            out_files.append(bin_file)
                        else:
                            _logger.info(f"Compressing binary file {bin_file}")
                            cbin_file = sr.compress_file()
                            sr.close()
                            bin_file.unlink()
                            out_files.append(cbin_file)
                            out_files.append(cbin_file.with_suffix('.ch'))
                else:
                    out_files.append(bin_file)

        return out_files


class EphysCompressNP21(base_tasks.EphysTask):
    priority = 90
    cpu = 2
    io_charge = 100  # this jobs reads raw ap files
    job_size = 'large'

    @property
    def signature(self):
        signature = {
            'input_files': [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*ap.*bin', f'{self.device_collection}/{self.pname}', True),
                            ('*wiring.json', f'{self.device_collection}/{self.pname}', False)],
            'output_files': [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*ap.cbin', f'{self.device_collection}/{self.pname}', True),
                             ('*ap.ch', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.meta', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.cbin', f'{self.device_collection}/{self.pname}', True),
                             ('*lf.ch', f'{self.device_collection}/{self.pname}', True),
                             ('*wiring.json', f'{self.device_collection}/{self.pname}', False)]
        }
        return signature

    def _run(self):

        out_files = []
        # Detect wiring files
        wiring_file = next(self.session_path.joinpath(self.device_collection, self.pname).glob('*wiring.json'), None)
        if wiring_file is not None:
            out_files.append(wiring_file)

        ephys_files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname))
        if len(ephys_files) > 0:
            bin_file = ephys_files[0].get('ap', None)

        # This is the case where no ap.bin/.cbin file exists
        if len(ephys_files) == 0 or not bin_file:
            ephys_files += spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname), ext="ch")
            ephys_files += spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname), ext="meta")
            for ef in ephys_files:
                for typ in ["ap", "lf"]:
                    bin_file = ef.get(typ)
                    if bin_file:
                        out_files.append(bin_file)

            return out_files

        # If the ap.bin / .cbin file does exists instantiate the NP2converter
        np_conv = neuropixel.NP2Converter(bin_file, compress=True)
        np_conv_status = np_conv.process()
        np_conv_files = np_conv.get_processed_files_NP21()
        np_conv.sr.close()

        # Status = 1 - successfully complete
        if np_conv_status == 1:  # This is the status that it has completed successfully
            out_files += np_conv_files
            return out_files
        # Status = 0 - Already processed
        elif np_conv_status == 0:
            ephys_files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname))
            ephys_files += spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname), ext="ch")
            ephys_files += spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname), ext="meta")
            for ef in ephys_files:
                for typ in ["ap", "lf"]:
                    bin_file = ef.get(typ)
                    if bin_file and bin_file.suffix != '.bin':
                        out_files.append(bin_file)

            return out_files

        else:
            return


class EphysCompressNP24(base_tasks.EphysTask):
    """
    Compresses NP2.4 data by splitting into N binary files, corresponding to N shanks
    :param pname: a probe name string
    :param device_collection: the collection containing the probes (usually 'raw_ephys_data')
    :param nshanks: number of shanks used (usually 4 but it may be less depending on electrode map), optional
    """

    priority = 90
    cpu = 2
    io_charge = 100  # this jobs reads raw ap files
    job_size = 'large'

    def __init__(self, session_path, *args, pname=None, device_collection='raw_ephys_data', nshanks=None, **kwargs):
        assert pname, "pname is a required argument"
        if nshanks is None:
            meta_file = next(session_path.joinpath(device_collection, pname).glob('*ap.meta'))
            nshanks = spikeglx._get_nshanks_from_meta(spikeglx.read_meta_data(meta_file))
        assert nshanks > 1
        super(EphysCompressNP24, self).__init__(
            session_path, *args, pname=pname, device_collection=device_collection, nshanks=nshanks, **kwargs)

    @property
    def signature(self):

        signature = {
            'input_files': [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*ap.*bin', f'{self.device_collection}/{self.pname}', True),
                            ('*wiring.json', f'{self.device_collection}/{self.pname}', False)],
            'output_files': [('*ap.meta', f'{self.device_collection}/{self.pname}{pext}', True) for pext in self.pextra] +
                            [('*ap.cbin', f'{self.device_collection}/{self.pname}{pext}', True) for pext in self.pextra] +
                            [('*ap.ch', f'{self.device_collection}/{self.pname}{pext}', True) for pext in self.pextra] +
                            [('*lf.meta', f'{self.device_collection}/{self.pname}{pext}', True) for pext in self.pextra] +
                            [('*lf.cbin', f'{self.device_collection}/{self.pname}{pext}', True) for pext in self.pextra] +
                            [('*lf.ch', f'{self.device_collection}/{self.pname}{pext}', True) for pext in self.pextra] +
                            [('*wiring.json', f'{self.device_collection}/{self.pname}{pext}', False) for pext in self.pextra]
        }
        return signature

    def _run(self):

        # Do we need the ability to register the files once it already been processed and original file deleted?

        files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, self.pname))
        assert len(files) == 1
        bin_file = files[0].get('ap', None)

        np_conv = neuropixel.NP2Converter(bin_file, post_check=True, compress=True, delete_original=False)
        # TODO once we happy change delete_original=True
        np_conv_status = np_conv.process()
        out_files = np_conv.get_processed_files_NP24()
        np_conv.sr.close()

        if np_conv_status == 1:
            wiring_file = next(self.session_path.joinpath(self.device_collection, self.pname).glob('*wiring.json'), None)
            if wiring_file is not None:
                # copy wiring file to each sub probe directory and add to output files
                for pext in self.pextra:
                    new_wiring_file = self.session_path.joinpath(self.device_collection, f'{self.pname}{pext}', wiring_file.name)
                    shutil.copyfile(wiring_file, new_wiring_file)
                    out_files.append(new_wiring_file)
            return out_files
        elif np_conv_status == 0:
            for pext in self.pextra:
                ephys_files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, f'{self.pname}{pext}'))
                ephys_files += spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection,
                                                                                    f'{self.pname}{pext}'), ext="ch")
                ephys_files += spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection,
                                                                                    f'{self.pname}{pext}'), ext="meta")
                for ef in ephys_files:
                    for typ in ["ap", "lf"]:
                        bin_file = ef.get(typ)
                        if bin_file and bin_file.suffix != '.bin':
                            out_files.append(bin_file)

                wiring_file = next(self.session_path.joinpath(self.device_collection,
                                                              f'{self.pname}{pext}').glob('*wiring.json'), None)
                if wiring_file is None:
                    # See if we have the original wiring file
                    orig_wiring_file = next(self.session_path.joinpath(self.device_collection,
                                                                       self.pname).glob('*wiring.json'), None)
                    if orig_wiring_file is not None:
                        # copy wiring file to sub probe directory and add to output files
                        new_wiring_file = self.session_path.joinpath(self.device_collection, f'{self.pname}{pext}',
                                                                     orig_wiring_file.name)
                        shutil.copyfile(orig_wiring_file, new_wiring_file)
                        out_files.append(new_wiring_file)
                else:
                    out_files.append(wiring_file)

            return out_files
        else:
            return


class EphysSyncPulses(SyncPulses):

    priority = 90
    cpu = 2
    io_charge = 30  # this jobs reads raw ap files
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('*nidq.cbin', self.sync_collection, True),
                            ('*nidq.ch', self.sync_collection, False),
                            ('*nidq.meta', self.sync_collection, True),
                            ('*nidq.wiring.json', self.sync_collection, True)],
            'output_files': [('_spikeglx_sync.times.npy', self.sync_collection, True),
                             ('_spikeglx_sync.polarities.npy', self.sync_collection, True),
                             ('_spikeglx_sync.channels.npy', self.sync_collection, True)]
        }

        return signature


class EphysPulses(base_tasks.EphysTask):
    """
    Extract Pulses from raw electrophysiology data into numpy arrays
    Perform the probes synchronisation with nidq (3B) or main probe (3A)
    First the job extract the sync pulses from the synchronisation task in all probes, and then perform the
     synchronisation with the nidq

    :param pname: a list of probes names or a single probe name string
    :param device_collection: the collection containing the probes (usually 'raw_ephys_data')
    :param sync_collection: the collection containing the synchronisation device - nidq (usually 'raw_ephys_data')
    """
    priority = 90
    cpu = 2
    io_charge = 30  # this jobs reads raw ap files
    job_size = 'small'

    def __init__(self, *args, **kwargs):
        super(EphysPulses, self).__init__(*args, **kwargs)
        assert self.device_collection, "device_collection is a required argument"
        assert self.sync_collection, "sync_collection is a required argument"
        self.pname = [self.pname] if isinstance(self.pname, str) else self.pname
        assert type(self.pname) == list, 'pname task argument should be a list or a string'

    @property
    def signature(self):
        signature = {
            'input_files': [('*ap.meta', f'{self.device_collection}/{pname}', True) for pname in self.pname] +
                           [('*ap.cbin', f'{self.device_collection}/{pname}', True) for pname in self.pname] +
                           [('*ap.ch', f'{self.device_collection}/{pname}', True) for pname in self.pname] +
                           [('*ap.wiring.json', f'{self.device_collection}/{pname}', False) for pname in self.pname] +
                           [('_spikeglx_sync.times.npy', self.sync_collection, True),
                            ('_spikeglx_sync.polarities.npy', self.sync_collection, True),
                            ('_spikeglx_sync.channels.npy', self.sync_collection, True)],
            'output_files': [(f'_spikeglx_sync.times.{pname}.npy', f'{self.device_collection}/{pname}', True)
                             for pname in self.pname] +
                            [(f'_spikeglx_sync.polarities.{pname}.npy', f'{self.device_collection}/{pname}', True)
                             for pname in self.pname] +
                            [(f'_spikeglx_sync.channels.{pname}.npy', f'{self.device_collection}/{pname}', True)
                             for pname in self.pname] +
                            [('*sync.npy', f'{self.device_collection}/{pname}', True) for pname in
                             self.pname] +
                            [('*timestamps.npy', f'{self.device_collection}/{pname}', True) for pname in
                             self.pname]
        }

        return signature

    def _run(self, overwrite=False):

        out_files = []
        for probe in self.pname:
            files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.device_collection, probe))
            assert len(files) == 1  # will error if the session is split
            bin_file = files[0].get('ap', None)
            if not bin_file:
                return []
            _, out = extract_sync(self.session_path, ephys_files=files, overwrite=overwrite)
            out_files += out

        status, sync_files = sync_probes.sync(self.session_path, probe_names=self.pname)

        return out_files + sync_files


class RawEphysQC(base_tasks.EphysTask):

    cpu = 2
    io_charge = 30  # this jobs reads raw ap files
    priority = 10  # a lot of jobs depend on this one
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*lf.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*lf.ch', f'{self.device_collection}/{self.pname}', False),
                            ('*lf.*bin', f'{self.device_collection}/{self.pname}', False)],
            'output_files': [('_iblqc_ephysChannels.apRMS.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysChannels.rawSpikeRates.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysChannels.labels.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysSpectralDensityLF.freqs.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysSpectralDensityLF.power.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysSpectralDensityAP.freqs.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysSpectralDensityAP.power.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysTimeRmsLF.rms.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysTimeRmsLF.timestamps.npy', f'{self.device_collection}/{self.pname}', True)]
        }
        return signature

    # TODO make sure this works with NP2 probes (at the moment not sure it will due to raiseError mapping)
    def _run(self, overwrite=False):

        eid = self.one.path2eid(self.session_path)
        probe = self.one.alyx.rest('insertions', 'list', session=eid, name=self.pname)

        # We expect there to only be one probe
        if len(probe) != 1:
            _logger.warning(f"{self.pname} for {eid} not found")  # Should we create it?
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
    io_charge = 100  # this jobs reads raw ap files
    priority = 60
    job_size = 'large'
    force = True

    SHELL_SCRIPT = Path.home().joinpath(
        "Documents/PYTHON/iblscripts/deploy/serverpc/kilosort2/run_pykilosort.sh"
    )
    SPIKE_SORTER_NAME = 'pykilosort'
    PYKILOSORT_REPO = Path.home().joinpath('Documents/PYTHON/SPIKE_SORTING/pykilosort')

    @property
    def signature(self):
        signature = {
            'input_files': [('*ap.meta', f'{self.device_collection}/{self.pname}', True),
                            ('*ap.cbin', f'{self.device_collection}/{self.pname}', True),
                            ('*ap.ch', f'{self.device_collection}/{self.pname}', True),
                            ('*sync.npy', f'{self.device_collection}/{self.pname}', True)],
            'output_files': [('spike_sorting_pykilosort.log', f'spike_sorters/pykilosort/{self.pname}', True),
                             ('_iblqc_ephysTimeRmsAP.rms.npy', f'{self.device_collection}/{self.pname}', True),
                             ('_iblqc_ephysTimeRmsAP.timestamps.npy', f'{self.device_collection}/{self.pname}', True)]
        }
        return signature

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
        """
        Parse the following line (2 formats depending on version) from the log files to get the version
        '\x1b[0m15:39:37.919 [I] ibl:90               Starting Pykilosort version ibl_1.2.1, output in gnagga^[[0m\n'
        '\x1b[0m15:39:37.919 [I] ibl:90               Starting Pykilosort version ibl_1.3.0^[[0m\n'
        """
        with open(log_file) as fid:
            line = fid.readline()
        version = re.search('version (.*), output', line)
        version = version or re.search('version (.*)', line)  # old versions have output, new have a version line
        version = re.sub('\\^[[0-9]+m', '', version.group(1))  # removes the coloring tags
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
        log_file = temp_dir.joinpath(f"spike_sorting_{self.SPIKE_SORTER_NAME}.log")
        _logger.info(f"job progress command: tail -f {temp_dir} *.log")
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
                    shutil.copyfile(logfile, probe_out_path.joinpath(f"_ibl_log.info_{self.SPIKE_SORTER_NAME}.log"))
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


class EphysCellsQc(base_tasks.EphysTask):
    priority = 90
    job_size = 'small'

    @property
    def signature(self):
        signature = {
            'input_files': [('spikes.times.npy', f'alf/{self.pname}*', True),
                            ('spikes.clusters.npy', f'alf/{self.pname}*', True),
                            ('spikes.amps.npy', f'alf/{self.pname}*', True),
                            ('spikes.depths.npy', f'alf/{self.pname}*', True),
                            ('clusters.channels.npy', f'alf/{self.pname}*', True)],
            'output_files': [('clusters.metrics.pqt', f'alf/{self.pname}*', True)]
        }
        return signature

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
        pdict = self.one.alyx.rest('insertions', 'list', session=eid, name=self.pname, no_cache=True)
        if len(pdict) != 1:
            _logger.warning(f'No probe found for probe name: {self.pname}')
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
        files_spikes = Path(self.session_path).joinpath('alf', self.pname).rglob('spikes.times.npy')
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

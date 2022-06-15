from ibllib.pipes import tasks
from ibllib.io.extractors.ephys_fpga import _sync_to_alf, extract_sync
from ibllib.ephys import sync_probes

import one.alf.io as alfio

import spikeglx


class SyncMtscomp(tasks.Task):
    """
    Must pass in sync_collection using runtime_args when instantiating task
    e.g runtime_args = {'sync_collection': 'raw_widefield_data'}
    """
    signature = {
        'input_files': [('*nidq.bin', 'raw_XX_data', True),
                        ('*nidq.meta', 'raw_XX_data', True),],
        'output_files': [('daq.raw.nidq.cbin', 'raw_XX_data', True),
                         ('daq.raw.nidq.ch', 'raw_XX_data', True),
                         ('daq.raw.nidq.meta', 'raw_XX_data', True)]
    }

    def _run(self):

        out_files = []
        collection = self.runtime_args.get('sync_collection')
        # search for .bin files in the raw_widefield_data folder
        files = spikeglx.glob_ephys_file(self.session_path.joinpath(collection))
        assert len(files) == 1
        bin_file = files[0].get('nidq', None)
        # TODO account for case of nidq not existing
        # Rename files (only if they haven't already been renamed)
        if 'daq.raw.nidq' not in bin_file.stem:
            new_bin_file = bin_file.parent.joinpath('daq.raw.nidq' + bin_file.suffix)
            bin_file.replace(new_bin_file)
            meta_file = bin_file.with_suffix('.meta')
            new_meta_file = meta_file.parent.joinpath('daq.raw.nidq.meta')
            meta_file.replace(new_meta_file)
        else:
            new_bin_file = bin_file
            new_meta_file = bin_file.with_suffix('.meta')

        sr = spikeglx.Reader(new_bin_file)
        # Compress files (only if they haven't already been compressed)
        if not sr.is_mtscomp:
            cbin_file = sr.compress_file(keep_original=False)
            ch_file = cbin_file.with_suffix('.ch')
        else:
            cbin_file = new_bin_file
            ch_file = cbin_file.with_suffix('.ch')
            assert cbin_file.suffix == '.cbin'

        out_files.append(cbin_file)
        out_files.append(ch_file)
        out_files.append(new_meta_file)

        return out_files

    def get_signatures(self, **kwargs):

        # Add the correct collection to the task signatures
        collection = self.runtime_args.get('sync_collection')

        full_input_files = []
        for sig in self.signature['input_files']:
            full_input_files.append((sig[0], collection, sig[2]))

        full_output_files = []
        for sig in self.signature['output_files']:
            full_output_files.append((sig[0], collection, sig[2]))

        self.input_files = full_input_files
        self.output_files = full_output_files


class SyncPulses(tasks.Task):
    """
    Must pass in sync_collection using runtime_args when instantiating task
    e.g runtime_args = {'sync_collection': 'raw_widefield_data'}
    """

    signature = {
        'input_files': [('daq.raw.nidq.meta', 'raw_XX_data', True), # TODO figure out the naming convention we will use
                        ('daq.raw.nidq.cbin', 'raw_XX_data', True),
                        ('daq.raw.nidq.ch', 'raw_XX_data', True)],
        'output_files': [('_spikeglx_sync.channels.npy', 'raw_XX_data', True),
                         ('_spikeglx_sync.polarities.npy', 'raw_XX_data', True),
                         ('_spikeglx_sync.polarities.times', 'raw_XX_data', True)]
    }

    def _run(self, overwrite=False):

        # TODO this is replicating a lot of ephys_fpga.extract_sync refactor to make generalisable along with Dynamic pipeline
        syncs = []
        outputs = []

        collection = self.runtime_args.get('collection')
        files = spikeglx.glob_ephys_files(self.session_path.joinpath(collection))
        assert len(files) == 1
        bin_file = files[0].get('nidq', None)
        if not bin_file:
            return []

        alfname = dict(object='sync', namespace='spikeglx')
        file_exists = alfio.exists(bin_file.parent, **alfname)

        if not overwrite and file_exists:
            _logger.warning(f'Skipping raw sync: SGLX sync found for folder {files[0].label}!')
            sync = alfio.load_object(bin_file.parent, **alfname)
            out_files, _ = alfio._ls(bin_file.parent, **alfname)
        else:
            sr = spikeglx.Reader(bin_file)
            sync, out_files = _sync_to_alf(sr, bin_file.parent, save=True)

        outputs.extend(out_files)
        syncs.extend([sync])

        return outputs, syncs

    def get_signatures(self, **kwargs):

        # Add the correct collection to the task signatures
        collection = self.runtime_args.get('collection')

        full_input_files = []
        for sig in self.signature['input_files']:
            full_input_files.append((sig[0], collection, sig[2]))

        full_output_files = []
        for sig in self.signature['output_files']:
            full_output_files.append((sig[0], collection, sig[2]))

        self.input_files = full_input_files
        self.output_files = full_output_files


# TODO for now ephyspulses it's own task but see how we can generalise / inherit from base sync class
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

    def _run(self, overwrite=False):
        # outputs numpy
        syncs, out_files = extract_sync(self.session_path, overwrite=overwrite)
        for out_file in out_files:
            _logger.info(f"extracted pulses for {out_file}")

        status, sync_files = sync_probes.sync(self.session_path)
        return out_files + sync_files


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

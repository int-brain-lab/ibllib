from ibllib.pipes import tasks
from ibllib.io.extractors.ephys_fpga import _sync_to_alf
import logging

import one.alf.io as alfio

import spikeglx

_logger = logging.getLogger('ibllib')


class SyncRegisterRaw(tasks.RegisterTask):
    """
    Task to register raw daq data
    """
    signature = {
        'input_files': [],
        'output_files': []
    }

    def dynamic_signatures(self):
        input_signature = []
        output_signature = [(f'daq.raw.{self.sync_ext}', self.sync_collection, True),
                            (f'daq.raw.wiring.json', self.sync_collection, True)]

        return input_signature, output_signature

    def setUp(self):
        self.signature['input_files'], self.signature['output_files'] = self.dynamic_signatures()

        return super().setUp()


class SyncMtscomp(tasks.Task):
    """
    Task to rename, compress and register raw daq data with .bin format collected using NIDAQ
    """
    signature = {
        'input_files': [],
        'output_files': []
    }

    def dynamic_signatures(self):
        input_signature = [('*.bin', self.sync_collection, True),
                           ('*.meta', self.sync_collection, True),
                           ('*.wiring.json', self.sync_collection, True),]
        output_signature = [('daq.raw.cbin', self.sync_collection, True),
                            ('daq.raw.ch', self.sync_collection, True),
                            ('daq.raw.meta', self.sync_collection, True),
                            ('daq.raw.wiring.json', self.sync_collection, True)]

        return input_signature, output_signature

    def setup(self):
        """
        Overwrite setup method to allow inputs and outputs to be only one probe
        :param probes: list of probes e.g ['probe00']
        :return:
        """

        self.signature['input_files'], self.signature['output_files'] = self.dynamic_signatures()

        return super().setUp()

    def _run(self):

        out_files = []
        # search for .bin files in the sync_collection folder
        files = spikeglx.glob_ephys_file(self.session_path.joinpath(self.sync_collection))
        assert len(files) == 1
        bin_file = files[0].get('nidq', None)
        if bin_file is None:
            return

        # Rename files (only if they haven't already been renamed)
        if 'daq.raw' not in bin_file.stem:
            new_bin_file = bin_file.parent.joinpath('daq.raw' + bin_file.suffix)
            bin_file.replace(new_bin_file)
            meta_file = bin_file.with_suffix('.meta')
            new_meta_file = meta_file.parent.joinpath('daq.raw.meta')
            meta_file.replace(new_meta_file)
        else:
            new_bin_file = bin_file
            new_meta_file = bin_file.with_suffix('.meta')

        # Compress files (only if they haven't already been compressed)
        sr = spikeglx.Reader(new_bin_file)
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

        # Rename the wiring file (if it hasn't already been renamed)
        wiring_file = next(self.session_path.joinpath(self.sync_collection).glob('*wiring.json'), None)
        if wiring_file is not None:
            if 'daq.raw.wiring' not in wiring_file.stem:
                new_wiring_file = wiring_file.parent.joinpath('daq.raw.wiring.json')
                wiring_file.replace(new_wiring_file)
            else:
                new_wiring_file = wiring_file

            out_files.append(new_wiring_file)

        return out_files


class SyncPulses(tasks.Task):
    """
    Extract sync pulses from NIDAQ .bin / .cbin file
    N.B Only extracts sync from sync collection (i.e not equivalent to EphysPulses that extracts sync pulses for each probe)

    # TODO generalise to other daq and file formats, generalise to 3A probes
    """

    signature = {
        'input_files': [],
        'output_files': []
    }

    def dynamic_signatures(self):
        input_signature = [('daq.raw.nidq.cbin', self.sync_collection, True),
                           ('daq.raw.nidq.ch', self.sync_collection, True),
                           ('daq.raw.nidq.meta', self.sync_collection, True),
                           ('daq.raw.wiring.json', self.sync_collection, True)]

        output_signature = [('_spikeglx_sync.times.npy', self.sync_collection, True),
                            ('_spikeglx_sync.polarities.npy', self.sync_collection, True),
                            ('_spikeglx_sync.channels.npy', self.sync_collection, True)]

        return input_signature, output_signature


    def _run(self, overwrite=False):

        # TODO this is replicating a lot of ephys_fpga.extract_sync refactor to make generalisable along with Dynamic pipeline
        # Need to make this daq agnostic
        syncs = []
        outputs = []

        files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.sync_collection))
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

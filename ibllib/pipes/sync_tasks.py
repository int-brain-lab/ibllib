from ibllib.pipes import base_tasks
from ibllib.io.extractors.ephys_fpga import _sync_to_alf
import logging

import one.alf.io as alfio

import spikeglx

_logger = logging.getLogger('ibllib')


class SyncRegisterRaw(base_tasks.RegisterTaskData):
    """
    Task to register raw daq data
    """

    def dynamic_signatures(self):
        input_signature = []
        output_signature = [(f'daq.raw.{self.sync}.{self.sync_ext}', self.sync_collection, True),
                            (f'daq.raw.{self.sync}.wiring.json', self.sync_collection, True)]

        return input_signature, output_signature


class SyncMtscomp(base_tasks.DynamicTask):
    """
    Task to rename, compress and register raw daq data with .bin format collected using NIDAQ
    """

    def dynamic_signatures(self):
        input_signature = [('*.*bin', self.sync_collection, True),
                           ('*.meta', self.sync_collection, True),
                           ('*.wiring.json', self.sync_collection, True)]
        output_signature = [(f'daq.raw.{self.sync}.cbin', self.sync_collection, True),
                            (f'daq.raw.{self.sync}.ch', self.sync_collection, True),
                            (f'daq.raw.{self.sync}.meta', self.sync_collection, True),
                            (f'daq.raw.{self.sync}.wiring.json', self.sync_collection, True)]

        return input_signature, output_signature

    def _run(self):

        out_files = []
        # search for .bin files in the sync_collection folder
        files = spikeglx.glob_ephys_files(self.session_path.joinpath(self.sync_collection))
        assert len(files) == 1
        bin_file = files[0].get('nidq', None)

        if bin_file is None:
            return

        # Compress files (only if they haven't already been compressed)
        sr = spikeglx.Reader(bin_file)
        if sr.is_mtscomp:
            sr.close()
            cbin_file = bin_file
            assert cbin_file.suffix == '.cbin'
        else:
            cbin_file = sr.compress_file()
            sr.close()
            bin_file.unlink()

        # Rename files (only if they haven't already been renamed)
        if 'daq.raw' not in cbin_file.stem:
            new_bin_file = cbin_file.parent.joinpath(f'daq.raw.{self.sync}' + cbin_file.suffix)
            cbin_file.replace(new_bin_file)

            meta_file = cbin_file.with_suffix('.meta')
            new_meta_file = new_bin_file.with_suffix('.meta')
            meta_file.replace(new_meta_file)

            ch_file = cbin_file.with_suffix('.ch')
            new_ch_file = new_bin_file.with_suffix('.ch')
            ch_file.replace(new_ch_file)
        else:
            new_bin_file = cbin_file
            new_meta_file = cbin_file.with_suffix('.meta')
            new_ch_file = cbin_file.with_suffix('.ch')

        out_files.append(new_bin_file)
        out_files.append(new_ch_file)
        out_files.append(new_meta_file)

        # Rename the wiring file (if it hasn't already been renamed)
        wiring_file = next(self.session_path.joinpath(self.sync_collection).glob('*.wiring.json'), None)
        if wiring_file is not None:
            if 'daq.raw' not in wiring_file.stem:
                new_wiring_file = wiring_file.parent.joinpath(f'daq.raw.{self.sync}.wiring.json')
                wiring_file.replace(new_wiring_file)
            else:
                new_wiring_file = wiring_file

            out_files.append(new_wiring_file)

        return out_files


class SyncPulses(base_tasks.DynamicTask):
    """
    Extract sync pulses from NIDAQ .bin / .cbin file
    N.B Only extracts sync from sync collection (i.e not equivalent to EphysPulses that extracts sync pulses for each probe)

    # TODO generalise to other daq and file formats, generalise to 3A probes
    """

    def dynamic_signatures(self):
        input_signature = [(f'daq.raw.{self.sync}.*bin', self.sync_collection, True),
                           (f'daq.raw.{self.sync}.ch', self.sync_collection, False),  # not mandatory if we have .bin file
                           (f'daq.raw.{self.sync}.meta', self.sync_collection, True),
                           (f'daq.raw.{self.sync}.wiring.json', self.sync_collection, True)]

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
            sr.close()
            for out_file in out_files:
                _logger.info(f"extracted pulses for {out_file}")

        outputs.extend(out_files)
        syncs.extend([sync])

        return outputs

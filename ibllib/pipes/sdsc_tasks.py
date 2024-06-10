import numpy as np

import spikeglx
from ibllib.ephys.sync_probes import apply_sync
from ibllib.pipes.tasks import Task


class RegisterSpikeSortingSDSC(Task):

    @property
    def signature(self):
        signature = {
            'input_files': [('*sync.npy', f'raw_ephys_data/{self.pname}', False),
                            ('*ap.meta', f'raw_ephys_data/{self.pname}', False)],
            'output_files': []
        }
        return signature

    def __init__(self, session_path, pname=None, revision_label='#test#', **kwargs):
        super().__init__(session_path, **kwargs)

        self.pname = pname
        self.revision_label = revision_label

    def _run(self):

        out_path = self.session_path.joinpath('alf', self.pname, 'pykilosort', self.revision_label)

        def _fs(meta_file):
            # gets sampling rate from data
            md = spikeglx.read_meta_data(meta_file)
            return spikeglx._get_fs_from_meta(md)

        sync_file = next(self.session_path.joinpath('raw_ephys_data', self.pname).glob('*sync.npy'))
        meta_file = next(self.session_path.joinpath('raw_ephys_data', self.pname).glob('*ap.meta'))

        st_file = out_path.joinpath('spikes.times.npy')
        spike_samples = np.load(out_path.joinpath('spikes.samples.npy'))
        interp_times = apply_sync(sync_file, spike_samples / _fs(meta_file), forward=True)
        np.save(st_file, interp_times)

        out = list(self.session_path.joinpath('alf', self.pname, 'pykilosort', self.revision_label).glob('*'))
        return out

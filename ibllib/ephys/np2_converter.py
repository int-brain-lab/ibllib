from ibllib.io import spikeglx
from ibllib.dsp.utils import WindowGenerator
import scipy.signal
import numpy as np
from pathlib import Path
import copy
import logging
_logger = logging.getLogger('ibllib')

class NP2Converter:
    # Get out the AP, Get out the LFP, Split into shanks, write to file (all different functions)
    # make metadata

    def __init__(self, ap_file, post_check=True, delete_original=False):
        self.ap_file = Path(ap_file)
        self.sr = spikeglx.Reader(ap_file)
        self.post_check = post_check
        self.delete_original = delete_original
        self.np_version = spikeglx._get_neuropixel_version_from_meta(self.sr.meta)
        self.init_params()

    def init_params(self, nsamples=None, nwindow=None, nchns=None, extra=None, nshank=None):
        self.fs_ap = 30000
        self.fs_lf = 2500
        self.ratio = int(self.fs_ap / self.fs_lf)
        self.nsamples = nsamples or self.sr.ns
        self.samples_window = nwindow or 2 * self.fs_ap
        assert np.mod(self.samples_window, self.ratio) == 0, \
            f'nwindow must be a factor or {self.ratio}'
        self.samples_overlap = 576
        assert np.mod(self.samples_overlap, self.ratio) == 0, \
            f'samples_overlap must be a factor or {self.ratio}'
        self.samples_taper = int(self.samples_overlap / 4)
        assert np.mod(self.samples_taper, self.ratio) == 0, \
            f'samples_taper must be a factor or {self.ratio}'
        self.taper = np.r_[0, scipy.signal.windows.cosine((self.samples_taper - 1) * 2), 0]

        # Low pass filter (acts as both anti-aliasing and LP filter)
        butter_lp_kwargs = {'N': 2, 'Wn': 1000 / 2500 / 2, 'btype': 'lowpass'}
        self.sos_lp = scipy.signal.butter(**butter_lp_kwargs, output='sos')

        # Number of ap channels
        self.napch = nchns or int(self.sr.meta['snsApLfSy'][0])
        # Position of start of sync channels in the raw data
        self.idxsyncch = int(self.sr.meta['snsApLfSy'][0])

        self.extra = extra or ''
        self.nshank = nshank or None

    def check_meta(self):
        self.processed_meta = spikeglx.get_processed_meta(self.sr.meta)


    def _prepare_files_NP2_4(self, overwrite=False):
        # if already a processed file we can't do anything
        chn_info = spikeglx._map_channels_from_meta(self.sr.meta)
        n_shanks = self.nshank or np.unique(chn_info['shank'])
        label = self.ap_file.parent.parts[-1]
        shank_info = {}
        exists = False

        for sh in n_shanks:
            _shank_info = {}
            # channels for individual shank + sync channel
            _shank_info['chns'] = np.r_[np.where(chn_info['shank'] == sh)[0],
                                        np.array(spikeglx._get_sync_trace_indices_from_meta(
                                            self.sr.meta))]

            probe_path = self.ap_file.parent.parent.joinpath(label + chr(97 + int(sh)) + self.extra)

            if not probe_path.exists() or overwrite:
                probe_path.mkdir(parents=True, exist_ok=True)
                _shank_info['ap_file'] = probe_path.joinpath(self.ap_file.name)
                _shank_info['ap_open_file'] = open(_shank_info['ap_file'], 'wb')
                _shank_info['lf_file'] = probe_path.joinpath(
                    self.ap_file.name.replace('ap', 'lf'))
                _shank_info['lf_open_file'] = open(_shank_info['lf_file'], 'wb')
            else:
                exists = True
                # TODO should we check if it has ap and lf files properly or the folder existsing
                # is enough
                # TODO better warning message
                _logger.warning(f'directory {probe_path} derived from file {self.ap_file} '
                                f'already exists. To overwrite content in {probe_path}, set'
                                f'overwrite = True')
            shank_info[f'shank{sh}'] = _shank_info

        return shank_info, exists

    def _prepare_files_NP2_1(self, save=True, overwrite=False):
        # need to see if lf files exist in the ap_file folder
        # TODO not yet implemented
        lf_file = self.ap_file.parent.joinpath(self.ap_file.name.replace('ap', 'lf'))
        if not lf_file.exists():
            lala=  1


    def process(self, overwrite=False):
        if self.np_version == 'NP2.4':
            self._process_NP2_4(overwrite=overwrite)
        elif self.np_version == 'NP2.1':
            self.info = self._prepare_files_NP2_1(overwrite=overwrite)
        else:
            self.info = None


    def _process_NP2_4(self, overwrite=False):
        #if self.sr.meta.get['original_meta']
        #    _logger.warning('This ap file is an NP2.4 that has already been split into shanks, '
        #                    'nothing to do here')
        #    return

        self.shank_info, exists = self._prepare_files_NP2_4(overwrite=overwrite)
        if exists:
            _logger.warning('One or more of the sub shank folders already exists, to force reprocessing'
                            'set overwrite to True')
            return

        # Initial checks out the way. Let's goooo!
        wg = WindowGenerator(self.nsamples, self.samples_window, self.samples_overlap)

        # TODO some logging of how far in we are
        for first, last in wg.firstlast:
            print(first)
            print(last)
            chunk_ap = self.sr[first:last, :self.napch].T
            chunk_ap_sync = self.sr[first:last, self.idxsyncch:].T
            chunk_lf = self.extract_lfp(self.sr[first:last, :self.napch].T)
            chunk_lf_sync = self.extract_lfp_sync(self.sr[first:last, self.idxsyncch:].T)

            chunk_ap2save = self._ind2save(chunk_ap, chunk_ap_sync, wg, ratio=1, type='ap')
            chunk_lf2save = self._ind2save(chunk_lf, chunk_lf_sync, wg, ratio=self.ratio,
                                           type='lf')

            self._split2shanks(chunk_ap2save, type='ap')
            self._split2shanks(chunk_lf2save, type='lf')

        self._closefiles(type='ap')
        self._closefiles(type='lf')

        self._writemetadata_ap()
        self._writemetadata_lf()

        if self.post_check:
            self.check()
        if self.delete_original:
            self.delete()

    def check(self):
        for sh in self.shank_info.keys():
            self.shank_info[sh]['self.sr'] = spikeglx.Reader(self.shank_info[sh]['ap_file'])

        wg = WindowGenerator(self.nsamples, self.samples_window, 0)
        for first, last in wg.firstlast:
            expected = self.sr[first:last, :]
            chunk = np.zeros_like(expected)
            for ish, sh in enumerate(self.shank_info.keys()):
                if ish == 0:
                    chunk[:, self.shank_info[sh]['chns']] = self.shank_info[sh]['self.sr'][first:last, :]
                else:
                    chunk[:, self.shank_info[sh]['chns'][:-1]] = \
                        self.shank_info[sh]['self.sr'][first:last, :-1]
            assert np.array_equal(expected, chunk)


    def delete(self):
        # TODO need to delete the original wahhhhh
        pass



    def _process_NP2_1(self):
        self.info = self._prepare_files_NP2_1()

        wg = WindowGenerator(self.nsamples, self.samples_window, self.samples_overlap)

        # TODO some logging of how far in we are
        for first, last in wg.firstlast:
            print(first)
            print(last)
            chunk_lf = self.extract_lfp(self.sr[first:last, :self.napch].T)
            chunk_lf_sync = self.extract_lfp(self.sr[first:last, self.idxsyncch:].T)

            chunk_lf2save = self._ind2save(chunk_lf, chunk_lf_sync, wg, ratio=self.ratio, type='lf')

            self._split2shanks(chunk_lf2save, type='lf')


    def _split2shanks(self, chunk, type='ap'):
        for sh in self.shank_info.keys():
            open = self.shank_info[sh][f'{type}_open_file']
            (chunk[:, self.shank_info[sh]['chns']]).tofile(open)


    def _ind2save(self, chunk, chunk_sync, wg, ratio=1, type='ap'):
        ind2save = [int(self.samples_taper * 2 / ratio),
                    int((self.samples_window - self.samples_taper * 2) / ratio)]
        if wg.iw == 0:
            ind2save[0] = 0
        if wg.iw == wg.nwin - 1:
            ind2save[1] = int(self.samples_window / ratio)

        print(ind2save)
        chunk2save = (np.c_[chunk[:, slice(*ind2save)].T / self.sr.channel_conversion_sample2v[type][:self.napch],
                              chunk_sync[:, slice(*ind2save)].T
                              / self.sr.channel_conversion_sample2v[type][self.idxsyncch:]]).astype(np.int16)

        return chunk2save

    def extract_lfp(self, chunk):
        chunk[:, :self.samples_taper] *= self.taper[:self.samples_taper]
        chunk[:, -self.samples_taper:] *= self.taper[self.samples_taper:]
        chunk = scipy.signal.sosfiltfilt(self.sos_lp, chunk)
        chunk = chunk[:, ::self.ratio]
        return chunk

    def extract_lfp_sync(self, chunk_sync):
        chunk_sync = chunk_sync[:, ::self.ratio]
        return chunk_sync

    def _closefiles(self, type='ap'):
        for sh in self.shank_info.keys():
            open = self.shank_info[sh].pop(f'{type}_open_file')
            open.close()

    def _writemetadata_ap(self):
        for sh in self.shank_info.keys():
            n_chns = len(self.shank_info[sh]['chns'])
            # First for the ap file
            meta_shank = copy.copy(self.sr.meta)
            meta_shank['acqApLfSy'][0] = n_chns - 1
            meta_shank['snsApLfSy'][0] = n_chns - 1
            meta_shank['nSavedChans'] = n_chns
            meta_shank['fileSizeBytes'] = self.shank_info[sh]['ap_file'].stat().st_size
            # TODO figure out if this is what we want
            meta_shank['snsSaveChanSubset'] = spikeglx._get_save_chan_subset(self.shank_info[sh]['chns'])
            # TODO do we want to remove the number of channels

            meta_shank['original_meta'] = False
            meta_file = self.shank_info[sh]['ap_file'].with_suffix('.meta')
            spikeglx.write_meta_data(meta_shank, meta_file)

    def _writemetadata_lf(self):
        for sh in self.shank_info.keys():
            n_chns = len(self.shank_info[sh]['chns'])
            meta_shank = copy.copy(self.sr.meta)
            meta_shank['acqApLfSy'][0] = 0
            meta_shank['acqApLfSy'][1] = n_chns - 1
            meta_shank['snsApLfSy'][0] = 0
            meta_shank['snsApLfSy'][1] = n_chns - 1
            meta_shank['nSavedChans'] = n_chns
            meta_shank['fileSizeBytes'] = self.shank_info[sh]['lf_file'].stat().st_size
            meta_shank['imSampRate'] = 2500
            meta_shank['snsSaveChanSubset'] = spikeglx._get_save_chan_subset(self.shank_info[sh]['chns'])
            meta_shank['original_meta'] = False
            meta_file = self.shank_info[sh]['lf_file'].with_suffix('.meta')
            spikeglx.write_meta_data(meta_shank, meta_file)



# my tests go here
import numpy as np
from ibllib.io import spikeglx

file_path = r'C:\Users\Mayo\Downloads\NP2\probe00\_spikeglx_ephysData_g0_t0.imec0.ap.bin'
# check the integrity of my down sampling and windowing
FS = 30000
NSAMPLES = int(2 * FS)
np0_5s = NP2Converter(file_path, post_check=False)
np0_5s.init_params(nsamples=NSAMPLES, nwindow=0.5*FS, extra='_0_5s', nshank=[0])
np0_5s.process()

np1s = NP2Converter(file_path, post_check=False)
np1s.init_params(nsamples=NSAMPLES, nwindow=1*FS, extra='_1s', nshank=[0])
np1s.process()

np2s = NP2Converter(file_path, post_check=False)
np2s.init_params(nsamples=NSAMPLES, nwindow=2*FS, extra='_2s', nshank=[0])
np2s.process()

sr = spikeglx.Reader(file_path)
sr0_5s_ap = spikeglx.Reader(np0_5s.shank_info['shank0']['ap_file'])
sr0_5s_lf = spikeglx.Reader(np0_5s.shank_info['shank0']['lf_file'])
sr1s_ap = spikeglx.Reader(np1s.shank_info['shank0']['ap_file'])
sr1s_lf = spikeglx.Reader(np1s.shank_info['shank0']['lf_file'])
sr2s_ap = spikeglx.Reader(np2s.shank_info['shank0']['ap_file'])
sr2s_lf = spikeglx.Reader(np2s.shank_info['shank0']['lf_file'])

# Make sure all the aps are the same regardless of window size we used
assert np.array_equal(sr0_5s_ap[:, :], sr1s_ap[:, :])
assert np.array_equal(sr0_5s_ap[:, :], sr2s_ap[:, :])
assert np.array_equal(sr1s_ap[:, :], sr2s_ap[:, :])

# Make sure all the lfps are the same regardless of window size we used
assert np.array_equal(sr0_5s_lf[:, :], sr1s_lf[:, :])
assert np.array_equal(sr0_5s_lf[:, :], sr2s_lf[:, :])
assert np.array_equal(sr1s_lf[:, :], sr2s_lf[:, :])

# Also check that the values are the same as the original. This is done by the check but
# to be extra safe!
assert np.array_equal(sr0_5s_ap[:, :], sr[:NSAMPLES, np0_5s.shank_info['shank0']['chns']])
assert np.array_equal(sr1s_ap[:, :], sr[:NSAMPLES, np1s.shank_info['shank0']['chns']])
assert np.array_equal(sr2s_ap[:, :], sr[:NSAMPLES, np2s.shank_info['shank0']['chns']])

# clean up
sr.close()
sr0_5s_ap.close()
sr0_5s_lf.close()
sr1s_ap.close()
sr1s_lf.close()
sr2s_ap.close()
sr2s_lf.close()

shutil.rmtree()




# check the whole process

# check that if we change part of the file the check is no longer valid

# check that overwrite flag works

# check that we can override the overwrite flag

# check that if we give it a processed file it doesn't do anything



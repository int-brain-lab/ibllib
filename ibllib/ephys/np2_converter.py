from ibllib.io import spikeglx
from ibllib.dsp.utils import WindowGenerator
import scipy.signal
import numpy as np
from pathlib import Path
import copy
import shutil
import logging
_logger = logging.getLogger('ibllib')


class NP2Converter:
    """
    Class used to 1. Extract LFP data from NP2 data and 2. If NP2.4 split the data into
    individual shanks
    """

    def __init__(self, ap_file, post_check=True, delete_original=False, compress=True):
        """
        :param ap_file: ap.bin spikeglx file to process
        :param post_check: whether to apply post-check integrity test to ensure split content is
        identical to original content (only applicable to NP2.4)
        :param delete_original: whether to delete the original ap file after data has been split
        :param compress: whether to apply mtscomp to extracted .bin files
        split into shanks (only applicable to NP2.4)
        """
        self.ap_file = Path(ap_file)
        self.sr = spikeglx.Reader(ap_file)
        self.post_check = post_check
        self.compress = compress
        self.delete_original = delete_original
        self.np_version = spikeglx._get_neuropixel_version_from_meta(self.sr.meta)
        self.check_metadata()
        self.init_params()

    def init_params(self, nsamples=None, nwindow=None, extra=None, nshank=None):
        """
        Initiliases parameters for processing.

        :param nsamples: the number of samples to process
        :param nwindow: the number of samples in each window when iterating through nsamples
        :param extra: extra string to add the individual shank folder names
        :param nshank: number of shanks to process, must be a list [0], you would only want to
        override this for testing purposes
        :return:
        """
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
        self.napch = int(self.sr.meta['snsApLfSy'][0])
        # Position of start of sync channels in the raw data
        self.idxsyncch = int(self.sr.meta['snsApLfSy'][0])

        self.extra = extra or ''
        self.nshank = nshank or None
        self.check_completed = False

    def check_metadata(self):
        """
        Checks the keys in meta data to see if we are trying to process an ap file that has already
        been split into shanks. If we are sets flag and prevents further processing occurring
        :return:
        """
        if self.sr.meta.get(f'{self.np_version}_shank', None) is not None:
            self.already_processed = True
        else:
            self.already_processed = False

    def process(self, overwrite=False):
        """
        Function to call to process NP2 data

        :param overwrite:
        :return:
        """
        if self.np_version == 'NP2.4':
            status = self._process_NP24(overwrite=overwrite)
        elif self.np_version == 'NP2.1':
            status = self._process_NP21(overwrite=overwrite)
        else:
            _logger.warning('Meta file is not of type NP2.1 or NP2.4, cannot process')
            status = 0
        return status

    def _process_NP24(self, overwrite=False):
        """
        Splits AP signal into individual shanks and also extracts the LFP signal. Writes ap and
        lf data to ap.bin and lf.bin files in individual shank folders. Don't call this function
        directly but access through process() method

        :param overwrite:
        :return:
        """
        if self.already_processed:
            _logger.warning('This ap file is an NP2.4 that has already been split into shanks, '
                            'nothing to do here')
            return 0

        self.shank_info = self._prepare_files_NP24(overwrite=overwrite)
        if self.already_exists:
            _logger.warning('One or more of the sub shank folders already exists, '
                            'to force reprocessing set overwrite to True')
            return 0

        # Initial checks out the way. Let's goooo!
        wg = WindowGenerator(self.nsamples, self.samples_window, self.samples_overlap)

        for first, last in wg.firstlast:
            chunk_ap = self.sr[first:last, :self.napch].T
            chunk_ap_sync = self.sr[first:last, self.idxsyncch:].T
            chunk_lf = self.extract_lfp(self.sr[first:last, :self.napch].T)
            chunk_lf_sync = self.extract_lfp_sync(self.sr[first:last, self.idxsyncch:].T)

            chunk_ap2save = self._ind2save(chunk_ap, chunk_ap_sync, wg, ratio=1, etype='ap')
            chunk_lf2save = self._ind2save(chunk_lf, chunk_lf_sync, wg, ratio=self.ratio,
                                           etype='lf')

            self._split2shanks(chunk_ap2save, etype='ap')
            self._split2shanks(chunk_lf2save, etype='lf')

            wg.print_progress(desc='Extracting LFP + Splitting:')

        self._closefiles(etype='ap')
        self._closefiles(etype='lf')

        self._writemetadata_ap()
        self._writemetadata_lf()

        if self.post_check:
            self.check_NP24()
        if self.compress:
            self.compress_NP24(overwrite=overwrite)
        if self.delete_original:
            self.delete_NP24()

        return 1

    def _prepare_files_NP24(self, overwrite=False):
        """
        Creates folders for individual shanks and creates and opens ap.bin and lf.bin files for
        each shank. Checks to see if and of the expected shank folders already exist
        and will only rerun if overwrite=True. Don't call this function directly but access through
        process() method

        :param overwrite: set to True to force rerunning even if lf.bin file already exists
        :return:
        """

        chn_info = spikeglx._map_channels_from_meta(self.sr.meta)
        n_shanks = self.nshank or np.unique(chn_info['shank']).astype(np.int16)
        label = self.ap_file.parent.parts[-1]
        shank_info = {}
        self.already_exists = False

        for sh in n_shanks:
            _shank_info = {}
            # channels for individual shank + sync channel
            _shank_info['chns'] = np.r_[np.where(chn_info['shank'] == sh)[0],
                                        np.array(spikeglx._get_sync_trace_indices_from_meta(
                                            self.sr.meta))]

            probe_path = self.ap_file.parent.parent.joinpath(label + chr(97 + int(sh)) + self.extra)

            if not probe_path.exists() or overwrite:
                if self.sr.is_mtscomp:
                    ap_file_bin = self.ap_file.with_suffix('.bin').name
                else:
                    ap_file_bin = self.ap_file.name
                probe_path.mkdir(parents=True, exist_ok=True)
                _shank_info['ap_file'] = probe_path.joinpath(ap_file_bin)
                _shank_info['ap_open_file'] = open(_shank_info['ap_file'], 'wb')
                _shank_info['lf_file'] = probe_path.joinpath(
                    ap_file_bin.replace('ap', 'lf'))
                _shank_info['lf_open_file'] = open(_shank_info['lf_file'], 'wb')
            else:
                self.already_exists = True
                _logger.warning('One or more of the sub shank folders already exists, '
                                'to force reprocessing set overwrite to True')
            shank_info[f'shank{sh}'] = _shank_info

        return shank_info

    def _process_NP21(self, overwrite=False):
        """
        Extracts LFP signal from original data and writes to lf.bin file. Also created lf.meta
        file. Don't call this function directly but access through process() method

        :param overwrite: set to True to force rerunning even if lf.bin file already exists
        :return:
        """
        if self.already_processed:
            _logger.warning('This ap file is an NP2.4 that has already been split into shanks, '
                            'nothing to do here')
            return 0

        self.shank_info = self._prepare_files_NP21(overwrite=overwrite)
        if self.already_exists:
            _logger.warning('One or more of the sub shank folders already exists, '
                            'to force reprocessing set overwrite to True')
            return 0

        wg = WindowGenerator(self.nsamples, self.samples_window, self.samples_overlap)

        for first, last in wg.firstlast:

            chunk_lf = self.extract_lfp(self.sr[first:last, :self.napch].T)
            chunk_lf_sync = self.extract_lfp_sync(self.sr[first:last, self.idxsyncch:].T)

            chunk_lf2save = self._ind2save(chunk_lf, chunk_lf_sync, wg, ratio=self.ratio,
                                           etype='lf')

            self._split2shanks(chunk_lf2save, etype='lf')

            wg.print_progress(desc='Extracting LFP:')

        self._closefiles(etype='lf')

        self._writemetadata_lf()

        if self.compress:
            self.compress_NP21(overwrite=overwrite)

        return 1

    def _prepare_files_NP21(self, overwrite=False):
        """
        Creates and opens lf.bin file in order to extract the lfp signal from full signal. Checks
        to see if file already exists and will only rerun if overwrite=True. Don't call this
        function directly but access through process() method

        :param overwrite: set to True to force rerunning even if lf.bin file already exists
        :return:
        """

        chn_info = spikeglx._map_channels_from_meta(self.sr.meta)
        n_shanks = np.unique(chn_info['shank']).astype(np.int16)
        assert (len(n_shanks) == 1)
        shank_info = {}
        self.already_exists = False

        lf_file = self.ap_file.parent.joinpath(self.ap_file.name.replace('ap', 'lf')).with_suffix('.bin')
        lf_cbin_file = lf_file.with_suffix('.cbin')
        if not (lf_file.exists() or lf_cbin_file.exists()) or overwrite:
            for sh in n_shanks:
                _shank_info = {}
                # channels for individual shank + sync channel
                _shank_info['chns'] = np.r_[np.where(chn_info['shank'] == sh)[0],
                                            np.array(spikeglx._get_sync_trace_indices_from_meta(
                                                self.sr.meta))]
                _shank_info['lf_file'] = lf_file
                _shank_info['lf_open_file'] = open(_shank_info['lf_file'], 'wb')

                shank_info[f'shank{sh}'] = _shank_info
        else:
            self.already_exists = True
            _logger.warning('LF file for this probe already exists, '
                            'to force reprocessing set overwrite to True')

        return shank_info

    def check_NP24(self):
        """
        Check that the splitting into shanks process has completed correctly. Compares the original
        file to the reconstructed file from the individual shanks

        :return:
        """
        for sh in self.shank_info.keys():
            self.shank_info[sh]['sr'] = spikeglx.Reader(self.shank_info[sh]['ap_file'])

        wg = WindowGenerator(self.nsamples, self.samples_window, 0)
        for first, last in wg.firstlast:
            expected = self.sr[first:last, :]
            chunk = np.zeros_like(expected)
            for ish, sh in enumerate(self.shank_info.keys()):
                if ish == 0:
                    chunk[:, self.shank_info[sh]['chns']] = self.shank_info[sh]['sr'][first:last, :]
                else:
                    chunk[:, self.shank_info[sh]['chns'][:-1]] = \
                        self.shank_info[sh]['sr'][first:last, :-1]
            assert np.array_equal(expected, chunk), \
                'data in original file and split files do no match'

            wg.print_progress(desc='Checking:')

        # close the sglx instances once we are done checking
        for sh in self.shank_info.keys():
            sr = self.shank_info[sh].pop('sr')
            sr.close()

        self.check_completed = True

    def compress_NP24(self, overwrite=False, **kwargs):
        """
        Compress spikeglx files
        :return:
        """
        for sh in self.shank_info.keys():
            bin_file = self.shank_info[sh]['ap_file']
            if overwrite:
                cbin_file = bin_file.with_suffix('.cbin')
                cbin_file.unlink()

            sr_ap = spikeglx.Reader(bin_file)
            cbin_file = sr_ap.compress_file(**kwargs)
            sr_ap.close()
            bin_file.unlink()
            self.shank_info[sh]['ap_file'] = cbin_file

            bin_file = self.shank_info[sh]['lf_file']
            if overwrite:
                cbin_file = bin_file.with_suffix('.cbin')
                cbin_file.unlink()
            sr_lf = spikeglx.Reader(bin_file)
            cbin_file = sr_lf.compress_file(**kwargs)
            sr_lf.close()
            bin_file.unlink()
            self.shank_info[sh]['lf_file'] = cbin_file

    def compress_NP21(self, overwrite=False):
        """
        Compress spikeglx files
        :return:
        """
        for sh in self.shank_info.keys():
            if not self.sr.is_mtscomp:
                cbin_file = self.sr.compress_file()
                self.sr.close()
                self.ap_file.unlink()
                self.ap_file = cbin_file
                self.sr = spikeglx.Reader(self.ap_file)

            bin_file = self.shank_info[sh]['lf_file']
            if overwrite:
                cbin_file = bin_file.with_suffix('.cbin')
                cbin_file.unlink()
            sr_lf = spikeglx.Reader(bin_file)
            cbin_file = sr_lf.compress_file()
            sr_lf.close()
            bin_file.unlink()
            self.shank_info[sh]['lf_file'] = cbin_file

    def delete_NP24(self):
        """
        Delete the original ap file that doesn't has all shanks in one file

        :return:
        """
        if self.check_completed and self.delete_original:
            _logger.info(f'Removing original files in folder {self.ap_file.parent}')
            self.sr.close()
            shutil.rmtree(self.ap_file.parent)

    def _split2shanks(self, chunk, etype='ap'):
        """
        Splits the signal on the 384 channels into the individual shanks and saves to file

        :param chunk: portion of signal with all 384 channels
        :param type: ephys type, either 'ap' or 'lf'
        :return:
        """

        for sh in self.shank_info.keys():
            open = self.shank_info[sh][f'{etype}_open_file']
            (chunk[:, self.shank_info[sh]['chns']]).tofile(open)

    def _ind2save(self, chunk, chunk_sync, wg, ratio=1, etype='ap'):
        """
        Determines the portion of the full chunk to save based on the window and taper used. Cuts
        off beginning and end to get rid of filtering/ decimating artefacts

        :param chunk: chunk of ephys signal
        :param chunk_sync: chunk of sync signal
        :param wg: Window generator object
        :param ratio: downsample ratio
        :param etype: ephys type, either 'ap' or 'lf'
        :return:
        """

        ind2save = [int(self.samples_taper * 2 / ratio),
                    int((self.samples_window - self.samples_taper * 2) / ratio)]
        if wg.iw == 0:
            ind2save[0] = 0
        if wg.iw == wg.nwin - 1:
            ind2save[1] = int(self.samples_window / ratio)

        chunk2save = (np.c_[chunk[:, slice(*ind2save)].T /
                            self.sr.channel_conversion_sample2v[etype][:self.napch],
                            chunk_sync[:, slice(*ind2save)].T /
                            self.sr.channel_conversion_sample2v[etype][self.idxsyncch:]]).\
            astype(np.int16)

        return chunk2save

    def extract_lfp(self, chunk):
        """
        Extracts LFP signal from full band signal, first applies low pass to anti-alias and LP,
        then downsamples

        :param chunk: portion of signal to extract LFP from
        :return: LFP signal
        """

        chunk[:, :self.samples_taper] *= self.taper[:self.samples_taper]
        chunk[:, -self.samples_taper:] *= self.taper[self.samples_taper:]
        chunk = scipy.signal.sosfiltfilt(self.sos_lp, chunk)
        chunk = chunk[:, ::self.ratio]
        return chunk

    def extract_lfp_sync(self, chunk_sync):
        """
        Extracts downsampled signal of imec sync trace

        :param chunk_sync: portion of sync signal to downsample
        :return: downsampled sync signal
        """

        chunk_sync = chunk_sync[:, ::self.ratio]
        return chunk_sync

    def _closefiles(self, etype='ap'):
        """
        Close .bin files that were being written to

        :param etype: ephys type, either 'ap' or 'lf'
        :return:
        """

        for sh in self.shank_info.keys():
            open = self.shank_info[sh].pop(f'{etype}_open_file')
            open.close()

    def _writemetadata_ap(self):
        """
        Function to create ap meta data file. Adapts the relevant keys in the spikeglx meta file
        to contain the correct number of channels. Also adds key to indicate that this is not an
        original meta data file, but one that has been adapted

        :return:
        """

        for sh in self.shank_info.keys():
            n_chns = len(self.shank_info[sh]['chns'])
            # First for the ap file
            meta_shank = copy.deepcopy(self.sr.meta)
            meta_shank['acqApLfSy'][0] = n_chns - 1
            meta_shank['snsApLfSy'][0] = n_chns - 1
            meta_shank['nSavedChans'] = n_chns
            meta_shank['fileSizeBytes'] = self.shank_info[sh]['ap_file'].stat().st_size
            meta_shank['snsSaveChanSubset_orig'] = \
                spikeglx._get_savedChans_subset(self.shank_info[sh]['chns'])
            meta_shank['snsSaveChanSubset'] = f'0:{n_chns-1}'
            meta_shank['original_meta'] = False
            meta_shank[f'{self.np_version}_shank'] = int(sh[-1])
            meta_file = self.shank_info[sh]['ap_file'].with_suffix('.meta')
            spikeglx.write_meta_data(meta_shank, meta_file)

    def _writemetadata_lf(self):
        """
        Function to create lf meta data file. Adapts the relevant keys in the spikeglx meta file
        to contain the correct number of channels. Also adds key to indicate that this is not an
        original meta data file, but one that has been adapted

        :return:
        """

        for sh in self.shank_info.keys():
            n_chns = len(self.shank_info[sh]['chns'])
            meta_shank = copy.deepcopy(self.sr.meta)
            meta_shank['acqApLfSy'][0] = 0
            meta_shank['acqApLfSy'][1] = n_chns - 1
            meta_shank['snsApLfSy'][0] = 0
            meta_shank['snsApLfSy'][1] = n_chns - 1
            meta_shank['fileSizeBytes'] = self.shank_info[sh]['lf_file'].stat().st_size
            meta_shank['imSampRate'] = self.fs_lf
            if self.np_version == 'NP2.4':
                meta_shank['snsSaveChanSubset_orig'] = \
                    spikeglx._get_savedChans_subset(self.shank_info[sh]['chns'])
                meta_shank['snsSaveChanSubset'] = f'0:{n_chns-1}'
                meta_shank['nSavedChans'] = n_chns
            meta_shank['original_meta'] = False
            meta_shank[f'{self.np_version}_shank'] = int(sh[-1])
            meta_file = self.shank_info[sh]['lf_file'].with_suffix('.meta')
            spikeglx.write_meta_data(meta_shank, meta_file)

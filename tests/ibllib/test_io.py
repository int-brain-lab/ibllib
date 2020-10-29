import unittest
import os
import uuid
import tempfile
from pathlib import Path
import shutil

import numpy as np

from ibllib.io import params, flags, jsonable, spikeglx, hashfile, misc
import ibllib.io.raw_data_loaders as raw


class TestsParams(unittest.TestCase):

    def setUp(self):
        self.par_dict = {'A': 'tata',
                         'O': 'toto',
                         'I': 'titi',
                         'num': 15,
                         'liste': [1, 'turlu'],
                         'apath': Path('/gna/gna/gna')}
        params.write('toto', self.par_dict)
        params.write('toto', params.from_dict(self.par_dict))

    def test_params(self):
        #  first go to and from dictionary
        par_dict = self.par_dict
        par = params.from_dict(par_dict)
        self.assertEqual(params.as_dict(par), par_dict)
        # next go to and from dictionary via json
        par2 = params.read('toto')
        self.assertEqual(par, par2)

    def test_new_default_param(self):
        # in this case an updated version of the codes brings in a new parameter
        default = {'A': 'tata2',
                   'O': 'toto2',
                   'I': 'titi2',
                   'E': 'tete2',
                   'num': 15,
                   'liste': [1, 'turlu']}
        expected_result = {'A': 'tata',
                           'O': 'toto',
                           'I': 'titi',
                           'num': 15,
                           'liste': [1, 'turlu'],
                           'apath': str(Path('/gna/gna/gna')),
                           'E': 'tete2',
                           }
        par2 = params.read('toto', default=default)
        self.assertEqual(par2, params.from_dict(expected_result))
        # on the next path the parameter has been added to the param file
        par2 = params.read('toto', default=default)
        self.assertEqual(par2, params.from_dict(expected_result))
        # check that it doesn't break if a named tuple is given instead of a dict
        par3 = params.read('toto', default=par2)
        self.assertEqual(par2, par3)
        # check that a non-existing parfile returns None
        pstring = str(uuid.uuid4())
        par = params.read(pstring)
        self.assertIsNone(par)
        # check that a non-existing parfile with default returns default
        par = params.read(pstring, default=default)
        self.assertEqual(par, params.from_dict(default))
        # even if this default is a Params named tuple
        par = params.read(pstring, default=par)
        self.assertEqual(par, params.from_dict(default))

    def tearDown(self):
        # at last delete the param file
        os.remove(params.getfile('toto'))


class TestsRawDataLoaders(unittest.TestCase):

    def setUp(self):
        self.tempfile = tempfile.NamedTemporaryFile(delete=False)

    def testFlagFileRead(self):
        # empty file should return True
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)
        # test with 2 lines and a trailing
        with open(self.tempfile.name, 'w+') as fid:
            fid.write('file1\nfile2\n')
        self.assertEqual(set(flags.read_flag_file(self.tempfile.name)), set(['file1', 'file2']))
        # test with 2 lines and a trailing, Windows convention
        with open(self.tempfile.name, 'w+') as fid:
            fid.write('file1\r\nfile2\r\n')
        self.assertEqual(set(flags.read_flag_file(self.tempfile.name)), set(['file1', 'file2']))

    def testAppendFlagFile(self):
        #  DO NOT CHANGE THE ORDER OF TESTS BELOW
        # prepare a file with 3 dataset types
        file_list = ['_ibl_extraRewards.times', '_ibl_lickPiezo.raw', '_ibl_lickPiezo.timestamps']
        with open(self.tempfile.name, 'w+') as fid:
            fid.write('\n'.join(file_list))
        self.assertEqual(set(flags.read_flag_file(self.tempfile.name)), set(file_list))

        # with an existing file containing files, writing more files append to it
        file_list_2 = ['turltu']
        # also makes sure that if a string is provided it works
        flags.write_flag_file(self.tempfile.name, file_list_2[0])
        self.assertEqual(set(flags.read_flag_file(self.tempfile.name)),
                         set(file_list + file_list_2))

        # writing again keeps unique file values
        flags.write_flag_file(self.tempfile.name, file_list_2[0])
        n = sum([1 for f in flags.read_flag_file(self.tempfile.name) if f == file_list_2[0]])
        self.assertEqual(n, 1)

        # with an existing file containing files, writing empty filelist returns True for all files
        flags.write_flag_file(self.tempfile.name, None)
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)

        # with an existing empty file, writing filelist returns True for all files
        flags.write_flag_file(self.tempfile.name, ['file1', 'file2'])
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)

        # makes sure that read after write empty list also returns True
        flags.write_flag_file(self.tempfile.name, [])
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)

        # with an existing empty file, writing filelist returns the list if clobber
        flags.write_flag_file(self.tempfile.name, ['file1', 'file2', 'file3'], clobber=True)
        self.assertEqual(set(flags.read_flag_file(self.tempfile.name)),
                         set(['file1', 'file2', 'file3']))

        # test the removal of a file within the list
        flags.excise_flag_file(self.tempfile.name, removed_files='file1')
        self.assertEqual(sorted(flags.read_flag_file(self.tempfile.name)), ['file2', 'file3'])

        # if file-list is True it means all files and file_list should be empty after read
        flags.write_flag_file(self.tempfile.name, file_list=True)
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)

    def test_load_encoder_trial_info(self):
        self.session = Path(__file__).parent.joinpath('extractors', 'data', 'session_biased_ge5')
        data = raw.load_encoder_trial_info(self.session)
        self.assertTrue(data is not None)

    def tearDown(self):
        self.tempfile.close()
        os.unlink(self.tempfile.name)


class TestsJsonable(unittest.TestCase):

    def testReadWrite(self):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        data = [{'a': 'thisisa', 'b': 1, 'c': [1, 2, 3]},
                {'a': 'thisisb', 'b': 2, 'c': [2, 3, 4]}]
        jsonable.write(tfile.name, data)
        data2 = jsonable.read(tfile.name)
        self.assertEqual(data, data2)
        jsonable.append(tfile.name, data)
        data3 = jsonable.read(tfile.name)
        self.assertEqual(data + data, data3)
        tfile.close()
        os.unlink(tfile.name)


class TestSpikeGLX_glob_ephys(unittest.TestCase):
    """
    Creates mock acquisition folders architecture (omitting metadata files):
    ├── 3A
    │   ├── imec0
    │   │   ├── sync_testing_g0_t0.imec0.ap.bin
    │   │   └── sync_testing_g0_t0.imec0.lf.bin
    │   └── imec1
    │       ├── sync_testing_g0_t0.imec1.ap.bin
    │       └── sync_testing_g0_t0.imec1.lf.bin
    └── 3B
        ├── sync_testing_g0_t0.nidq.bin
        ├── imec0
        │   ├── sync_testing_g0_t0.imec0.ap.bin
        │   └── sync_testing_g0_t0.imec0.lf.bin
        └── imec1
            ├── sync_testing_g0_t0.imec1.ap.bin
            └── sync_testing_g0_t0.imec1.lf.bin
    """
    def setUp(self):
        def touchfile(p):
            if isinstance(p, Path):
                try:
                    p.parent.mkdir(exist_ok=True, parents=True)
                    p.touch(exist_ok=True)
                except Exception:
                    print('tutu')

        def create_tree(root_dir, dico):
            root_dir.mkdir(exist_ok=True, parents=True)
            for ldir in dico:
                for k in ldir:
                    if k == 'path' or k == 'label':
                        continue
                    touchfile(ldir[k])
                    Path(ldir[k]).with_suffix('.meta').touch()

        self.tmpdir = Path(tempfile.gettempdir()) / 'test_glob_ephys'
        self.tmpdir.mkdir(exist_ok=True)
        self.dir3a = self.tmpdir.joinpath('3A').joinpath('raw_ephys_data')
        self.dir3b = self.tmpdir.joinpath('3B').joinpath('raw_ephys_data')
        self.dict3a = [{'label': 'imec0',
                        'ap': self.dir3a / 'imec0' / 'sync_testing_g0_t0.imec0.ap.bin',
                        'lf': self.dir3a / 'imec0' / 'sync_testing_g0_t0.imec0.lf.bin',
                        'path': self.dir3a / 'imec0'},
                       {'label': 'imec1',
                        'ap': self.dir3a / 'imec1' / 'sync_testing_g0_t0.imec1.ap.bin',
                        'lf': self.dir3a / 'imec1' / 'sync_testing_g0_t0.imec1.lf.bin',
                        'path': self.dir3a / 'imec1'}]
        # surprise ! one of them happens to be compressed
        self.dict3b = [{'label': 'imec0',
                        'ap': self.dir3b / 'imec0' / 'sync_testing_g0_t0.imec0.ap.cbin',
                        'lf': self.dir3b / 'imec0' / 'sync_testing_g0_t0.imec0.lf.bin',
                        'path': self.dir3b / 'imec0'},
                       {'label': 'imec1',
                        'ap': self.dir3b / 'imec1' / 'sync_testing_g0_t0.imec1.ap.bin',
                        'lf': self.dir3b / 'imec1' / 'sync_testing_g0_t0.imec1.lf.bin',
                        'path': self.dir3b / 'imec1'},
                       {'label': '',
                        'nidq': self.dir3b / 'sync_testing_g0_t0.nidq.bin',
                        'path': self.dir3b}]
        create_tree(self.dir3a, self.dict3a)
        create_tree(self.dir3b, self.dict3b)

    def test_glob_ephys(self):
        def dict_equals(d1, d2):
            return all([x in d1 for x in d2]) and all([x in d2 for x in d1])
        ef3b = spikeglx.glob_ephys_files(self.dir3b)
        ef3a = spikeglx.glob_ephys_files(self.dir3a)
        # test glob
        self.assertTrue(dict_equals(self.dict3a, ef3a))
        self.assertTrue(dict_equals(self.dict3b, ef3b))
        # test the version from glob
        self.assertTrue(spikeglx.get_neuropixel_version_from_files(ef3a) == '3A')
        self.assertTrue(spikeglx.get_neuropixel_version_from_files(ef3b) == '3B')
        # test the version from paths
        self.assertTrue(spikeglx.get_neuropixel_version_from_folder(self.dir3a) == '3A')
        self.assertTrue(spikeglx.get_neuropixel_version_from_folder(self.dir3b) == '3B')
        self.dir3b.joinpath('imec1', 'sync_testing_g0_t0.imec1.ap.bin').unlink()
        self.assertEqual(spikeglx.glob_ephys_files(self.dir3b.joinpath('imec1')), [])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)


class TestsSpikeGLX_compress(unittest.TestCase):

    def setUp(self):
        self._tempdir = tempfile.TemporaryDirectory()
        self.workdir = Path(self._tempdir.name)
        file_meta = Path(__file__).parent.joinpath('fixtures', 'io', 'spikeglx',
                                                   'sample3A_short_g0_t0.imec.ap.meta')
        self.file_bin = spikeglx._mock_spikeglx_file(
            self.workdir.joinpath('sample3A_short_g0_t0.imec.ap.bin'), file_meta, ns=76104,
            nc=385, sync_depth=16, random=True)['bin_file']
        self.sr = spikeglx.Reader(self.file_bin)

    def test_read_slices(self):
        sr = self.sr
        s2mv = sr.channel_conversion_sample2v['ap'][0]
        # test the slicing of reader object
        self.assertTrue(np.all(np.isclose(sr._raw[5:500, :-1] * s2mv, sr[5:500, :-1])))
        self.assertTrue(np.all(np.isclose(sr._raw[5:500, 5] * s2mv, sr[5:500, 5])))
        self.assertTrue(np.all(np.isclose(sr._raw[5, :-1] * s2mv, sr[5, :-1])))
        self.assertTrue(sr._raw[55, 5] * s2mv == sr[55, 5])
        self.assertTrue(np.all(np.isclose(sr._raw[55] * s2mv, sr[55])))
        self.assertTrue(np.all(np.isclose(sr._raw[5:500] * s2mv, sr[5:500])[:, :-1]))

    def test_compress(self):

        def compare_data(sr0, sr1):
            # test direct reading through memmap / mtscompreader
            self.assertTrue(np.all(sr0._raw[1200:1210, 12] == sr1._raw[1200:1210, 12]))
            # test reading through methods
            d0, s0 = sr0.read_samples(1200, 54245)
            d1, s1 = sr1.read_samples(1200, 54245)
            self.assertTrue(np.all(d0 == d1))
            self.assertTrue(np.all(s0 == s1))

        # create a reference file that will serve to compare for inplace operations
        ref_file = self.file_bin.parent.joinpath('REF_' + self.file_bin.name)
        ref_meta = self.file_bin.parent.joinpath('REF_' + self.file_bin.with_suffix('.meta').name)
        shutil.copy(self.file_bin, ref_file)
        shutil.copy(self.file_bin.with_suffix('.meta'), ref_meta)
        sr_ref = spikeglx.Reader(ref_file)

        # test file compression copy
        self.assertFalse(self.sr.is_mtscomp)
        self.file_cbin = self.sr.compress_file()
        self.sc = spikeglx.Reader(self.file_cbin)
        self.assertTrue(self.sc.is_mtscomp)
        compare_data(sr_ref, self.sc)

        # test decompression in-place
        self.sc.decompress_file(keep_original=False, overwrite=True)
        compare_data(sr_ref, self.sc)
        self.assertFalse(self.sr.is_mtscomp)
        self.assertFalse(self.file_cbin.exists())
        compare_data(sr_ref, self.sc)

        # test compression in-place
        self.sc.compress_file(keep_original=False, overwrite=True)
        compare_data(sr_ref, self.sc)
        self.assertTrue(self.sc.is_mtscomp)
        self.assertTrue(self.file_cbin.exists())
        self.assertFalse(self.file_bin.exists())
        compare_data(sr_ref, self.sc)


class TestsSpikeGLX_Meta(unittest.TestCase):

    def setUp(self):
        self.workdir = Path(__file__).parent / 'fixtures' / 'io' / 'spikeglx'
        self.meta_files = list(Path.glob(self.workdir, '*.meta'))

    def test_read_corrupt(self):
        # nidq has 1 analog and 1 digital sync channels
        self.tdir = tempfile.TemporaryDirectory(prefix='glx_test')
        int2volts = 5 / 32768
        nidq = spikeglx._mock_spikeglx_file(
            Path(self.tdir.name).joinpath('sample3B_g0_t0.nidq.bin'),
            self.workdir / 'sample3B_g0_t0.nidq.meta',
            ns=32, nc=2, sync_depth=8, int2volts=int2volts, corrupt=True)
        self.assert_read_glx(nidq)

    def test_read_nidq(self):
        # nidq has 1 analog and 1 digital sync channels
        self.tdir = tempfile.TemporaryDirectory(prefix='glx_test')
        int2volts = 5 / 32768
        nidq = spikeglx._mock_spikeglx_file(
            Path(self.tdir.name).joinpath('sample3B_g0_t0.nidq.bin'),
            self.workdir / 'sample3B_g0_t0.nidq.meta',
            ns=32, nc=2, sync_depth=8, int2volts=int2volts)
        self.assert_read_glx(nidq)

    def test_read_3A(self):
        self.tdir = tempfile.TemporaryDirectory(prefix='glx_test')
        bin_3a = spikeglx._mock_spikeglx_file(
            Path(self.tdir.name).joinpath('sample3A_g0_t0.imec.ap.bin'),
            self.workdir / 'sample3A_g0_t0.imec.ap.meta',
            ns=32, nc=385, sync_depth=16)
        self.assert_read_glx(bin_3a)

    def test_read_3B(self):
        self.tdir = tempfile.TemporaryDirectory(prefix='glx_test')
        bin_3b = spikeglx._mock_spikeglx_file(
            Path(self.tdir.name).joinpath('sample3B_g0_t0.imec1.ap.bin'),
            self.workdir / 'sample3B_g0_t0.imec1.ap.meta',
            ns=32, nc=385, sync_depth=16)
        self.assert_read_glx(bin_3b)

    def test_check_ephys_file(self):
        self.tdir = tempfile.TemporaryDirectory(prefix='glx_test')
        bin_3b = spikeglx._mock_spikeglx_file(
            Path(self.tdir.name).joinpath('sample3B_g0_t0.imec1.ap.bin'),
            self.workdir / 'sample3B_g0_t0.imec1.ap.meta',
            ns=32, nc=385, sync_depth=16)
        self.assertEqual(hashfile.md5(bin_3b['bin_file']), "207ba1666b866a091e5bb8b26d19733f")
        self.assertEqual(hashfile.sha1(bin_3b['bin_file']),
                         '1bf3219c35dea15409576f6764dd9152c3f8a89c')
        sr = spikeglx.Reader(bin_3b['bin_file'])
        self.assertTrue(sr.verify_hash())

    def assert_read_glx(self, tglx):
        sr = spikeglx.Reader(tglx['bin_file'])
        dexpected = sr.channel_conversion_sample2v[sr.type] * tglx['D']
        d, sync = sr.read_samples(0, tglx['ns'])
        # could be rounding errors with non-integer sampling rates
        self.assertTrue(sr.nc == tglx['nc'])
        self.assertTrue(sr.ns == tglx['ns'])
        # test the data reading with gain
        self.assertTrue(np.all(np.isclose(dexpected, d)))
        # test the sync reading, one front per channel
        self.assertTrue(np.sum(sync) == tglx['sync_depth'])
        for m in np.arange(tglx['sync_depth']):
            self.assertTrue(sync[m + 1, m] == 1)
        if sr.type in ['ap', 'lf']:  # exclude nidq from the slicing circus
            # teast reading only one channel
            d, _ = sr.read(slice(None), 10)
            self.assertTrue(np.all(np.isclose(d, dexpected[:, 10])))
            # test reading only one time
            d, _ = sr.read(5, slice(None))
            self.assertTrue(np.all(np.isclose(d, dexpected[5, :])))
            # test reading a few times
            d, _ = sr.read(slice(5, 7), slice(None))
            self.assertTrue(np.all(np.isclose(d, dexpected[5:7, :])))
            d, _ = sr.read([5, 6], slice(None))
            self.assertTrue(np.all(np.isclose(d, dexpected[5:7, :])))
            # test reading a few channels
            d, _ = sr.read(slice(None), slice(300, 310))
            self.assertTrue(np.all(np.isclose(d, dexpected[:, 300:310])))
            # test reading a few channels with a numpy array of indices
            ind = np.array([300, 302])
            d, _ = sr.read(slice(None), ind)
            self.assertTrue(np.all(np.isclose(d, dexpected[:, ind])))
            # test double slicing
            d, _ = sr.read(slice(5, 10), slice(300, 310))
            self.assertTrue(np.all(np.isclose(d, dexpected[5:10, 300:310])))
            # test empty slices
            d, _ = sr.read(slice(5, 10), [])
            self.assertTrue(d.size == 0)
            d, _ = sr.read([], [])
            self.assertTrue(d.size == 0)
            d, _ = sr.read([], slice(300, 310))
            self.assertTrue(d.size == 0)
            a = sr.read_sync_analog()
            self.assertIsNone(a)
            # test the read_samples method (should be deprecated ?)
            d, _ = sr.read_samples(0, 500, ind)
            self.assertTrue(np.all(np.isclose(d, dexpected[0:500, ind])))
            d, _ = sr.read_samples(0, 500)
            self.assertTrue(np.all(np.isclose(d, dexpected[0:500, :])))
        else:
            s = sr.read_sync()
            self.assertTrue(s.shape[1] == 17)

    def testGetSerialNumber(self):
        self.meta_files.sort()
        expected = [641251510, 641251510, 641251510, 17216703352, 18005116811, 18005116811, None]
        for meta_data_file, res in zip(self.meta_files, expected):
            md = spikeglx.read_meta_data(meta_data_file)
            self.assertEqual(md.serial, res)

    def testGetRevisionAndType(self):
        for meta_data_file in self.meta_files:
            md = spikeglx.read_meta_data(meta_data_file)
            self.assertTrue(len(md.keys()) >= 37)
            # test getting revision
            revision = meta_data_file.name[6:8]
            self.assertEqual(spikeglx._get_neuropixel_version_from_meta(md)[0:2], revision)
            # test getting acquisition type
            type = meta_data_file.name.split('.')[-2]
            self.assertEqual(spikeglx._get_type_from_meta(md), type)

    def testReadChannelGainAPLF(self):
        for meta_data_file in self.meta_files:
            print(meta_data_file)
            if meta_data_file.name.split('.')[-2] not in ['lf', 'ap']:
                continue
            md = spikeglx.read_meta_data(meta_data_file)
            cg = spikeglx._conversion_sample2v_from_meta(md)
            i2v = md.get('imAiRangeMax') / 512
            self.assertTrue(np.all(cg['lf'][0:-1] == i2v / 250))
            self.assertTrue(np.all(cg['ap'][0:-1] == i2v / 500))
            # also test consistent dimension with nchannels
            nc = spikeglx._get_nchannels_from_meta(md)
            self.assertTrue(len(cg['ap']) == len(cg['lf']) == nc)

    def testGetAnalogSyncIndex(self):
        for meta_data_file in self.meta_files:
            md = spikeglx.read_meta_data(meta_data_file)
            if spikeglx._get_type_from_meta(md) in ['ap', 'lf']:
                self.assertTrue(spikeglx._get_analog_sync_trace_indices_from_meta(md) == [])
            else:
                self.assertEqual(spikeglx._get_analog_sync_trace_indices_from_meta(md), [0])

    def testReadChannelGainNIDQ(self):
        for meta_data_file in self.meta_files:
            if meta_data_file.name.split('.')[-2] not in ['nidq']:
                continue
            md = spikeglx.read_meta_data(meta_data_file)
            nc = spikeglx._get_nchannels_from_meta(md)
            cg = spikeglx._conversion_sample2v_from_meta(md)
            i2v = md.get('niAiRangeMax') / 32768
            self.assertTrue(np.all(cg['nidq'][slice(0, int(np.sum(md.acqMnMaXaDw[:3])))] == i2v))
            self.assertTrue(np.all(cg['nidq'][slice(int(np.sum(md.acqMnMaXaDw[-1])), None)] == 1.))
            self.assertTrue(len(cg['nidq']) == nc)

    def testReadChannelMap(self):
        for meta_data_file in self.meta_files:
            md = spikeglx.read_meta_data(meta_data_file)
            cm = spikeglx._map_channels_from_meta(md)
            if 'snsShankMap' in md.keys():
                self.assertEqual(set(cm.keys()), set(['shank', 'col', 'row', 'flag']))

    def testSplitSyncTrace(self):
        sc = np.uint16(2 ** np.linspace(-1, 15, 17))
        out = spikeglx.split_sync(sc)
        for m in range(1, 16):
            self.assertEqual(np.sum(out[m]), 1)
            self.assertEqual(out[m, m - 1], 1)


class TestsHardwareParameters(unittest.TestCase):

    def setUp(self):
        self.workdir = Path(__file__).parent / 'fixtures' / 'io' / 'spikeglx'
        self.map3A = {'left_camera': 2,
                      'right_camera': 3,
                      'body_camera': 4,
                      'bpod': 7,
                      'frame2ttl': 12,
                      'rotary_encoder_0': 13,
                      'rotary_encoder_1': 14,
                      'audio': 15}
        self.map3B = {'left_camera': 0,
                      'right_camera': 1,
                      'body_camera': 2,
                      'imec_sync': 3,
                      'frame2ttl': 4,
                      'rotary_encoder_0': 5,
                      'rotary_encoder_1': 6,
                      'audio': 7,
                      'bpod': 16}
        self.file3a = self.workdir / 'sample3A_g0_t0.imec.wiring.json'
        self.file3b = self.workdir / 'sample3B_g0_t0.nidq.wiring.json'

    def test_default_values(self):
        from ibllib.io.extractors import ephys_fpga
        self.assertEqual(ephys_fpga.CHMAPS['3A']['ap'], self.map3A)
        self.assertEqual(ephys_fpga.CHMAPS['3B']['nidq'], self.map3B)

    def test_get_wiring(self):
        # get params providing full file path
        par = spikeglx.get_hardware_config(self.workdir)
        self.assertTrue(par)
        with tempfile.TemporaryDirectory() as tdir:
            # test from empty directory
            self.assertIsNone(spikeglx.get_hardware_config(tdir))
            # test from directory
            shutil.copy(self.file3a, Path(tdir) / self.file3a.name)
            par3a = spikeglx.get_hardware_config(tdir)
            # test from full file path
            par3a_ = spikeglx.get_hardware_config(Path(tdir) / self.file3a.name)
            self.assertEqual(par3a, par3a_)

    def test_get_channel_map(self):
        map = spikeglx.get_sync_map(self.file3a)
        self.assertEqual(map, self.map3A)
        map = spikeglx.get_sync_map(self.file3b)
        self.assertEqual(map, self.map3B)
        with tempfile.TemporaryDirectory() as tdir:
            self.assertIsNone(spikeglx.get_sync_map(Path(tdir) / 'idontexist.json'))


class TestsMisc(unittest.TestCase):

    def setUp(self):
        self.tempdir = Path(tempfile.TemporaryDirectory().name)
        self.tempdir.mkdir()
        self.subdirs = [
            self.tempdir / 'test_empty_parent',
            self.tempdir / 'test_empty_parent' / 'test_empty',
            self.tempdir / 'test_empty',
            self.tempdir / 'test_full',
        ]
        self.file = self.tempdir / 'test_full' / 'file.txt'

        _ = [x.mkdir() for x in self.subdirs]
        self.file.touch()

    def _resetup_folders(self):
        self.file.unlink()
        (self.tempdir / 'test_full').rmdir()
        _ = [x.rmdir() for x in self.subdirs if x.exists()]
        _ = [x.mkdir() for x in self.subdirs]
        self.file.touch()

    def test_delete_empty_folders(self):
        pre = [x.exists() for x in self.subdirs]
        pre_expected = [True, True, True, True]
        self.assertTrue(all([x == y for x, y in zip(pre, pre_expected)]))

        # Test dry run
        pos_expected = None
        pos = misc.delete_empty_folders(self.tempdir)
        self.assertTrue(pos == pos_expected)
        # Test dry=False, non recursive
        pos_expected = [True, False, False, True]
        misc.delete_empty_folders(self.tempdir, dry=False)
        pos = [x.exists() for x in self.subdirs]
        self.assertTrue(all([x == y for x, y in zip(pos, pos_expected)]))

        self._resetup_folders()

        # Test recursive
        pos_expected = [False, False, False, True]
        misc.delete_empty_folders(self.tempdir, dry=False, recursive=True)
        pos = [x.exists() for x in self.subdirs]
        self.assertTrue(all([x == y for x, y in zip(pos, pos_expected)]))

    def tearDown(self):
        self.file.unlink()
        (self.tempdir / 'test_full').rmdir()
        _ = [x.rmdir() for x in self.subdirs if x.exists()]


if __name__ == "__main__":
    unittest.main(exit=False)

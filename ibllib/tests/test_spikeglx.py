from pathlib import Path
import os
import shutil
import tempfile
import unittest

import numpy as np

from ibllib.io import spikeglx, hashfile


class TestSpikeGLX_hardwareInfo(unittest.TestCase):

    def setUp(self) -> None:
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
                      'bpod': 16,
                      'laser': 17,
                      'laser_ttl': 18}
        self.file3a = self.workdir / 'sample3A_g0_t0.imec.wiring.json'
        self.file3b = self.workdir / 'sample3B_g0_t0.nidq.wiring.json'

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

    def test_default_values(self):
        from ibllib.io.extractors import ephys_fpga
        self.assertEqual(ephys_fpga.CHMAPS['3A']['ap'], self.map3A)
        self.assertEqual(ephys_fpga.CHMAPS['3B']['nidq'], self.map3B)


class TestSpikeGLX_glob_ephys(unittest.TestCase):
    """
    Creates mock acquisition folders architecture (omitting metadata files):
    ├── 3A
    │   ├── imec0
    │   │   ├── sync_testing_g0_t0.imec0.ap.bin
    │   │   └── sync_testing_g0_t0.imec0.lf.bin
    │   └── imec1
    │       ├── sync_testing_g0_t0.imec1.ap.bin
    │       └── sync_testing_g0_t0.imec1.lf.bin
    └── 3B
        ├── sync_testing_g0_t0.nidq.bin
        ├── imec0
        │   ├── sync_testing_g0_t0.imec0.ap.bin
        │   └── sync_testing_g0_t0.imec0.lf.bin
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
        self.dict3b_ch = [{'label': 'imec0',
                           'ap': self.dir3b / 'imec0' / 'sync_testing_g0_t0.imec0.ap.ch',
                           'lf': self.dir3b / 'imec0' / 'sync_testing_g0_t0.imec0.lf.ch',
                           'path': self.dir3b / 'imec0'},
                          {'label': 'imec1',
                           'ap': self.dir3b / 'imec1' / 'sync_testing_g0_t0.imec1.ap.ch',
                           'lf': self.dir3b / 'imec1' / 'sync_testing_g0_t0.imec1.lf.ch',
                           'path': self.dir3b / 'imec1'},
                          {'label': '',
                           'nidq': self.dir3b / 'sync_testing_g0_t0.nidq.ch',
                           'path': self.dir3b}]
        create_tree(self.dir3a, self.dict3a)
        create_tree(self.dir3b, self.dict3b)
        create_tree(self.dir3b, self.dict3b_ch)

    def test_glob_ephys(self):
        def dict_equals(d1, d2):
            return all([x in d1 for x in d2]) and all([x in d2 for x in d1])
        ef3b = spikeglx.glob_ephys_files(self.dir3b)
        ef3a = spikeglx.glob_ephys_files(self.dir3a)
        ef3b_ch = spikeglx.glob_ephys_files(self.dir3b, ext='ch')
        # test glob
        self.assertTrue(dict_equals(self.dict3a, ef3a))
        self.assertTrue(dict_equals(self.dict3b, ef3b))
        self.assertTrue(dict_equals(self.dict3b_ch, ef3b_ch))
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
        # self.addClassCleanup(self._tempdir.cleanup)  # py3.8
        self.workdir = Path(self._tempdir.name)
        file_meta = Path(__file__).parent.joinpath('fixtures', 'io', 'spikeglx',
                                                   'sample3A_short_g0_t0.imec.ap.meta')
        self.file_bin = spikeglx._mock_spikeglx_file(
            self.workdir.joinpath('sample3A_short_g0_t0.imec.ap.bin'), file_meta, ns=76104,
            nc=385, sync_depth=16, random=True)['bin_file']
        self.sr = spikeglx.Reader(self.file_bin)

    def tearDown(self):
        self._tempdir.cleanup

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

    @unittest.skipIf(os.name == 'nt', 'SpikeGLX compression fails on Windows')
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

    def test_fix_meta_file(self):
        # test the case where the meta file shows a larger amount of samples
        with tempfile.TemporaryDirectory(prefix='glx_test') as tdir:
            bin_3a = spikeglx._mock_spikeglx_file(
                Path(tdir).joinpath('sample3A_g0_t0.imec.ap.bin'),
                self.workdir / 'sample3A_g0_t0.imec.ap.meta', ns=32, nc=385, sync_depth=16)
            with open(bin_3a['bin_file'], 'wb') as fp:
                np.random.randint(-20000, 20000, 385 * 22, dtype=np.int16).tofile(fp)
            sr = spikeglx.Reader(bin_3a['bin_file'])
            assert sr.meta['fileTimeSecs'] * 30000 == 22

    def test_read_corrupt(self):
        # nidq has 1 analog and 1 digital sync channels
        with tempfile.TemporaryDirectory(prefix='glx_test') as tdir:
            int2volts = 5 / 32768
            nidq = spikeglx._mock_spikeglx_file(
                Path(tdir).joinpath('sample3B_g0_t0.nidq.bin'),
                self.workdir / 'sample3B_g0_t0.nidq.meta',
                ns=32, nc=2, sync_depth=8, int2volts=int2volts, corrupt=True)
            self.assert_read_glx(nidq)

    def test_read_nidq(self):
        # nidq has 1 analog and 1 digital sync channels
        with tempfile.TemporaryDirectory(prefix='glx_test') as tdir:
            int2volts = 5 / 32768
            nidq = spikeglx._mock_spikeglx_file(
                Path(tdir).joinpath('sample3B_g0_t0.nidq.bin'),
                self.workdir / 'sample3B_g0_t0.nidq.meta',
                ns=32, nc=2, sync_depth=8, int2volts=int2volts)
            self.assert_read_glx(nidq)

    def test_read_3A(self):
        with tempfile.TemporaryDirectory(prefix='glx_test') as tdir:
            bin_3a = spikeglx._mock_spikeglx_file(
                Path(tdir).joinpath('sample3A_g0_t0.imec.ap.bin'),
                self.workdir / 'sample3A_g0_t0.imec.ap.meta',
                ns=32, nc=385, sync_depth=16)
            self.assert_read_glx(bin_3a)

    def test_read_3B(self):
        with tempfile.TemporaryDirectory(prefix='glx_test') as tdir:
            bin_3b = spikeglx._mock_spikeglx_file(
                Path(tdir).joinpath('sample3B_g0_t0.imec1.ap.bin'),
                self.workdir / 'sample3B_g0_t0.imec1.ap.meta',
                ns=32, nc=385, sync_depth=16)
            self.assert_read_glx(bin_3b)

    def test_check_ephys_file(self):
        self.tdir = tempfile.TemporaryDirectory(prefix='glx_test')
        self.addCleanup(self.tdir.cleanup)
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

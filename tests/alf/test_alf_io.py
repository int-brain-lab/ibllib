import unittest
import tempfile
from pathlib import Path
import shutil
import json
import uuid

import numpy as np

import alf.io


class TestAlfBunch(unittest.TestCase):

    def test_to_dataframe_scalars(self):
        simple = alf.io.AlfBunch({'titi': np.random.rand(500),
                                  'toto': np.random.rand(500)})
        df = simple.to_df()
        self.assertTrue(np.all(df['titi'].values == simple.titi))
        self.assertTrue(np.all(df['toto'].values == simple.toto))
        self.assertTrue(len(df.columns) == 2)

    def test_to_dataframe_vectors(self):
        vectors = alf.io.AlfBunch({'titi': np.random.rand(500, 1),
                                   'toto': np.random.rand(500),
                                   'tata': np.random.rand(500, 2)})
        df = vectors.to_df()
        self.assertTrue(np.all(df['titi'].values == vectors.titi[:, 0]))
        self.assertTrue(np.all(df['toto'].values == vectors.toto))
        self.assertTrue(np.all(df['tata_0'].values == vectors.tata[:, 0]))
        self.assertTrue(np.all(df['tata_1'].values == vectors.tata[:, 1]))
        self.assertTrue(len(df.columns) == 4)

    def test_append_numpy(self):
        a = alf.io.AlfBunch({'titi': np.random.rand(500),
                             'toto': np.random.rand(500)})
        b = alf.io.AlfBunch({})
        # test with empty elements
        self.assertTrue(np.all(np.equal(a.append({})['titi'], a['titi'])))
        self.assertTrue(np.all(np.equal(b.append(a)['titi'], a['titi'])))
        self.assertEqual(b.append({}), {})
        # test with numpy arrays
        b = alf.io.AlfBunch({'titi': np.random.rand(250),
                             'toto': np.random.rand(250)})
        c = a.append(b)
        t = np.all(np.equal(c['titi'][0:500], a['titi']))
        t &= np.all(np.equal(c['toto'][0:500], a['toto']))
        t &= np.all(np.equal(c['titi'][500:], b['titi']))
        t &= np.all(np.equal(c['toto'][500:], b['toto']))
        self.assertTrue(t)
        a.append(b, inplace=True)
        self.assertTrue(np.all(np.equal(c['toto'], a['toto'])))
        self.assertTrue(np.all(np.equal(c['titi'], a['titi'])))

    def test_append_list(self):
        # test with lists
        a = alf.io.AlfBunch({'titi': [0, 1, 3],
                             'toto': ['a', 'b', 'c']})
        b = alf.io.AlfBunch({'titi': [1, 2, 4],
                             'toto': ['d', 'e', 'f']})
        c = a.append(b)
        self.assertTrue(len(c['toto']) == 6)
        self.assertTrue(len(a['toto']) == 3)
        c = c.append(b)
        self.assertTrue(len(c['toto']) == 9)
        self.assertTrue(len(a['toto']) == 3)
        c.append(b, inplace=True)
        self.assertTrue(len(c['toto']) == 12)
        self.assertTrue(len(a['toto']) == 3)


class TestsAlfPartsFilters(unittest.TestCase):

    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.gettempdir()) / 'iotest'
        self.tmpdir.mkdir(exist_ok=True)

    def test_npy_parts_and_file_filters(self):
        a = {'riri': np.random.rand(100),
             'fifi': np.random.rand(100)}
        alf.io.save_object_npy(self.tmpdir, a, 'neuveux', parts='tutu')
        alf.io.save_object_npy(self.tmpdir, a, 'neuveux', parts='toto')
        alf.io.save_object_npy(self.tmpdir, a, 'neuveux', parts=['tutu', 'titi'])
        b = alf.io.load_object(self.tmpdir, 'neuveux')
        for k in a:
            self.assertTrue(np.all(a[k] == b[k + '.tutu.titi']))
            self.assertTrue(np.all(a[k] == b[k + '.tutu']))
            self.assertTrue(np.all(a[k] == b[k + '.toto']))
        # also test file filters through glob argument
        self.assertTrue(alf.io.exists(self.tmpdir, 'neuveux', glob='*.toto.*'))
        c = alf.io.load_object(self.tmpdir, 'neuveux', glob='*.toto.*')
        self.assertEqual(set(c.keys()), set([k for k in b.keys() if k.endswith('toto')]))
        # test with the short keys
        a = alf.io.load_object(self.tmpdir, 'neuveux', glob=['titi'])
        self.assertTrue(set(a.keys()) == set(['riri.tutu.titi', 'fifi.tutu.titi']))
        a = alf.io.load_object(self.tmpdir, 'neuveux', glob=['titi'], short_keys=True)
        self.assertTrue(set(a.keys()) == set(['riri', 'fifi']))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


class TestsAlf(unittest.TestCase):
    def setUp(self) -> None:
        # riri, fifi and loulou are huey, duey and louie in French (Donald nephews for ignorants)
        self.tmpdir = Path(tempfile.gettempdir()) / 'iotest'
        self.tmpdir.mkdir(exist_ok=True)
        self.vfile = self.tmpdir / 'toto.titi.npy'
        self.tfile = self.tmpdir / 'toto.timestamps.npy'
        self.object_files = [self.tmpdir / 'neuveu.riri.npy',
                             self.tmpdir / 'neuveu.fifi.npy',
                             self.tmpdir / 'neuveu.loulou.npy',
                             self.tmpdir / 'object.attribute.part1.part2.npy',
                             self.tmpdir / 'object.attribute.part1.npy']
        for f in self.object_files:
            np.save(file=f, arr=np.random.rand(5,))

    def test_exists(self):
        self.assertFalse(alf.io.exists(self.tmpdir, 'asodiujfas'))
        self.assertTrue(alf.io.exists(self.tmpdir, 'neuveu'))
        # test with attribute string only
        self.assertTrue(alf.io.exists(self.tmpdir, 'neuveu', attributes='riri'))
        # test with list of attributes
        self.assertTrue(alf.io.exists(self.tmpdir, 'neuveu', attributes=['riri', 'fifi']))
        self.assertFalse(alf.io.exists(self.tmpdir, 'neuveu', attributes=['riri', 'fifiasdf']))
        # test with globing
        self.assertTrue(alf.io.exists(self.tmpdir, 'object', glob='*part2*'))
        self.assertTrue(alf.io.exists(self.tmpdir, 'object', glob=['part1', 'part2']))
        # globing with list: an empty part should return true as well
        self.assertTrue(alf.io.exists(self.tmpdir, 'object', glob=['']))

    def test_metadata_columns(self):
        # simple test with meta data to label columns
        file_alf = self.tmpdir / '_ns_object.attribute.npy'
        data = np.random.rand(500, 4)
        cols = ['titi', 'tutu', 'toto', 'tata']
        np.save(file_alf, data)
        np.save(self.tmpdir / '_ns_object.gnagna.npy', data[:, -1])
        alf.io.save_metadata(file_alf, {'columns': cols})
        dread = alf.io.load_object(self.tmpdir, '_ns_object')
        self.assertTrue(np.all(dread['titi'] == data[:, 0]))
        self.assertTrue(np.all(dread['gnagna'] == data[:, -1]))
        # add another field to the metadata
        alf.io.save_metadata(file_alf, {'columns': cols, 'unit': 'potato'})
        dread = alf.io.load_object(self.tmpdir, '_ns_object')
        self.assertTrue(np.all(dread['titi'] == data[:, 0]))
        self.assertTrue(dread['attributemetadata']['unit'] == 'potato')
        self.assertTrue(np.all(dread['gnagna'] == data[:, -1]))

    def test_metadata_columns_UUID(self):
        data = np.random.rand(500, 4)
        # test with UUID extra field
        file_alf = self.tmpdir / '_ns_obj.attr1.2622b17c-9408-4910-99cb-abf16d9225b9.npy'
        file_meta = self.tmpdir / '_ns_obj.attr1.metadata.bd66f60e-fefc-4d92-b2c3-daaeee6c83af.npy'
        np.save(file_alf, data)
        cols = ['titi', 'tutu', 'toto', 'tata']
        file_meta = file_alf.parent / (file_alf.stem + '.metadata.json')
        with open(file_meta, 'w+') as fid:
            fid.write(json.dumps({'columns': cols}, indent=1))
        dread = alf.io.load_object(self.tmpdir, '_ns_obj')
        self.assertTrue(np.all(dread['titi'] == data[:, 0]))

    def test_read_ts(self):
        # simplest test possible with one column in each file
        t = np.arange(0, 10)
        d = np.random.rand(10)
        np.save(self.vfile, d)
        np.save(self.tfile, t)
        t_, d_ = alf.io.read_ts(self.vfile)
        self.assertTrue(np.all(t_ == t))
        self.assertTrue(np.all(d_ == d))

    def test_load_object(self):
        # first usage of load object is to provide one of the files belonging to the object
        obj = alf.io.load_object(self.object_files[0])
        self.assertTrue(set(obj.keys()) == set(['riri', 'fifi', 'loulou']))
        self.assertTrue(all([obj[o].shape == (5,) for o in obj]))
        # the second usage is to provide a directory and the object name
        obj = alf.io.load_object(self.tmpdir, 'neuveu')
        self.assertTrue(set(obj.keys()) == set(['riri', 'fifi', 'loulou']))
        self.assertTrue(all([obj[o].shape == (5,) for o in obj]))
        # and this should throw a value error
        with self.assertRaises(ValueError) as context:
            obj = alf.io.load_object(self.tmpdir)
        self.assertTrue('object name should be provided too' in str(context.exception))

    def test_save_npy(self):
        # test with straight vectors
        a = {'riri': np.random.rand(100),
             'fifi': np.random.rand(100)}
        alf.io.save_object_npy(self.tmpdir, a, 'neuveux')
        # read after write
        b = alf.io.load_object(self.tmpdir, 'neuveux')
        for k in a:
            self.assertTrue(np.all(a[k] == b[k]))
        # test with more exotic shapes, still valid
        a = {'riri': np.random.rand(100),
             'fifi': np.random.rand(100, 2),
             'loulou': np.random.rand(1, 2)}
        alf.io.save_object_npy(self.tmpdir, a, 'neuveux')
        # read after write
        b = alf.io.load_object(self.tmpdir, 'neuveux')
        for k in a:
            self.assertTrue(np.all(a[k] == b[k]))
        # test with non allowed shape
        a = {'riri': np.random.rand(100),
             'fifi': np.random.rand(100, 2),
             'loulou': np.random.rand(5, 2)}
        with self.assertRaises(Exception) as context:
            alf.io.save_object_npy(self.tmpdir, a, 'neuveux')
        self.assertTrue('Dimensions are not consistent' in str(context.exception))

    def test_check_dimensions(self):
        a = {'a': np.ones([10, 10]), 'b': np.ones([10, 2]), 'c': np.ones([10])}
        status = alf.io.check_dimensions(a)
        self.assertTrue(status == 0)
        a = {'a': np.ones([10, 10]), 'b': np.ones([10, 1]), 'c': np.ones([10])}
        status = alf.io.check_dimensions(a)
        self.assertTrue(status == 0)
        a = {'a': np.ones([10, 15]), 'b': np.ones([1, 15]), 'c': np.ones([10])}
        status = alf.io.check_dimensions(a)
        self.assertTrue(status == 0)
        a = {'a': np.ones([9, 10]), 'b': np.ones([10, 1]), 'c': np.ones([10])}
        status = alf.io.check_dimensions(a)
        self.assertTrue(status == 1)
        # test for timestamps which is an exception to the rule
        a = {'a': np.ones([10, 15]), 'b': np.ones([1, 15]), 'c': np.ones([10])}
        a['timestamps'] = np.ones([3, 1])
        a['timestamps.titi'] = np.ones([3, 1])
        status = alf.io.check_dimensions(a)
        self.assertTrue(status == 0)
        # gracefully exit if the dictionary only contains excepted attributes
        status = alf.io.check_dimensions({'timestamps': None})
        self.assertTrue(status == 0)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


class TestSessionFolder(unittest.TestCase):

    def test_isdatetime(self):
        inp = ['açsldfkça', '12312', 34, '2020-01-01', '01-01-2020']
        out = [False, False, False, True, False]
        for i, o in zip(inp, out):
            self.assertEqual(o, alf.io._isdatetime(i))

    def test_get_session_folder(self):
        inp = (Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001/raw_behavior_data/'
                    '_iblrig_micData.raw.wav'),
               Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),
               '/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001/raw_behavior_data'
               '/_iblrig_micData.raw.wav',
               '/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001',)
        out = (Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),
               Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),
               Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),
               Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'),)
        for i, o in zip(inp, out):
            self.assertEqual(o, alf.io.get_session_path(i))
        # Test if None is passed
        no_out = alf.io.get_session_path(None)
        self.assertTrue(no_out is None)

    def test_get_session_folder_regex(self):
        o = alf.io._regexp_session_path(r'C:\titi\toto\ZM_1368/2019-04-19/001', '\\')
        self.assertIsNotNone(o)
        o = alf.io._regexp_session_path(Path('/mnt/s0/Data/Subjects/ZM_1368/2019-04-19/001'), '\\')
        self.assertIsNotNone(o)
        o = alf.io._regexp_session_path(Path('/mnt/s0/Data/Subjects/ZM_1368/2019/1'), '\\')
        self.assertIsNone(o)

    def test_is_session_folder(self):
        inp = [(Path('/mnt/s0/Data/Subjects/ibl_witten_14/2019-12-04'), False),
               ('/mnt/s0/Data/Subjects/ibl_witten_14/2019-12-04', False),
               (Path('/mnt/s0/Data/Subjects/ibl_witten_14/2019-12-04/001'), True),
               (Path('/mnt/s0/Data/Subjects/ibl_witten_14/2019-12-04/001/tutu'), False),
               ('/mnt/s0/Data/Subjects/ibl_witten_14/2019-12-04/001/', True)]
        for i in inp:
            self.assertEqual(alf.io.is_session_path(i[0]), i[1])

    def test_is_uuid_string(self):
        testins = [
            None,
            'some_string',
            'f6ffe25827-06-425aaa-f5-919f70025835',
            'f6ffe258-2706-425a-aaf5-919f70025835']
        expected = [False, False, False, True]
        for i, e in zip(testins, expected):
            self.assertTrue(alf.io.is_uuid_string(i) == e)

    def test_is_details_dict(self):
        keys = [
            'subject',
            'start_time',
            'number',
            'lab',
            'project',
            'url',
            'task_protocol',
            'local_path'
        ]
        testins = [
            None,
            dict.fromkeys(keys[1:]),
            dict.fromkeys(keys),
        ]
        expected = [False, False, True]
        for i, e in zip(testins, expected):
            self.assertTrue(alf.io.is_details_dict(i) == e)


class TestUUID_Files(unittest.TestCase):

    def test_remove_uuid(self):
        with tempfile.TemporaryDirectory() as dir:
            f1 = Path(dir).joinpath('tutu.part1.part1.30c09473-4d3d-4f51-9910-c89a6840096e.json')
            f2 = Path(dir).joinpath('tata.part1.part1.json')
            f3 = Path(dir).joinpath('toto.json')
            f1.touch()
            f2.touch()
            f2.touch()
            self.assertTrue(alf.io.remove_uuid_file(f1) ==
                            Path(dir).joinpath('tutu.part1.part1.json'))
            self.assertTrue(alf.io.remove_uuid_file(f2) ==
                            Path(dir).joinpath('tata.part1.part1.json'))
            self.assertTrue(alf.io.remove_uuid_file(f3) ==
                            Path(dir).joinpath('toto.json'))
            self.assertTrue(alf.io.remove_uuid_file(str(f3)) ==
                            Path(dir).joinpath('toto.json'))

    def test_add_uuid(self):
        _uuid = uuid.uuid4()

        file_with_uuid = f'/titi/tutu.part1.part1.{_uuid}.json'
        inout = [
            (file_with_uuid, Path(file_with_uuid)),
            ('/tutu/tata.json', Path(f'/tutu/tata.{_uuid}.json')),
            ('/tutu/tata.part1.json', Path(f'/tutu/tata.part1.{_uuid}.json')),
        ]
        for tup in inout:
            self.assertEqual(tup[1], alf.io.add_uuid_string(tup[0], _uuid))
            self.assertEqual(tup[1], alf.io.add_uuid_string(tup[0], str(_uuid)))


if __name__ == "__main__":
    unittest.main(exit=False)

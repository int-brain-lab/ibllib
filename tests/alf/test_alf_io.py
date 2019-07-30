import unittest
import tempfile
from pathlib import Path
import shutil
import json

import numpy as np

import alf.io


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
        self.assertTrue(status == 1)
        a = {'a': np.ones([10, 10]), 'b': np.ones([10, 1]), 'c': np.ones([10])}
        status = alf.io.check_dimensions(a)
        self.assertTrue(status == 0)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


if __name__ == "__main__":
    unittest.main(exit=False)

import unittest
import os
import uuid
import tempfile
from pathlib import Path
import shutil

import numpy as np

from ibllib.io import params, flags, jsonable, spikeglx, alf


class TestsParams(unittest.TestCase):

    def setUp(self):
        self.par_dict = {'A': 'tata',
                         'O': 'toto',
                         'I': 'titi',
                         'num': 15,
                         'liste': [1, 'turlu']}
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
        self.tempfile = tempfile.NamedTemporaryFile()

    def testFlagFileRead(self):
        # empty file should return True
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)
        # test with 2 lines and a trailing
        with open(self.tempfile.name, 'w+') as fid:
            fid.write('file1\nfile2\n')
        self.assertEqual(flags.read_flag_file(self.tempfile.name), ['file1', 'file2'])
        # test with 2 lines and a trailing, Windows convention
        with open(self.tempfile.name, 'w+') as fid:
            fid.write('file1\r\nfile2\r\n')
        self.assertEqual(flags.read_flag_file(self.tempfile.name), ['file1', 'file2'])

    def testAppendFlagFile(self):
        #  DO NOT CHANGE THE ORDER OF TESTS BELOW
        # prepare a file with 3 dataset types
        file_list = ['_ibl_extraRewards.times', '_ibl_lickPiezo.raw', '_ibl_lickPiezo.timestamps']
        with open(self.tempfile.name, 'w+') as fid:
            fid.write('\n'.join(file_list))
        self.assertEqual(flags.read_flag_file(self.tempfile.name), file_list)

        # with an existing file containing files, writing more files append to it
        file_list_2 = ['turltu']
        # also makes sure that if a string is provided it works
        flags.write_flag_file(self.tempfile.name, file_list_2[0])
        self.assertEqual(flags.read_flag_file(self.tempfile.name), file_list + file_list_2)

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
        self.assertEqual(flags.read_flag_file(self.tempfile.name), ['file1', 'file2', 'file3'])

        # test the removal of a file within the list
        flags.excise_flag_file(self.tempfile.name, removed_files='file1')
        self.assertEqual(sorted(flags.read_flag_file(self.tempfile.name)), ['file2', 'file3'])

        # if file-list is True it means all files and file_list should be empty after read
        flags.write_flag_file(self.tempfile.name, file_list=True)
        self.assertEqual(flags.read_flag_file(self.tempfile.name), True)

    def tearDown(self):
        self.tempfile.close()


class TestsJsonable(unittest.TestCase):

    def testReadWrite(self):
        tfile = tempfile.NamedTemporaryFile()
        data = [{'a': 'thisisa', 'b': 1, 'c': [1, 2, 3]},
                {'a': 'thisisb', 'b': 2, 'c': [2, 3, 4]}]
        jsonable.write(tfile.name, data)
        data2 = jsonable.read(tfile.name)
        self.assertEqual(data, data2)
        jsonable.append(tfile.name, data)
        data3 = jsonable.read(tfile.name)
        self.assertEqual(data + data, data3)
        tfile.close()


class TestsSpikeGLX(unittest.TestCase):
    def setUp(self):
        self.workdir = Path(__file__).parent / 'fixtures' / 'io' / 'spikeglx'

    def testReadMetaData(self):
        meta_data_file = self.workdir / 'FC034_g0_t0.imec.lf.meta'
        md = spikeglx.read_meta_data(meta_data_file)
        self.assertTrue(len(md.keys()) == 37)

    def testReadChannelGain(self):
        meta_data_file = self.workdir / 'FC034_g0_t0.imec.lf.meta'
        md = spikeglx.read_meta_data(meta_data_file)
        cg = spikeglx._gain_channels(md)
        self.assertTrue(np.all(cg['lf'][0:-1] == 250))
        self.assertTrue(np.all(cg['ap'][0:-1] == 500))
        self.assertTrue(len(cg['ap']) == len(cg['lf']) == int(sum(md.get('snsApLfSy'))))


class TestsAlf(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.gettempdir()) / 'iotest'
        self.tmpdir.mkdir()
        self.vfile = self.tmpdir / 'toto.titi.npy'
        self.tfile = self.tmpdir / 'toto.timestamps.npy'
        self.object_files = [self.tmpdir / 'neuveu.riri.npy',
                             self.tmpdir / 'neuveu.fifi.npy',
                             self.tmpdir / 'neuveu.loulou.npy']
        for f in self.object_files:
            np.save(file=f, arr=np.random.rand(5,))

    def test_read_ts(self):
        # simplest test possible with one column in each file
        t = np.arange(0, 10)
        d = np.random.rand(10)
        np.save(self.vfile, d)
        np.save(self.tfile, t)
        t_, d_ = alf.read_ts(self.vfile)
        self.assertTrue(np.all(t_ == t))
        self.assertTrue(np.all(d_ == d))

    def test_load_object(self):
        # first usage of load object is to provide one of the files belonging to the object
        obj = alf.load_object(self.object_files[0])
        self.assertTrue(set(obj.keys()) == set(['riri', 'fifi', 'loulou']))
        self.assertTrue(all([obj[o].shape == (5,) for o in obj]))
        # the second usage is to provide a directory and the object name
        obj = alf.load_object(self.tmpdir, 'neuveu')
        self.assertTrue(set(obj.keys()) == set(['riri', 'fifi', 'loulou']))
        self.assertTrue(all([obj[o].shape == (5,) for o in obj]))
        # and this should throw a value error
        with self.assertRaises(ValueError) as context:
            obj = alf.load_object(self.tmpdir)
        self.assertTrue('object name should be provided too' in str(context.exception))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir)


if __name__ == "__main__":
    unittest.main(exit=False)

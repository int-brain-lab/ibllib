import ibllib.io.params as params
import unittest
import os


class TestUtils(unittest.TestCase):

    def test_params(self):
        #  first go to and from dictionary
        par_dict = {'A': 'tata',
                    'O': 'toto',
                    'I': 'titi'}
        par = params.from_dict(par_dict)
        self.assertEqual(params.as_dict(par), par_dict)
        # next go to and from dictionary via json
        params.write('toto', par)
        par2 = params.read('toto')
        self.assertEqual(par, par2)
        # at last delete the param file
        os.remove(params.getfile('toto'))

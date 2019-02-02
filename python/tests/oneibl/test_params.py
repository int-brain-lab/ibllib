import unittest
import shutil
from pathlib import Path

import oneibl.params as params
from ibllib.io import params as iopar


class TestONEParams(unittest.TestCase):
    def setUp(self):
        self.par_current = params.default()
        self.par_default = params.default()
        self.existing_params = Path(iopar.getfile(params._PAR_ID_STR))
        self.bk_params = self.existing_params.parent.joinpath('.one_params_bk')
        if self.existing_params.exists():
            shutil.copy(self.existing_params, self.bk_params)
            self.existing_params.unlink()

    def test__get_current_par(self):
        for k in iopar.as_dict(self.par_current):
            self.assertTrue(
                params._get_current_par(
                    k, self.par_current) == self.par_default.as_dict()[k])

    def test_setup_silent(self):
        self.assertIsNone(iopar.read(params._PAR_ID_STR))
        params.setup_silent()
        par = iopar.read(params._PAR_ID_STR)
        self.assertIsNotNone(par)
        # now do another test to see if it preserves current values
        par = par.as_dict()
        par['ALYX_LOGIN'] = 'oijkcjioifqer'
        iopar.write(params._PAR_ID_STR, par)
        params.setup_silent()
        par2 = iopar.read(params._PAR_ID_STR)
        self.assertEqual(par, par2.as_dict())

    def tearDown(self):
        if self.bk_params.exists():
            shutil.copy(self.bk_params, self.existing_params)
            self.bk_params.unlink()


if __name__ == "__main__":
    unittest.main(exit=False)

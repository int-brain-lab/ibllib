import unittest


class TestIBLRigImports(unittest.TestCase):

    def setUp(self):
        pass

    def test_iblrig_imports(self):
        # List of all import statements in iblrig on dev 20200609
        import ibllib.graphic as graph
        import ibllib.io.flags as flags
        import ibllib.io.params as lib_params
        import ibllib.io.raw_data_loaders as raw
        import ibllib.pipes.misc as misc
        import oneibl.params
        from ibllib.dsp.smooth import rolling_window as smooth
        from ibllib.graphic import numinput, popup, strinput
        from ibllib.io import raw_data_loaders as raw
        from ibllib.misc import logger_config
        from ibllib.pipes.experimental_data import (compress_video, create, extract,
                                                    register)
        from ibllib.pipes.purge_rig_data import purge_local_data
        from ibllib.pipes.transfer_rig_data import main
        from oneibl.one import ONE

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main(exit=False)

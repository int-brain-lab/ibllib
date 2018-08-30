# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, July 26th 2018, 4:22:08 pm
# @Last Modified by:   Niccolò Bonacchi
# @Last Modified time: 2018-07-26 18:02:27
import unittest


class TestImports(unittest.TestCase):
    def test_all_imports(self):
        import ibllib
        import ibllib as ibl
        from ibllib import misc, io, dsp, webclient, time
        import ibllib.dsp as dsp
        import ibllib.io as io
        import ibllib.misc as misc
        import ibllib.time
        import ibllib.webclient
        import ibllib.io.raw_data_loaders as raw
        from ibllib.misc import pprint, flatten, timing, is_uuid_string
        from ibllib.io import raw_data_loaders
        from ibllib.io.raw_data_loaders import (load_data, load_settings,
                                                load_encoder_events,
                                                load_encoder_positions,
                                                load_encoder_trial_info)
        from ibllib.dsp import savitzky_golay, smooth, smooth_demo


if __name__ == '__main__':
    ti = TestImports()
    ti.test_all_imports()
    print(dir())
    print("Done!")

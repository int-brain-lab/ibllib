#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Friday, October 30th 2020, 10:42:49 am
import unittest

import ibllib.io.extractors.ephys_passive as passive


class TestsPassiveExtractor(unittest.TestCase):
    def setUp(self):
        pass

    def test_load_passive_stim_meta(self):
        meta = passive._load_passive_stim_meta()
        self.assertTrue(isinstance(meta, dict))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main(exit=False, verbosity=2)

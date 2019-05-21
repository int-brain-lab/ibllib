#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, July 26th 2018, 4:22:08 pm
import unittest


class TestImports(unittest.TestCase):
    def test_all_imports(self):
        pass


if __name__ == '__main__':
    ti = TestImports()
    ti.test_all_imports()
    print(dir())
    print("Done!")

# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, July 26th 2018, 4:22:08 pm
# @Last Modified by:   Niccolò Bonacchi
# @Last Modified time: 2018-07-26 18:02:27
import unittest


class TestImports(unittest.TestCase):
    def test_all_imports(self):
        pass


if __name__ == '__main__':
    ti = TestImports()
    ti.test_all_imports()
    print(dir())
    print("Done!")

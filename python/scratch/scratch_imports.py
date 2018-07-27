# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, July 26th 2018, 4:22:08 pm
# @Last Modified by:   Niccolò Bonacchi
# @Last Modified time: 2018-07-26 18:02:27
import sys
sys.path.insert(0, '/home/nico/Projects/IBL/IBL-github/ibllib/python/')
from ibllib.misc import pprint
import ibllib as ibl
import ibllib.misc as misc


if __name__ == '__main__':
    print(ibl)
    print(ibl.misc)
    print(ibl.misc.pprint)
    print(ibl.misc.is_uuid_string)
    print(ibl.misc.timing)
    print(ibl.misc.flatten)
    print("Done!")

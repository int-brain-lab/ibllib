#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Wednesday, July 25th 2018, 3:22:57 pm
import time


def timing(f):
    """
    Timing decorator will print time took in milliseconds
    """
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print('{} function elapsed time = {} ms'.format(f, (time2 - time1) *
                                                        1000.0))
        return ret

    return wrap


if __name__ == '__main__':
    @timing
    def func(x, y):
        return x**y

    func(2, 3)
    print("Done!")

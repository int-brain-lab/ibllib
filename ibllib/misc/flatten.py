#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Wednesday, July 25th 2018, 3:56:27 pm
"""
Flatten a nested Iterable excluding strings and dicts.

Converts nested Iterable into flat list. Will not iterate through strings or
dicts.

:return: Flattened list or generator object.
:rtype: list or generator
"""
import collections


def iflatten(x):
    result = []
    for el in x:
        if isinstance(el, collections.abc.Iterable) and not (
                isinstance(el, str) or isinstance(el, dict)):
            result.extend(iflatten(el))
        else:
            result.append(el)
    return result


def gflatten(x):
    def iselement(e):
        return not(isinstance(e, collections.abc.Iterable) and not(
            isinstance(el, str) or isinstance(el, dict)))
    for el in x:
        if iselement(el):
            yield el
        else:
            yield from gflatten(el)


def flatten(x, generator=False):
    if generator:
        return gflatten(x)
    return iflatten(x)


if __name__ == '__main__':
    x = (1, 2, 3, [1, 2], 'string', 0.1, {1: None}, [[1, 2, 3], {1: 1}, 1])
    print(flatten(x))
    assert iflatten(x) == flatten(x)
    assert list(gflatten(x)) == list(flatten(x, generator=True))

    print("Done!")

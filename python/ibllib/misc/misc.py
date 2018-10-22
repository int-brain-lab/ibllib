# library of small functions
import numpy as np
import json
import re


def pprint(my_dict):
    print(json.dumps(my_dict, indent=4))


def is_uuid_string(string):
    if string is None:
        return False
    if len(string) != 36:
        return False
    UUID_PATTERN = re.compile(r'^[\da-f]{8}-([\da-f]{4}-){3}[\da-f]{12}$', re.IGNORECASE)
    if UUID_PATTERN.match(string):
        return True
    else:
        return False


def structarr(names, shape=None, formats=None):
    if not formats:
        formats = ['f8'] * len(names)
    dtyp = np.dtype({'names': names, 'formats': formats})
    return np.zeros(shape, dtype=dtyp)


if __name__ == '__main__':
    names = ['positions', 'times']
    x = structarr(names, shape=(1500,))

    x['positions'] = np.random.random(1500,)
    x['times'] = np.arange(1500,)

    x[1:10]['times']
    b = x[1:10]

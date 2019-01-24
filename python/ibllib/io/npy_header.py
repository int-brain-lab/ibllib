from collections import namedtuple
import ast


def read(filename):
    header = namedtuple('npy_header',
                        'magic_string, version, header_len, descr, fortran_order, shape')

    with open(filename, 'rb') as fid:
        header.magic_string = fid.read(6)
        header.version = fid.read(2)
        header.header_len = int.from_bytes(fid.read(2), byteorder='little')
        d = ast.literal_eval(fid.read(header.header_len).decode())

    for k in d.keys():
        print(k)
        setattr(header, k, d[k])

    return header

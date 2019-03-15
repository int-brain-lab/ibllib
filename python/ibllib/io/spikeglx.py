import re


def read_metadata(md_file):
    """
    Reads the spkike glx metadata file and parse in a dictionary
    Agnostic: does not make any assumption on the keys/content, it just parses key=values
    """
    with open(md_file) as fid:
        md = fid.read()
    d = {}
    for a in md.splitlines():
        k, v = a.split('=')
        # if all char
        if v and re.fullmatch('[0-9,.]*', v):
            v = [float(val) for val in v.split(',')]
        d[k.replace('~', '')] = v
    return d

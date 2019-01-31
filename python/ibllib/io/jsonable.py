import json


def read(file):
    data = []
    with open(file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def _write(file, data, mode):
    with open(file, mode) as f:
        for obj in data:
            f.write(json.dumps(obj) + '\n')


def write(file, data):
    _write(file, data, 'w+')


def append(file, data):
    _write(file, data, 'a')

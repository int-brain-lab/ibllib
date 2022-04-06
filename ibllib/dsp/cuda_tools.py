import re
from pathlib import Path

from iblutil.util import Bunch


REGEX_PATTERN = r'const int\s+\S+\s+=\s+\S+.+'

def get_cuda(fn, **constants):
    """
    Finds the CUDA code for a function and optionally changes the default constants
    :param fn: String, name of cuda function
    :param constants: Integer constants to replace
    :return: code: String
             constants: Bunch
    """
    path = Path(__file__).parent / (fn + '.cu')
    assert path.exists
    code = path.read_text()
    code = code.replace('__global__ void', 'extern "C" __global__ void')
    if not constants:
        return code, Bunch(extract_constants_from_cuda(code))
    return change_cuda_constants(code, constants)


def extract_constants_from_cuda(code):
    """
    Find integer constants in the CUDA function
    :param code: String, CUDA function
    :return: names and values of integer constants
    """
    r = re.compile(REGEX_PATTERN)
    m = r.search(code)
    if m:
        constants = m.group(0).replace('const int', '').replace(';', '').split(',')
        for const in constants:
            a, b = const.strip().split('=')
            yield a.strip(), int(b.strip())


def change_cuda_constants(code, constants):
    """
    Changes default integer constants in the CUDA function with any that are passed
    :param code: String, CUDA function
    :param constants: dict, constants to change
    :return: code: String
             constants: Bunch
    """
    r = re.compile(REGEX_PATTERN)
    m = r.match(code)
    assert m, 'No constants found in CUDA code'
    pattern_length = m.span(0)[1] - 1
    default_constants_string = m.group(0).replace('const int', '').replace(';', '').split(',')
    code_constants = {}

    # Find default constants in CUDA code
    for default_constants_string in default_constants_string:
        name, value = default_constants_string.split('=')
        code_constants[name.strip()] = int(value.strip())

    # Replace default constants with the new user constants
    for name, value in constants.items():
        code_constants[name] = value

    new_strings = []
    for name, value in code_constants.items():
        new_strings.append(f'{name} = {value}')
    new_constants_string = ', '.join(new_strings)

    new_code = f'const int  {new_constants_string}{code[pattern_length:]}'

    return new_code, Bunch(code_constants)

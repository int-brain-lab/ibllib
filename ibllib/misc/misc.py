# library of small functions
import logging
import subprocess
import numpy as np
from ibllib.exceptions import NvidiaDriverNotReady

_logger = logging.getLogger('ibllib')


def _parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


def structarr(names, shape=None, formats=None):
    if not formats:
        formats = ['f8'] * len(names)
    dtyp = np.dtype({'names': names, 'formats': formats})
    return np.zeros(shape, dtype=dtyp)


def print_progress(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    """
    Call in a loop to create terminal progress bar

    :param iteration: Required  : current iteration (Int)
    :param total: Required  : total iterations (Int)
    :param prefix: Optional  : prefix string (Str)
    :param suffix: Optional: suffix string (Str)
    :param decimals: Optional: positive number of decimals in percent complete (Int)
    :param length: Optional: character length of bar (Int)
    :param fill: Optional: bar fill character (Str)
    :return: None
    """
    iteration += 1
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()


def check_nvidia_driver():
    """
    Checks if the GPU driver reacts and otherwise raises a custom error.
    Useful to check before long GPU-dependent processes.
    """
    process = subprocess.Popen('nvidia-smi', shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, executable="/bin/bash")
    info, error = process.communicate()
    if process.returncode != 0:
        raise NvidiaDriverNotReady(f"{error.decode('utf-8')}")
    _logger.info("nvidia-smi command successful")

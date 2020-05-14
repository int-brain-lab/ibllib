# library of small functions
import json
import logging

import numpy as np

_logger = logging.getLogger('ibllib')


def pprint(my_dict):
    print(json.dumps(my_dict, indent=4))


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


def logger_config(name=None):
    import logging
    import colorlog
    """
        Setup the logging environment
    """
    if not name:
        log = logging.getLogger()  # root logger
    else:
        log = logging.getLogger(name)
    log.setLevel(logging.INFO)
    format_str = '%(asctime)s.%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    cformat = '%(log_color)s' + format_str
    colors = {'DEBUG': 'green',
              'INFO': 'cyan',
              'WARNING': 'bold_yellow',
              'ERROR': 'bold_red',
              'CRITICAL': 'bold_purple'}
    formatter = colorlog.ColoredFormatter(cformat, date_format,
                                          log_colors=colors)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    log.addHandler(stream_handler)
    return log


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

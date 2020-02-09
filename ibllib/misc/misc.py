# library of small functions
import json
import logging
import traceback
from pathlib import Path

import numpy as np

from ibllib.misc import version
_logger = logging.getLogger('ibllib')


def pprint(my_dict):
    print(json.dumps(my_dict, indent=4))


def _parametrized(dec):
    def layer(*args, **kwargs):
        def repl(f):
            return dec(f, *args, **kwargs)
        return repl
    return layer


@_parametrized
def log2session_static(func, log_file_name):
    """ Decorator that will fork the log output of any function that takes a session path as
    first argument to a {session_path}/logs/yyyy-mm-dd_{log_filename}_ibllib_v1.2.3.log"""
    def func_wrapper(session_path, *args, **kwargs):
        fh = log2sessions_set(session_path, log_file_name)
        try:
            f = func(session_path, *args, **kwargs)
        except Exception as e:
            log2sessions_catch(e, session_path, log_file_name)
            f = None
        log2sessions_unset(log_file_name, fh)
        return f
    return func_wrapper


@_parametrized
def log2session(func, log_file_name):
    """ Decorator that will fork the log output of any method that takes a session path as
    first argument to a {session_path}/logs/yyyy-mm-dd_{log_filename}_ibllib_v1.2.3.log"""
    def func_wrapper(self, session_path, *args, **kwargs):
        fh = log2sessions_set(session_path, log_file_name)
        try:
            f = func(self, session_path, *args, **kwargs)
        except Exception as e:
            log2sessions_catch(e, session_path, log_file_name)
            f = None
        log2sessions_unset(log_file_name, fh)
        return f
    return func_wrapper


def log2sessions_set(session_path, log_type):
    log_file = Path(session_path).joinpath(
        'logs', f'_ibl_log.info.{log_type}_v{version.ibllib()}.log')
    log_file.parent.mkdir(exist_ok=True, parents=True)
    fh = logging.FileHandler(log_file)
    str_format = '%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    fh.setFormatter(logging.Formatter(str_format))
    _logger.addHandler(fh)
    return fh


def log2sessions_unset(log_type, fh=None):
    for hdlr in _logger.handlers:
        if '_ibl_log.' in str(hdlr):
            _logger.removeHandler(hdlr)
    if fh:
        fh.close()


def log2sessions_catch(e, sessionpath, log_type):
    error_message = f'{sessionpath} failed extraction \n  {str(e)} \n' \
                    f'{traceback.format_exc()}'
    err_file = Path(sessionpath).joinpath(
        'logs', f'_ibl_log.error.{log_type}_v{version.ibllib()}.log')
    with open(err_file, 'w+') as fid:
        fid.write(error_message)
    _logger.error(error_message)


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

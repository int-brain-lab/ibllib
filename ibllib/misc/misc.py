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


def bincount2D(x, y, xbin=0, ybin=0, xlim=None, ylim=None, weights=None):
    """
    Computes a 2D histogram by aggregating values in a 2D array.

    :param x: values to bin along the 2nd dimension (c-contiguous)
    :param y: values to bin along the 1st dimension
    :param xbin: bin size along 2nd dimension (set to 0 to aggregate according to unique values)
    :param ybin: bin size along 1st dimension (set to 0 to aggregate according to unique values)
    :param xlim: (optional) 2 values (array or list) that restrict range along 2nd dimension
    :param ylim: (optional) 2 values (array or list) that restrict range along 1st dimension
    :param weights: (optional) defaults to None, weights to apply to each value for aggregation
    :return: 3 numpy arrays MAP [ny,nx] image, xscale [nx], yscale [ny]
    """
    # if no bounds provided, use min/max of vectors
    if not xlim:
        xlim = [np.min(x), np.max(x)]
    if not ylim:
        ylim = [np.min(y), np.max(y)]

    # create the indices on which to aggregate: binning is different that aggregating
    if xbin:
        xscale = np.arange(xlim[0], xlim[1] + xbin / 2, xbin)
        xind = (np.floor((x - xlim[0]) / xbin)).astype(np.int64)
    else:  # if bin size = 0 , aggregate over unique values
        xscale, xind = np.unique(x, return_inverse=True)
    if ybin:
        yscale = np.arange(ylim[0], ylim[1] + ybin / 2, ybin)
        yind = (np.floor((y - ylim[0]) / ybin)).astype(np.int64)
    else:  # if bin size = 0 , aggregate over unique values
        yscale, yind = np.unique(y, return_inverse=True)

    # aggregate by using bincount on absolute indices for a 2d array
    nx, ny = [xscale.size, yscale.size]
    ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
    r = np.bincount(ind2d, minlength=nx * ny, weights=weights).reshape(ny, nx)
    return r, xscale, yscale

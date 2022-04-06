from math import ceil

import cupy as cp
from scipy.signal import sosfilt_zi

from .cuda_tools import get_cuda


def sosfiltfilt_gpu(sos, x, axis=-1):
    """
    GPU implementation of Scipy's forward-backward digital filter using cascaded second-order
    sections.
    :param sos : Numpy array with shape (n_sections, 6)
        Each row corresponds to a second-order section, with the first three columns providing the
        numerator coefficients and the last three providing the denominator coefficients.
    :param x: Cupy array, data to be filtered
    :param axis: Axis along which the filter is applied
    :return: Filtered array
    """

    sos = cp.asnumpy(sos)

    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')

    ntaps = 2 * n_sections + 1
    ntaps -= min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())

    x_pad, edge = default_pad(x, axis=axis, ntaps=ntaps)

    zi_cpu = sosfilt_zi(sos)  # shape (n_sections, 2) --> (n_sections, ..., 2, ...)
    zi_shape = [1] * x.ndim
    zi_shape[axis] = 2
    zi_cpu.shape = [n_sections] + zi_shape
    zi = cp.array(zi_cpu)

    x_0 = axis_slice(x_pad, stop=1, axis=axis)
    y = sosfilt_gpu(sos, x_pad, axis=axis, zi=zi * x_0)
    y_0 = axis_slice(y, start=-1, axis=axis)
    y = cp.ascontiguousarray(axis_reverse(y, axis=axis))
    y = sosfilt_gpu(sos, y, axis=axis, zi=zi * y_0)
    y = axis_reverse(y, axis=axis)

    if edge > 0:
        y = axis_slice(y, start=edge, stop=-edge, axis=axis)

    return cp.ascontiguousarray(y)


def sosfilt_gpu(sos, x, axis, zi):
    """
    GPU implementation of Scipy function
    Filter data along one dimension using cascaded second-order sections.
    Filter a cupy array x, using a digital IIR filter defined by sos
    :param sos : Numpy/cupy array with shape (n_sections, 6)
        Each row corresponds to a second-order section, with the first three columns providing the
        numerator coefficients and the last three providing the denominator coefficients.
    :param x: Data to be filtered, float32 cupy array
    :param axis: Axis along which the filter is applied
    :param zi: Initial conditions for the cascaded filter delays.
    :return:
    """

    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')

    x_zi_shape = list(x.shape)
    x_zi_shape[axis] = 2
    x_zi_shape = tuple([n_sections] + x_zi_shape)

    if zi is not None:
        assert zi.shape == x_zi_shape, f'zi has shape {zi.shape}, expected {x_zi_shape}'
        zi = cp.array(zi, dtype='float32')
    else:
        zi = cp.zeros(x_zi_shape, dtype='float32')

    sos = cp.array(sos, dtype='float32')
    assert x.dtype == 'float32', f'Expected float32 data, got {x.dtype}'

    axis = axis % x.ndim
    x = cp.ascontiguousarray(cp.moveaxis(x, axis, -1))
    zi = cp.ascontiguousarray(cp.moveaxis(zi, [0, axis + 1], [-2, -1]))

    _cuda_sosfilt(sos, x, zi)

    x = cp.moveaxis(x, -1, axis)

    return x


def _cuda_sosfilt(sos, x, zi):

    n_signals, n_samples = x.shape
    n_sections = sos.shape[0]

    n_blocks, n_threads = sosfilt_kernel_params(n_signals)

    code, consts = get_cuda('sosfilt', n_signals=n_signals, n_samples=n_samples,
                            n_sections=n_sections)
    kernel = cp.RawKernel(code, 'sosfilt')
    kernel((n_blocks,), (n_threads,), (sos, x, zi))


N_THREADS_BLOCK = 1024
MIN_GPU_BLOCKS = 8


def sosfilt_kernel_params(n_signals):
    """
    Get grid and block parameters for the CUDA kernel
    :param n_signals: Int, no of signals to be independently filtered
    :return: n_blocks: Int, no of blocks
             n_threads, Int, no of threads per block
    """
    # From testing, using at least 8 blocks increases speed when there are fewer signals

    if n_signals >= MIN_GPU_BLOCKS * N_THREADS_BLOCK:
        return ceil(n_signals / N_THREADS_BLOCK), N_THREADS_BLOCK

    return MIN_GPU_BLOCKS, ceil(n_signals / MIN_GPU_BLOCKS)


def default_pad(x, axis, ntaps=None, padlen=None):
    """
    Applies the default padding used by Scipy to a cupy array before filtering
    :param x: Array to be padded, cupy array
    :param axis: Axis to pad
    :param ntaps: Number of FIR taps
    :param padlen: Padding length, optional int
    :return: Padded array
    """
    edge = padlen or ntaps * 3
    assert edge is not None

    # Use odd padding by default
    x_pad = odd_ext(x, edge, axis=axis)

    return x_pad, edge


def odd_ext(x, n, axis=-1):
    """
    Odd extension at the boundaries of an array
    Generate a new cupy array by making an odd extension of `x` along an axis.
    :param x: Array to be padded, cupy array
    :param n: Padding length, int
    :param axis: Axis to pad
    :return: Padded array
    """
    if n < 1:
        return x
    if n > x.shape[axis] - 1:
        raise ValueError(("The extension length n (%d) is too big. " +
                          "It must not exceed x.shape[axis]-1, which is %d.")
                         % (n, x.shape[axis] - 1))
    left_end = axis_slice(x, start=0, stop=1, axis=axis)
    left_ext = axis_slice(x, start=n, stop=0, step=-1, axis=axis)
    right_end = axis_slice(x, start=-1, axis=axis)
    right_ext = axis_slice(x, start=-2, stop=-(n + 2), step=-1, axis=axis)
    ext = cp.concatenate((2 * left_end - left_ext,
                          x,
                          2 * right_end - right_ext),
                         axis=axis)
    return ext


def axis_slice(a, start=None, stop=None, step=None, axis=-1):
    """
    Take slice along array
    """
    a_slice = [slice(None)] * a.ndim
    a_slice[axis] = slice(start, stop, step)
    b = a[tuple(a_slice)]
    return b


def axis_reverse(a, axis=-1):
    """
    Reverse array along axis
    """
    return axis_slice(a, step=-1, axis=axis)

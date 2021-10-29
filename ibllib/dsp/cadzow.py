import numpy as np

from iblutil.numerical import ismember2d


def derank(T, r):
    u, s, v = np.linalg.svd(T)
    # try non-integer rank as a proportion of singular values ?
    # ik = np.searchsorted(np.cumsum(s) / np.sum(s), KEEP)
    T_ = np.zeros_like(T)
    for i in np.arange(r):
        T_ += s[i] * np.outer(u.T[i], v[i])
    return T_


def traj_matrix_indices(n):
    """
    Computes the single spatial dimension Toeplitz-like indices from a number of spatial traces
    :param n: number of dimensions
    :return: 2-D int matrix whose elements are indices of the spatial dimension
    """
    nrows = int(np.floor(n / 2 + 1))
    ncols = int(np.ceil(n / 2))
    itraj = np.tile(np.arange(nrows), (ncols, 1)).T + np.flipud(np.arange(ncols))
    return itraj


def trajectory(x, y):
    """
    Computes the 2 spatial dimensions block-Toeplitz indices from x and y coordinates
    """
    xu, ix = np.unique(x, return_inverse=True)
    yu, iy = np.unique(y, return_inverse=True)
    nx, ny = (np.size(xu), np.size(yu))

    tiy_ = traj_matrix_indices(ny)
    tix_ = traj_matrix_indices(nx)
    tiy = np.tile(tiy_, tix_.shape)
    tix = np.repeat(np.repeat(tix_, tiy_.shape[0], axis=0), tiy_.shape[1], axis=1)

    it, itr = ismember2d(np.c_[tix.flatten(), tiy.flatten()], np.c_[ix, iy])
    it = np.unravel_index(np.where(it)[0], tiy.shape)

    T = np.zeros(tix.shape, dtype=np.complex128)

    trcount = np.bincount(itr)
    return T, it, itr, trcount


def denoise(WAV, x, y, r, imax=None, niter=1):
    """
    Applies cadzow denoising by de-ranking spatial matrices in frequency domain
    :param WAV: np array nc / ns in frequency domain
    :param x: trace spatial coordinate (np.array)
    :param y: trace spatial coordinate (np.array)
    :param r: rank
    :param imax: index of the maximum frequency to keep, all frequencies are de-ranked if None (None)
    :param niter: number of iterations (1)
    :return: WAV_: np array nc / ns in frequency domain
    """
    WAV_ = np.copy(WAV)
    imax = np.minimum(WAV.shape[-1], imax) if imax else WAV.shape[-1]
    T, it, itr, trcount = trajectory(x, y)
    for ind_f in np.arange(imax):
        for _ in np.arange(niter):
            T[it] = WAV_[itr, ind_f]
            T_ = derank(T, r)
            WAV_[:, ind_f] = np.bincount(itr, weights=np.real(T_[it]))
            WAV_[:, ind_f] += 1j * np.bincount(itr, weights=np.imag(T_[it]))
            WAV_[:, ind_f] /= trcount

    return WAV_

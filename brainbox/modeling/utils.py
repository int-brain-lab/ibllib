import numpy as np
from numpy.matlib import repmat


def raised_cosine(duration, nbases, binfun):
    nbins = binfun(duration)
    ttb = repmat(np.arange(1, nbins + 1).reshape(-1, 1), 1, nbases)
    dbcenter = nbins / nbases
    cwidth = 4 * dbcenter
    bcenters = 0.5 * dbcenter + dbcenter * np.arange(0, nbases)
    x = ttb - repmat(bcenters.reshape(1, -1), nbins, 1)
    bases = (np.abs(x / cwidth) < 0.5) * (np.cos(x * np.pi * 2 / cwidth) * 0.5 + 0.5)
    return bases


def full_rcos(duration, nbases, binfun, n_before=1):
    if not isinstance(n_before, int):
        n_before = int(n_before)
    nbins = binfun(duration)
    ttb = repmat(np.arange(1, nbins + 1).reshape(-1, 1), 1, nbases)
    dbcenter = nbins / (nbases - 2)
    cwidth = 4 * dbcenter
    bcenters = 0.5 * dbcenter + dbcenter * np.arange(-n_before, nbases - n_before)
    x = ttb - repmat(bcenters.reshape(1, -1), nbins, 1)
    bases = (np.abs(x / cwidth) < 0.5) * (np.cos(x * np.pi * 2 / cwidth) * 0.5 + 0.5)
    return bases


def neglog(weights, x, y):
    xproj = x @ weights
    f = np.exp(xproj)
    nzidx = f != 0
    if np.any(y[~nzidx] != 0):
        return np.inf
    return -y[nzidx].reshape(1, -1) @ xproj[nzidx] + np.sum(f)


class SequentialSelector:
    def __init__(self, model, n_features_to_select=None, direction='forward', scoring=None):
        """
        Sequential feature selection for neural models

        Parameters
        ----------
        model : brainbox.modeling.neural_model.NeuralModel
            Any class which inherits NeuralModel and has already been instantiated.
        n_features_to_select : int, optional
            Number of covariates to select. When None, will sequentially fit all parameters and
            store the associated scores. By default None
        direction : str, optional
            Direction of sequential selection. 'forward' indicates model will be built from 1
            regressor up, while 'backward' indicates regrssors will be removed one at a time until
            n_features_to_select is reached or 1 regressor remains. By default 'forward'
        scoring : str, optional
            Scoring function to use. Must be a valid argument to the subclass of NeuralModel passed
            to SequentialSelector. By default None
        """
        raise NotImplementedError('Sorry UwU this isn\'t ready yet')
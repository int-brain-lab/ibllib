from .neural_model import NeuralModel

from warnings import warn, catch_warnings
import numpy as np
from numpy.linalg.linalg import LinAlgError
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import r2_score
import numba as nb
from numpy.matlib import repmat
from scipy.optimize import minimize
from scipy.special import xlogy
from tqdm import tqdm
from .utils import *

class PoissonGLM(NeuralModel):
    def __init__(design_matrix, spk_times, spk_clu, fitting_metric='dsq',
                 model='default', alpha=0,
                 train=0.8, blocktrain=False, mintrials=100, subset=False)
    super().__init__(design_matrix, spk_times, spk_clu,
                     train, blocktrain, mintrials, subset)
    assert(model in ['default', 'without_intercept']), 'model must be default or without_intercept'
    self.fitting_metric = fitting_metric
    if model=='default':
        self.fit_intercept = True
    else:
        self.fit_intercept = False
    self.alpha=alpha

    """
    Linear-nonlinear poisson model, a type of generalized linear model, for neural responses.
    """
    def _fit_sklearn(self, dm, binned, cells=None, noncovwarn=True):
        """
        Fit a GLM using scikit-learn implementation of PoissonRegressor. Uses a regularization
        strength parameter alpha, which is the strength of ridge regularization term.

        Parameters
        ----------
        dm : numpy.ndarray
            Design matrix, in which rows are observations and columns are regressor values. Should
            NOT contain a bias column for the intercept. Scikit-learn handles that.
        binned : numpy.ndarray
            Vector of observed spike counts which we seek to predict. Must be of the same length
            as dm.shape[0]
        alpha : float
            Regularization strength, applied as multiplicative constant on ridge regularization.
        cells : list
            List of cells which should be fit. If None is passed, will default to fitting all cells
            in clu_ids
        """
        if cells is None:
            cells = self.clu_ids.flatten()
        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts')
        nonconverged = []
        for cell in tqdm(cells, 'Fitting units:', leave=False):
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            cellbinned = binned[:, cell_idx]
            with catch_warnings(record=True) as w:
                fitobj = PoissonRegressor(alpha=self.alpha,
                                          max_iter=300,
                                          fit_intercept=self.fit_intercept).fit(dm, cellbinned)
            if len(w) != 0:
                nonconverged.append(cell)
            wts = np.concatenate([[fitobj.intercept_], fitobj.coef_], axis=0)
            biasdm = np.pad(dm.copy(), ((0, 0), (1, 0)), 'constant', constant_values=1)
            coefs.at[cell] = fitobj.coef_
            if fit_intercept:
                intercepts.at[cell] = fitobj.intercept_
            else:
                intercepts.at[cell] = 0
        if noncovwarn:
            if len(nonconverged) != 0:
                warn(f'Fitting did not converge for some units: {nonconverged}')
        
        return coefs, intercepts

    def score(self, metric='dsq', **kwargs):

        'negLog'
        """
        Utility function for computing D^2 (pseudo R^2) on a given set of weights and
        intercepts. Is be used in both model subsetting and the mother score() function of the GLM.

        Parameters
        ----------
        weights : pd.Series
            Series in which entries are numpy arrays containing the weights for a given cell.
            Indices should be cluster ids.
        intercepts : pd.Series
            Series in which elements are the intercept fit to each cell. Indicies should match
            weights.
        dm : numpy.ndarray
            Design matrix. Should not contain the bias column. dm.shape[1] should be the same as
            the length of an element in weights.
        binned : numpy.ndarray
            nT x nCells array, in which each column is the binned spike train for a single unit.
            Should be the same number of rows as dm.

        Compute the squared deviance of the model, i.e. how much variance beyond the null model
        (a poisson process with the same mean, defined by the intercept, at every time step) the
        model which was fit explains.
        For a detailed explanation see https://bookdown.org/egarpor/PM-UC3M/glm-deviance.html`

        Returns
        -------
        pandas.Series
            A series in which the index are cluster IDs and each entry is the D^2 for the model fit
            to that cluster
        """
        assert(metric in ['dsq', 'rsq', 'negLog']), 'metric must be dsq, rsq or negLog'
        assert(len(kwargs)==0 or len(kwargs)==4), 'wrong input specification in score'
        if not hasattr(self, 'coefs') or 'weights' not in kwargs.keys():
            raise AttributeError('Fit was not run. Please run fit first.')
        if hasattr(self, 'submodel_scores'):
            return self.submodel_scores

        if len(kwargs)==4:
            weights, intercepts, dm, binned = kwargs['weights'], kwargs['intercepts'], \
                                                            kwargs['dm'], kwargs['binned']
        else:
            testmask = np.isin(self.trlabels, self.testinds).flatten()      
            weights, intercepts, dm, binned = self.coefs, self.intercepts, self.dm[testmask, :], self.binnedspikes[testmask]

        scores = pd.Series(index=weights.index, name='scores')
        for cell in weights.index:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            wt = weights.loc[cell].reshape(-1, 1)
            bias = intercepts.loc[cell]
            y = binned[:, cell_idx]
            if metric in ['dsq', 'rsq']:
                pred = np.exp(dm @ wt + bias)
                if metric=='dsq':
                    null_pred = np.ones_like(pred) * np.mean(y)
                    null_deviance = 2 * np.sum(xlogy(y, y / null_pred.flat) - y + null_pred.flat)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        full_deviance = 2 * np.sum(xlogy(y, y / pred.flat) - y + pred.flat)
                    d_sq = 1 - (full_deviance / null_deviance)
                    scores.at[cell] = d_sq
                else:
                    scores.at[cell] = r2_score(y, pred)
            else:
                biasdm = np.pad(dm, ((0, 0), (1, 0)), constant_values=1)
                scores.at[cell] = -neglog(np.vstack((bias, wt)).flatten(), biasdm, y) / np.sum(y)
        return scores





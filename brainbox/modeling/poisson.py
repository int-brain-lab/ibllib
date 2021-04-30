from .neural_model import NeuralModel

from warnings import warn, catch_warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm


class PoissonGLM(NeuralModel):
    def __init__(self, design_matrix, spk_times, spk_clu,
                 binwidth=0.02, metric='dsq', model='default', alpha=0,
                 train=0.8, blocktrain=False, mintrials=100, subset=False):
        super().__init__(design_matrix, spk_times, spk_clu,
                         binwidth, train, blocktrain, mintrials, subset)
        self.metric = metric
        if model == 'default':
            self.fit_intercept = True
        elif model == 'no_intercept':
            self.fit_intercept = False
        else:
            raise ValueError('model must be \'default\' or \'no_intercept\'')
        self.alpha = alpha

    def _fit(self, dm, binned, cells=None, noncovwarn=False):
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
            coefs.at[cell] = fitobj.coef_
            if self.fit_intercept:
                intercepts.at[cell] = fitobj.intercept_
            else:
                intercepts.at[cell] = 0
        if noncovwarn:
            if len(nonconverged) != 0:
                warn(f'Fitting did not converge for some units: {nonconverged}')

        return coefs, intercepts

    def score(self, metric='dsq', **kwargs):
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
        assert(len(kwargs) == 0 or len(kwargs) == 4), 'wrong input specification in score'
        if not hasattr(self, 'coefs') or 'weights' not in kwargs.keys():
            raise AttributeError('Fit was not run. Please run fit first.')
        if hasattr(self, 'submodel_scores'):
            return self.submodel_scores

        if len(kwargs) == 4:
            weights, intercepts, dm, binned = kwargs['weights'], kwargs['intercepts'], \
                kwargs['dm'], kwargs['binned']
        else:
            testmask = np.isin(self.trlabels, self.testinds).flatten()
            weights, intercepts, dm, binned = self.coefs, self.intercepts,\
                self.dm[testmask, :], self.binnedspikes[testmask]

        scores = pd.Series(index=weights.index, name='scores')
        for cell in weights.index:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            wt = weights.loc[cell].reshape(-1, 1)
            bias = intercepts.loc[cell]
            y = binned[:, cell_idx]
            scores.at[cell] = self._scorer(wt, bias, dm, y)
        return scores

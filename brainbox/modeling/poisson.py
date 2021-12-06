from .neural_model import NeuralModel

from warnings import warn, catch_warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from tqdm import tqdm


class PoissonGLM(NeuralModel):
    def __init__(self, design_matrix, spk_times, spk_clu,
                 binwidth=0.02, metric='dsq', fit_intercept=True, alpha=0,
                 train=0.8, blocktrain=False, mintrials=100, subset=False):
        super().__init__(design_matrix, spk_times, spk_clu,
                         binwidth, train, blocktrain, mintrials, subset)
        # TODO: Implement grid search over alphas to find optimal value
        self.metric = metric
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.link = np.exp
        self.invlink = np.log

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
            List of cells labels for columns in binned. Will default to all cells in model if None
            is passed. Must be of the same length as columns in binned. By default None.
        """
        if cells is None:
            cells = self.clu_ids.flatten()
        if cells.shape[0] != binned.shape[1]:
            raise ValueError('Length of cells does not match shape of binned')

        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts')
        nonconverged = []
        for cell in tqdm(cells, 'Fitting units:', leave=False):
            cell_idx = np.argwhere(cells == cell)[0, 0]
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

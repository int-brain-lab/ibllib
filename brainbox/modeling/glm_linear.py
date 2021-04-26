
"""
GLM fitting utilities based on NeuroGLM by Il Memming Park, Jonathan Pillow:

https://github.com/pillowlab/neuroGLM

Berk Gercek
International Brain Lab, 2020
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from scipy.special import xlogy
from .neural_model import NeuralModel


class LinearGLM(NeuralModel):

    def __init__(self, design_matrix, spk_times, spk_clu, fitting_metric='rsq',
                 model='default', alpha=None,
                 train=0.8, blocktrain=False, mintrials=100, stepwise=False):
        super().__init__(design_matrix, spk_times, spk_clu,
                         train, blocktrain, mintrials, stepwise)
        assert(model in ['default', 'ridge']), 'model but be default or ridge'
        if model == 'default' and alpha is not None:
            assert('problem in model specification')
        self.fitting_metric = fitting_metric
        if model == 'ridge':
            if alpha is None:
                print('alpha is None in ridge model.'
                      'Falling back on sklearn default argument:alpha=(0.1, 1.0, 10.0)')
                alpha = (0.1, 1.0, 10.0)
            self.ridge, self.alpha = True, alpha
        else:
            self.ridge = False

    def _fit(self, dm, binned, cells=None):
        if cells is None:
            cells = self.clu_ids.flatten()
        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts')

        if self.ridge:
            lm = RidgeCV(alphas=self.alpha, cv=3).fit(dm, binned)
        else:
            lm = LinearRegression().fit(dm, binned)
        weight, intercept = lm.coef_, lm.intercept_
        for cell in cells:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            coefs.at[cell] = weight[cell_idx, :]
            intercepts.at[cell] = intercept[cell_idx]
        return coefs, intercepts

    def score(self, metric='rsq', **kwargs):
        assert(metric in ['dsq', 'rsq', 'msespike']), 'metric must be dsq, rsq or msespike'
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
            pred = (dm @ wt + bias).flatten()
            if metric == 'dsq':
                null_pred = np.ones_like(pred) * np.mean(y)
                null_deviance = 2 * np.sum(xlogy(y, y / null_pred.flat) - y + null_pred.flat)
                with np.errstate(invalid='ignore', divide='ignore'):
                    full_deviance = 2 * np.sum(xlogy(y, y / pred.flat) - y + pred.flat)
                d_sq = 1 - (full_deviance / null_deviance)
                scores.at[cell] = d_sq
            elif metric == 'msespike':
                residuals = (y - pred) ** 2
                msespike = residuals.sum() / y.sum()
                scores.at[cell] = msespike
            else:
                scores.at[cell] = r2_score(y, pred)
        return scores

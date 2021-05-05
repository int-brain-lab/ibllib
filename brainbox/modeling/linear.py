
"""
GLM fitting utilities based on NeuroGLM by Il Memming Park, Jonathan Pillow:

https://github.com/pillowlab/neuroGLM

Berk Gercek
International Brain Lab, 2020
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LinearRegression
from .neural_model import NeuralModel


class LinearGLM(NeuralModel):
    def __init__(self, design_matrix, spk_times, spk_clu,
                 binwidth=0.02, metric='rsq', estimator=None,
                 train=0.8, blocktrain=False, mintrials=100):
        """
        Fit a linear model using a DesignMatrix object and spike data. Can use ridge regression
        or pure linear regression

        Parameters
        ----------
        design_matrix : brainbox.modeling.design.DesignMatrix
            Design matrix specification that includes information about groups of regressors
        spk_times : np.array
            1-D Array of spike times
        spk_clu : np.array
            1-D Array, same shape as spk_times, assigning cluster identities to each spike time
        binwidth : float, optional
            Length of the bins to be used to count spikes, by default 0.02
        metric : str, optional
            Scoring metric which to use for the models .score() method. Can be rsq, dsq, msepike,
            by default 'rsq'
        estimator : sklearn.linear_model estimator, optional
            Estimator to use for model fitting. If None will default to pure linear regression.
            Must have a .fit(X, y) method and after fitting contain .coef_ and .intercept_
            attributes. By default None.
        train : float, optional
            Proportion of data to use as training set, by default 0.8
        blocktrain : bool, optional
            Whether to use contiguous blocks of trials for cross-validation, by default False
        mintrials : int, optional
            Minimum number of trials in which a neuron must fire >0 spikes to be considered for
            fitting, by default 100
        """
        super().__init__(design_matrix, spk_times, spk_clu,
                         binwidth, train, blocktrain, mintrials)
        if estimator is None:
            estimator = LinearRegression()
        if not isinstance(estimator, BaseEstimator):
            raise ValueError('Estimator must be a scikit-learn estimator, e.g. LinearRegression')
        self.metric = metric
        self.estimator = estimator

    def _fit(self, dm, binned, cells=None):
        """
        Fitting primitive that brainbox.NeuralModel.fit method will call
        """
        if cells is None:
            cells = self.clu_ids.flatten()
        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts')

        lm = self.estimator.fit(dm, binned)
        weight, intercept = lm.coef_, lm.intercept_
        for cell in cells:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            coefs.at[cell] = weight[cell_idx, :]
            intercepts.at[cell] = intercept[cell_idx]
        return coefs, intercepts

    def score(self):
        """
        Score model using chosen metric

        Returns
        -------
        pandas.Series
            Score using chosen metric (defined at instantiation) for each unit fit by the model.
        """
        if not hasattr(self, 'coefs'):
            raise AttributeError('Model has not been fit yet.')

        testmask = np.isin(self.design.trlabels, self.testinds).flatten()
        dm, binned = self.design[testmask, :], self.binnedspikes[testmask]

        scores = pd.Series(index=self.coefs.index, name='scores')
        for cell in self.coefs.index:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            wt = self.coefs.loc[cell].reshape(-1, 1)
            bias = self.intercepts.loc[cell]
            y = binned[:, cell_idx]
            scores.at[cell] = self._scorer(wt, bias, dm, y)
        return scores

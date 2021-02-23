
"""
GLM fitting utilities based on NeuroGLM by Il Memming Park, Jonathan Pillow:

https://github.com/pillowlab/neuroGLM

Berk Gercek
International Brain Lab, 2020
"""
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from scipy.special import xlogy
from .glm import NeuralGLM


class LinearGLM(NeuralGLM):
    def _fit_purelinear(self, dm, binned, cells=None):
        if cells is None:
            cells = self.clu_ids.flatten()
        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts')

        lm = LinearRegression().fit(dm, binned)
        weight, intercept = lm.coef_, lm.intercept_
        for cell in cells:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            coefs.at[cell] = weight[cell_idx, :]
            intercepts.at[cell] = intercept[cell_idx]
        return coefs, intercepts

    def _fit_ridge(self, dm, binned, cells=None):
        if cells is None:
            cells = self.clu_ids.flatten()
        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts')

        lm = RidgeCV(cv=3).fit(dm, binned)
        weight, intercept = lm.coef_, lm.intercept_
        for cell in cells:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            coefs.at[cell] = weight[cell_idx, :]
            intercepts.at[cell] = intercept[cell_idx]
        return coefs, intercepts

    def fit(self, method='pure', alpha=0, multi_score=False,
            printcond=True):
        """
        Fit the current set of binned spikes as a function of the current design matrix. Requires
        NeuralGLM.bin_spike_trains and NeuralGLM.compile_design_matrix to be run first. Will store
        the fit weights to an internal variable. To access these fit weights in a pandas DataFrame
        use the NeuralGLM.combine_weights method.

        Parameters
        ----------
        method : str, optional
            'sklearn' or 'minimize', describes the fitting method used to obtain weights.
            Scikit-learn uses weight normalization and regularization and will return significantly
            different results from 'minimize', which simply minimizes the negative log likelihood
            of the data given the covariates, by default 'sklearn'
        alpha : float, optional
            Regularization strength for scikit-learn implementation of GLM fitting, where 0 is
            effectively unregularized weights. Does not function in the minimize
            option, by default 1
        epochs : int
            Used for _fit_pytorch funtion, see details there
        optim : string
            Used for _fit_pytorch funtion, see details there
        lr : float
            Used for _fit_pytorch funtion, see details there
        fit_intercept : bool
            Only works when using 'sklearn' method. Whether or not to fit the bias term of the GLM
        printcond : bool
            Whether or not to print the condition number of the design matrix. Defaults to True

        Returns
        -------
        coefs : list
            List of coefficients fit. Not recommended to use these for interpretation. Use
            the .combine_weights() method instead.
        intercepts : list
            List of intercepts (bias terms) fit. Not recommended to use these for interpretation.
        """
        if not self.compiled:
            raise AttributeError('Design matrix has not been compiled yet. Please run '
                                 'neuroglm.compile_design_matrix() before fitting.')
        if method not in ('pure', 'ridge'):
            raise ValueError('Method must be \'pure\' or \'ridge\'')
        trainmask = np.isin(self.trlabels, self.traininds).flatten()  # Mask for training data
        trainbinned = self.binnedspikes[trainmask]
        if printcond:
            print(f'Condition of design matrix is {np.linalg.cond(self.dm[trainmask])}')

        if not self.subset:
            if method == 'pure':
                traindm = self.dm[trainmask]
                coefs, intercepts, = self._fit_purelinear(traindm, trainbinned)
            elif method == 'ridge':
                traindm = self.dm[trainmask]
                coefs, intercepts, = self._fit_ridge(traindm, trainbinned)

            self.coefs = coefs
            self.intercepts = intercepts
            self.fitmethod = method
            return
        else:
            # Get testing matrices for scoring in submodels
            testmask = np.isin(self.trlabels, self.testinds).flatten()
            testbinned = self.binnedspikes[testmask]

            # Build single-parameter-group models first:
            singlepar_scores = pd.DataFrame(columns=['cell', 'covar', 'scores'])
            if multi_score:
                altsinglepar_scores = pd.DataFrame(columns=['cell', 'covar', 'scores'])
            for cov in tqdm(self.covar, desc='Fitting single-cov models:', leave=False):
                dmcols = self.covar[cov]['dmcol_idx']
                colmask = np.zeros(self.dm.shape[1], dtype=bool)
                colmask[dmcols] = True
                traindm = self.dm[np.ix_(trainmask, colmask)]
                testdm = self.dm[np.ix_(testmask, colmask)]
                if method == 'pure':
                    coefs, intercepts, = self._fit_purelinear(traindm, trainbinned)
                if method == 'ridge':
                    coefs, intercepts, = self._fit_ridge(traindm, trainbinned)
                scores = self._score_submodel(coefs, intercepts, testdm, testbinned)
                scoresdf = pd.DataFrame(scores).reset_index()
                scoresdf.rename(columns={'index': 'cell'}, inplace=True)
                scoresdf['covar'] = cov
                singlepar_scores = pd.concat([singlepar_scores, scoresdf], sort=False)
                if multi_score:
                    altscores = self._score_submodel(coefs, intercepts, testdm, testbinned,
                                                     msespike=True)
                    altscoresdf = pd.DataFrame(altscores).reset_index()
                    altscoresdf.rename(columns={'index': 'cell'}, inplace=True)
                    altscoresdf['covar'] = cov
                    altsinglepar_scores = pd.concat([altsinglepar_scores, altscoresdf], sort=False)
            singlepar_scores.set_index(['cell', 'covar'], inplace=True)
            singlepar_scores.sort_values(by=['cell', 'scores'], ascending=False, inplace=True)
            if multi_score:
                altsinglepar_scores.set_index(['cell', 'covar'], inplace=True)
                altsinglepar_scores.sort_values(by=['cell', 'scores'], ascending=False,
                                                inplace=True)
            fitcells = singlepar_scores.index.levels[0]
            # Iteratively build model with 2 through K parameter groups:
            submodel_scores = singlepar_scores.unstack()
            submodel_scores.columns = submodel_scores.columns.droplevel()
            if multi_score:
                altsubmodel_scores = altsinglepar_scores.unstack()
                altsubmodel_scores.columns = altsubmodel_scores.columns.droplevel()
            for i in tqdm(range(2, len(self.covar) + 1), desc='Fitting submodels', leave=False):
                cellcovars = {cell: tuple(singlepar_scores.loc[cell].iloc[:i].index)
                              for cell in fitcells}
                covarsets = [frozenset(cov) for cov in cellcovars.values()]
                # Iterate through unique unordered combinations of covariates
                iscores = []
                altiscores = []
                progressdesc = f'Fitting covariate sets for {i} parameter groups'
                for covarset in tqdm(set(covarsets), desc=progressdesc, leave=False):
                    currcells = [cell for cell, covar in cellcovars.items()
                                 if set(covar) == covarset]
                    currcols = np.hstack([self.covar[cov]['dmcol_idx'] for cov in covarset])
                    colmask = np.zeros(self.dm.shape[1], dtype=bool)
                    colmask[currcols] = True
                    traindm = self.dm[np.ix_(trainmask, colmask)]
                    testdm = self.dm[np.ix_(testmask, colmask)]
                    if method == 'pure':
                        coefs, intercepts = self._fit_purelinear(traindm, trainbinned,
                                                                 cells=currcells)
                    if method == 'ridge':
                        coefs, intercepts = self._fit_ridge(traindm, trainbinned,
                                                            cells=currcells)
                    iscores.append(self._score_submodel(coefs, intercepts, testdm, testbinned))
                    if multi_score:
                        altiscores.append(self._score_submodel(coefs, intercepts, testdm,
                                                               testbinned, msespike=True))
                submodel_scores[f'{i}cov'] = pd.concat(iscores).sort_index()
                if multi_score:
                    altsubmodel_scores[f'{i}cov'] = pd.concat(altiscores).sort_index()
            self.submodel_scores = submodel_scores
            if multi_score:
                self.altsubmodel_scores = altsubmodel_scores
            self.coefs = coefs
            self.intercepts = intercepts
            return

    def _score_submodel(self, weights, intercepts, dm, binned, msespike=False):
        scores = pd.Series(index=weights.index, name='scores')
        for cell in weights.index:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            wt = weights.loc[cell].reshape(-1, 1)
            bias = intercepts.loc[cell]
            y = binned[:, cell_idx]
            pred = (dm @ wt + bias).flatten()
            if msespike:
                residuals = (y - pred) ** 2
                msespike = residuals.sum() / y.sum()
                scores.at[cell] = msespike
            else:
                scores.at[cell] = r2_score(y, pred)
        return scores

    def score(self, dsq=False, msespike=False):
        if not hasattr(self, 'coefs'):
            raise AttributeError('Fit was not run. Please run fit first.')
        if hasattr(self, 'submodel_scores'):
            return self.submodel_scores
        testmask = np.isin(self.trlabels, self.testinds).flatten()
        testdm = self.dm[testmask, :]
        scores = pd.Series(index=self.coefs.index)
        for cell in self.coefs.index:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            wt = self.coefs.loc[cell].reshape(-1, 1)
            bias = self.intercepts.loc[cell]
            y = self.binnedspikes[testmask, cell_idx]
            pred = (testdm @ wt + bias).flatten()
            if dsq:
                null_pred = np.ones_like(pred) * np.mean(y)
                null_deviance = 2 * np.sum(xlogy(y, y / null_pred.flat) - y + null_pred.flat)
                with np.errstate(invalid='ignore', divide='ignore'):
                    full_deviance = 2 * np.sum(xlogy(y, y / pred.flat) - y + pred.flat)
                d_sq = 1 - (full_deviance / null_deviance)
                scores.at[cell] = d_sq
            elif msespike:
                residuals = (y - pred) ** 2
                msespike = residuals.sum() / y.sum()
                scores.at[cell] = msespike
            else:
                scores.at[cell] = r2_score(y, pred)
        return scores

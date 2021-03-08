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


class PoissonGLM(NeuralModel):
    """
    Linear-nonlinear poisson model, a type of generalized linear model, for neural responses.
    """
    def _fit_sklearn(self, dm, binned, alpha, cells=None, retvar=False, noncovwarn=True,
                     fit_intercept=True):
        """
        Fit a GLM using scikit-learn implementation of PoissonRegressor. Uses a regularization
        strength parameter alpha, which is the strength of ridge regularization term. When alpha
        is set to 0, this is the same as _fit_minimize, but without the use of the hessian for
        fitting.

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
        variances : bool
            Whether or not to return variances on parameters in dm.
        """
        if cells is None:
            cells = self.clu_ids.flatten()
        if not fit_intercept:
            retvar = False
        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts')
        variances = pd.Series(index=cells, name='variances', dtype=object)
        nonconverged = []
        for cell in tqdm(cells, 'Fitting units:', leave=False):
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            cellbinned = binned[:, cell_idx]
            with catch_warnings(record=True) as w:
                fitobj = PoissonRegressor(alpha=alpha,
                                          max_iter=300,
                                          fit_intercept=fit_intercept).fit(dm, cellbinned)
            if len(w) != 0:
                nonconverged.append(cell)
            wts = np.concatenate([[fitobj.intercept_], fitobj.coef_], axis=0)
            biasdm = np.pad(dm.copy(), ((0, 0), (1, 0)), 'constant', constant_values=1)
            if retvar:
                wvar = np.diag(np.linalg.inv(dd_neglog(wts, biasdm, cellbinned)))
            else:
                wvar = np.ones((wts.shape[0], wts.shape[0])) * np.nan
            coefs.at[cell] = fitobj.coef_
            variances.at[cell] = wvar[1:]
            if fit_intercept:
                intercepts.at[cell] = fitobj.intercept_
            else:
                intercepts.at[cell] = 0
        if noncovwarn:
            if len(nonconverged) != 0:
                warn(f'Fitting did not converge for some units: {nonconverged}')
        
        return coefs, intercepts, variances

    def _fit_minimize(self, dm, binned, cells=None, retvar=False):
        """
        Fit a GLM using direct minimization of the negative log likelihood. No regularization.

        Parameters
        ----------
        dm : numpy.ndarray
            Design matrix, in which rows are observations and columns are regressor values. First
            column must be a bias column of ones.
        binned : numpy.ndarray
            Vector of observed spike counts which we seek to predict. Must be of the same length
            as dm.shape[0]
        cells : list
            List of cells which should be fit. If None is passed, will default to fitting all cells
            in clu_ids
        variances : bool
            Whether or not to return variances on parameters in dm.
        """
        if cells is None:
            cells = self.clu_ids.flatten()
        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts')
        variances = pd.Series(index=cells, name='variances', dtype=object)
        for cell in tqdm(cells, 'Fitting units:', leave=False):
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            cellbinned = binned[:, cell_idx]
            wi = np.linalg.lstsq(dm, cellbinned, rcond=None)[0]
            res = minimize(neglog, wi, (dm, cellbinned),
                           method='trust-ncg', jac=d_neglog, hess=dd_neglog)
            if retvar:
                hess = dd_neglog(res.x, dm, cellbinned)
                try:
                    wvar = np.diag(np.linalg.inv(hess))
                except LinAlgError:
                    wvar = np.ones_like(np.diag(hess)) * np.inf
            else:
                wvar = np.ones((res.x.shape[0], res.x.shape[0])) * np.nan
            coefs.at[cell] = res.x[1:]
            intercepts.at[cell] = res.x[0]
            variances.at[cell] = wvar[1:]
        return coefs, intercepts, variances

    def fit(self, method='sklearn', alpha=0, singlepar_var=False,
            fit_intercept=True, printcond=True, dsq=False, rsq=False):
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
        if method not in ('sklearn', 'minimize', 'pytorch'):
            raise ValueError('Method must be \'minimize\' or \'sklearn\' or \'pytorch\'')
        # TODO: Make this optionally parallel across multiple cores of CPU
        # Initialize pd Series to store output coefficients and intercepts for fits
        trainmask = np.isin(self.trlabels, self.traininds).flatten()  # Mask for training data
        trainbinned = self.binnedspikes[trainmask]
        if printcond:
            print(f'Condition of design matrix is {np.linalg.cond(self.dm[trainmask])}')

        if not self.subset:
            if method == 'sklearn':
                traindm = self.dm[trainmask]
                coefs, intercepts, variances = self._fit_sklearn(traindm, trainbinned, alpha,
                                                                 fit_intercept=fit_intercept,
                                                                 retvar=True)
            else:
                biasdm = np.pad(self.dm.copy(), ((0, 0), (1, 0)), 'constant', constant_values=1)
                traindm = biasdm[trainmask]
                coefs, intercepts, variances = self._fit_minimize(traindm, trainbinned,
                                                                  retvar=True)
            self.coefs = coefs
            self.intercepts = intercepts
            self.variances = variances
            self.fitmethod = method
            return
        else:
            # Get testing matrices for scoring in submodels
            testmask = np.isin(self.trlabels, self.testinds).flatten()
            testbinned = self.binnedspikes[testmask]

            # Build single-parameter-group models first:
            singlepar_models = {}
            singlepar_scores = pd.DataFrame(columns=['cell', 'covar', 'scores'])
            for cov in tqdm(self.covar, desc='Fitting single-cov models:', leave=False):
                dmcols = self.covar[cov]['dmcol_idx']
                colmask = np.zeros(self.dm.shape[1], dtype=bool)
                colmask[dmcols] = True
                traindm = self.dm[np.ix_(trainmask, colmask)]
                testdm = self.dm[np.ix_(testmask, colmask)]
                if method == 'sklearn':
                    coefs, intercepts, variances = self._fit_sklearn(traindm, trainbinned, alpha,
                                                                     fit_intercept=fit_intercept,
                                                                     retvar=singlepar_var)
                else:
                    biasdm = np.pad(traindm.copy(), ((0, 0), (1, 0)), 'constant',
                                    constant_values=1)
                    coefs, intercepts, variances = self._fit_minimize(biasdm, trainbinned,
                                                                      retvar=singlepar_var)
                scores = self._score_submodel(coefs, intercepts, testdm, testbinned, dsq=dsq,
                                              rsq=rsq)
                scoresdf = pd.DataFrame(scores).reset_index()
                scoresdf.rename(columns={'index': 'cell'}, inplace=True)
                scoresdf['covar'] = cov
                if singlepar_var:
                    singlepar_models[cov] = (coefs, intercepts, variances, scores)
                else:
                    singlepar_models[cov] = (coefs, intercepts, np.nan, scores)
                singlepar_scores = pd.concat([singlepar_scores, scoresdf], sort=False)
            singlepar_scores.set_index(['cell', 'covar'], inplace=True)
            singlepar_scores.sort_values(by=['cell', 'scores'], ascending=False, inplace=True)
            fitcells = singlepar_scores.index.levels[0]
            # Iteratively build model with 2 through K parameter groups:
            submodel_scores = singlepar_scores.unstack()
            submodel_scores.columns = submodel_scores.columns.droplevel()
            for i in tqdm(range(2, len(self.covar) + 1), desc='Fitting submodels', leave=False):
                cellcovars = {cell: tuple(singlepar_scores.loc[cell].iloc[:i].index)
                              for cell in fitcells}
                covarsets = [frozenset(cov) for cov in cellcovars.values()]
                retvar = True if i == len(self.covar) else False
                # Iterate through unique unordered combinations of covariates
                iscores = []
                progressdesc = f'Fitting covariate sets for {i} parameter groups'
                for covarset in tqdm(set(covarsets), desc=progressdesc, leave=False):
                    currcells = [cell for cell, covar in cellcovars.items()
                                 if set(covar) == covarset]
                    currcols = np.hstack([self.covar[cov]['dmcol_idx'] for cov in covarset])
                    colmask = np.zeros(self.dm.shape[1], dtype=bool)
                    colmask[currcols] = True
                    traindm = self.dm[np.ix_(trainmask, colmask)]
                    testdm = self.dm[np.ix_(testmask, colmask)]
                    if method == 'sklearn':
                        coefs, intercepts, variances = self._fit_sklearn(traindm, trainbinned,
                                                                         alpha, cells=currcells,
                                                                         retvar=retvar,
                                                                         noncovwarn=False)
                    else:
                        biasdm = np.pad(traindm.copy(), ((0, 0), (1, 0)), 'constant',
                                        constant_values=1)
                        coefs, intercepts, variances = self._fit_minimize(biasdm, trainbinned,
                                                                          cells=currcells,
                                                                          retvar=retvar)
                    iscores.append(self._score_submodel(coefs, intercepts, testdm, testbinned,
                                                        dsq=dsq, rsq=rsq))
                submodel_scores[f'{i}cov'] = pd.concat(iscores).sort_index()
            self.submodel_scores = submodel_scores
            self.coefs = coefs
            self.intercepts = intercepts
            self.variances = variances
            return

    def _score_submodel(self, weights, intercepts, dm, binned, dsq=False, rsq=False):
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

        Returns
        -------
        pd.Series
            Pandas series containing the scores of the given model for each cell.
        """
        scores = pd.Series(index=weights.index, name='scores')
        biasdm = np.pad(dm, ((0, 0), (1, 0)), constant_values=1)
        for cell in weights.index:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            wt = weights.loc[cell].reshape(-1, 1)
            bias = intercepts.loc[cell]
            y = binned[:, cell_idx]
            pred = np.exp(dm @ wt + bias)
            if dsq:
                null_pred = np.ones_like(pred) * np.mean(y)
                with np.errstate(divide='ignore', invalid='ignore'):
                    null_deviance = 2 * np.sum(xlogy(y, y / null_pred.flat) - y + null_pred.flat)
                    full_deviance = 2 * np.sum(xlogy(y, y / pred.flat) - y + pred.flat)
                d_sq = 1 - (full_deviance / null_deviance)
                scores.at[cell] = d_sq
            elif rsq:
                scores.at[cell] = r2_score(y, pred)
            else:
                scores.at[cell] = -neglog(np.vstack((bias, wt)).flatten(), biasdm, y) / np.sum(y)
        return scores

    def combine_weights(self):
        """
        Combined fit coefficients and intercepts to produce kernels where appropriate, which
        describe activity.

        Returns
        -------
        pandas.DataFrame
            DataFrame in which each row is the fit weights for a given spiking unit. Columns are
            individual covariates added during the construction process. Indices are the cluster
            IDs for each of the cells that were fit (NOT a simple range(start, stop) index.)
        """
        outputs = {}
        varoutputs = {}
        for var in self.covar.keys():
            if self.covar[var]['bases'] is None:
                wind = self.covar[var]['dmcol_idx']
                outputs[var] = self.coefs.apply(lambda w: w[wind])
                continue
            winds = self.covar[var]['dmcol_idx']
            bases = self.covar[var]['bases']
            weights = self.coefs.apply(lambda w: np.sum(w[winds] * bases, axis=1))
            if hasattr(self, 'variances'):
                variances = self.variances.apply(lambda v: np.sum(v[winds] * bases, axis=1))
            offset = self.covar[var]['offset']
            tlen = bases.shape[0] * self.binwidth
            tstamps = np.linspace(0 + offset, tlen + offset, bases.shape[0])
            outputs[var] = pd.DataFrame(weights.values.tolist(),
                                        index=weights.index,
                                        columns=tstamps)
            if hasattr(self, 'variances'):
                varoutputs[var] = pd.DataFrame(variances.values.tolist(),
                                               index=weights.index,
                                               columns=tstamps)
        self.combined_weights = outputs
        if hasattr(self, 'variances'):
            self.combined_variances = varoutputs
        return outputs

    def score(self, dsq=False, rsq=False):
        """
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
        if not hasattr(self, 'coefs'):
            raise AttributeError('Fit was not run. Please run fit first.')
        if hasattr(self, 'submodel_scores'):
            return self.submodel_scores
        testmask = np.isin(self.trlabels, self.testinds).flatten()
        testdm = self.dm[testmask, :]
        biasdm = np.pad(testdm, ((0, 0), (1, 0)), constant_values=1)
        scores = pd.Series(index=self.coefs.index)
        for cell in self.coefs.index:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            wt = self.coefs.loc[cell].reshape(-1, 1)
            bias = self.intercepts.loc[cell]
            y = self.binnedspikes[testmask, cell_idx]
            if dsq or rsq:
                pred = np.exp(testdm @ wt + bias)
                if dsq:
                    null_pred = np.ones_like(pred) * np.mean(y)
                    null_deviance = 2 * np.sum(xlogy(y, y / null_pred.flat) - y + null_pred.flat)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        full_deviance = 2 * np.sum(xlogy(y, y / pred.flat) - y + pred.flat)
                    d_sq = 1 - (full_deviance / null_deviance)
                    scores.at[cell] = d_sq
                else:
                    scores.at[cell] = r2_score(y, pred)
            else:
                scores.at[cell] = -neglog(np.vstack((bias, wt)).flatten(), biasdm, y) / np.sum(y)
        return scores

    def binf(self, t):
        """
        Bin function for a given timestep. Returns the number of bins after trial start a given t
        would occur at.

        Parameters
        ----------
        t : float
            Seconds after trial start

        Returns
        -------
        int
            Number of bins corresponding to t using the binwidth of the model.
        """
        return np.ceil(t / self.binwidth).astype(int)


def convbasis(stim, bases, offset=0):
    if offset < 0:
        stim = np.pad(stim, ((0, -offset), (0, 0)), 'constant')
    elif offset > 0:
        stim = np.pad(stim, ((offset, 0), (0, 0)), 'constant')

    X = denseconv(stim, bases)

    if offset < 0:
        X = X[-offset:, :]
    elif offset > 0:
        X = X[: -(1 + offset), :]
    return X


# Precompilation for speed
@nb.njit
def denseconv(X, bases):
    T, dx = X.shape
    TB, M = bases.shape
    indices = np.ones((dx, M))
    sI = np.sum(indices, axis=1)
    BX = np.zeros((T, int(np.sum(sI))))
    sI = np.cumsum(sI)
    k = 0
    for kCov in range(dx):
        A = np.zeros((T + TB - 1, int(np.sum(indices[kCov, :]))))
        for i, j in enumerate(np.argwhere(indices[kCov, :]).flat):
            A[:, i] = np.convolve(X[:, kCov], bases[:, j])
        BX[:, k: sI[kCov]] = A[: T, :]
        k = sI[kCov]
    return BX


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


def d_neglog(weights, x, y):
    xproj = x @ weights
    f = np.exp(xproj)
    df = f
    nzidx = (f != 0).reshape(-1)
    if np.any(y[~nzidx] != 0):
        return np.inf
    return x[nzidx, :].T @ ((1 - y[nzidx] / f[nzidx]) * df[nzidx])


def dd_neglog(weights, x, y):
    xproj = x @ weights
    f = np.exp(xproj)
    df = f
    ddf = df
    nzidx = (f != 0).reshape(-1)
    if np.any(y[~nzidx] != 0):
        return np.inf
    yf = y[nzidx] / f[nzidx]
    p1 = ddf[nzidx] * (1 - yf) + (y[nzidx] * (df[nzidx] / f[nzidx])**2)
    p2 = x[nzidx, :]
    return (p1.reshape(-1, 1) * p2).T @ x[nzidx, :]


def dd_neglog_cp(weights, x, y):
    weights = cp.array(weights)
    x = cp.array(x)
    y = cp.array(y)
    xproj = x @ weights
    f = cp.exp(xproj)
    df = f
    ddf = df
    nzidx = (f != 0).reshape(-1)
    if np.any(y[~nzidx] != 0):
        return np.inf
    yf = y[nzidx] / f[nzidx]
    p1 = ddf[nzidx] * (1 - yf) + (y[nzidx] * (df[nzidx] / f[nzidx])**2)
    p2 = x[nzidx, :]
    return (p1.reshape(-1, 1) * p2).T @ x[nzidx, :]
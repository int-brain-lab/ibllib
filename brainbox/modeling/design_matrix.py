from warnings import warn
import numpy as np
import pandas as pd
import scipy.sparse as sp
import numba as nb


class DesignMatrixConstructor:
    """
    Design matrix constructor that will take in information about the temporal structure of a trial
    and allow the generation of a design matrix with specified regressors
    """

    def __init__(self, trialsdf, vartypes, binwidth=0.02):
        """
        I'll tell you more later, buddy
        """
        # Data checks #
        if not all([name in vartypes for name in trialsdf.columns]):
            raise KeyError("Some columns were not described in vartypes")
        if not all([value in ('timing', 'continuous', 'value') for value in vartypes.values()]):
            raise ValueError("Invalid values were passed in vartypes")

        # Filter out cells which don't meet the criteria for minimum spiking, while doing trial
        # assignment
        self.vartypes = vartypes
        self.vartypes['duration'] = 'value'
        trialsdf = trialsdf.copy()  # Make sure we don't modify the original dataframe
        trbounds = trialsdf[['trial_start', 'trial_end']]  # Get the start/end of trials
        # Empty trial duration value to use later
        trialsdf['duration'] = np.nan
        timingvars = [col for col in trialsdf.columns if vartypes[col] == 'timing']
        for i, (start, end) in trbounds.iterrows():
            if any(np.isnan((start, end))):
                warn(f"NaN values found in trial start or end at trial number {i}. "
                     "Discarding trial.")
                trialsdf.drop(i, inplace=True)
                continue
            for col in timingvars:
                trialsdf.at[i, col] = np.round(trialsdf.at[i, col] - start, decimals=5)
            trialsdf.at[i, 'duration'] = end - start

        # Set model parameters to begin with
        self.binwidth = binwidth
        self.covar = {}
        self.trialsdf = trialsdf
        self.compiled = False
        return

    def add_covariate_timing(self, covlabel, eventname, bases,
                             offset=0, deltaval=None, cond=None, desc=''):
        """
        Convenience wrapper for adding timing event regressors to the GLM. Automatically generates
        a one-hot vector for each trial as the regressor and adds the appropriate data structure
        to the model.

        Parameters
        ----------
        covlabel : str
            Label which the covariate will use. Can be accessed via dot syntax of the instance
            usually.
        eventname : str
            Label of the column in trialsdf which has the event timing for each trial.
        bases : numpy.array
            nTB x nB array, i.e. number of time bins for the bases functions by number of bases.
            Each column in the array is used together to describe the response of a unit to that
            timing event.
        offset : float, seconds
            Offset of bases functions relative to timing event. Negative values will ensure that

        deltaval : None, str, or pandas series, optional
            Values of the kronecker delta function peak used to encode the event. If a string, the
            column in trialsdf with that label will be used. If a pandas series with indexes
            matching trialsdf, corresponding elements of the series will be the delta funtion val.
            If None (default) height is 1.
        cond : None, list, or fun, optional
            Condition which to apply this covariate. Can either be a list of trial indices, or a
            function which takes in rows of the trialsdf and returns booleans.
        desc : str, optional
            Additional information about the covariate, if desired. by default ''
        """
        if covlabel in self.covar:
            raise AttributeError(f'Covariate {covlabel} already exists in model.')
        self._compile_check()
        if deltaval is None:
            gainmod = False
        elif isinstance(deltaval, pd.Series):
            gainmod = True
        elif isinstance(deltaval, str) and deltaval in self.trialsdf.columns:
            gainmod = True
            deltaval = self.trialsdf[deltaval]
        else:
            raise TypeError('deltaval must be None, pandas series, or string reference'
                            f' to trialsdf column. {type(deltaval)} was passed instead.')
        if eventname not in self.vartypes:
            raise ValueError('Event name specified not found in trialsdf')
        elif self.vartypes[eventname] != 'timing':
            raise TypeError(f'Column {eventname} in trialsdf is not registered as a timing')

        vecsizes = self.trialsdf['duration'].apply(self.binf)
        stiminds = self.trialsdf[eventname].apply(self.binf)
        stimvecs = []
        for i in self.trialsdf.index:
            vec = np.zeros(vecsizes[i])
            if gainmod:
                vec[stiminds[i]] = deltaval[i]
            else:
                vec[stiminds[i]] = 1
            stimvecs.append(vec.reshape(-1, 1))
        regressor = pd.Series(stimvecs, index=self.trialsdf.index)
        self.add_covariate(covlabel, regressor, bases, offset, cond, desc)
        return

    def add_covariate_boxcar(self, covlabel, boxstart, boxend,
                             cond=None, height=None, desc=''):
        if covlabel in self.covar:
            raise AttributeError(f'Covariate {covlabel} already exists in model.')
        self._compile_check()
        if boxstart not in self.trialsdf.columns or boxend not in self.trialsdf.columns:
            raise KeyError('boxstart or boxend not found in trialsdf columns.')
        if self.vartypes[boxstart] != 'timing':
            raise TypeError(f'Column {boxstart} in trialsdf is not registered as a timing. '
                            'boxstart and boxend need to refer to timing events in trialsdf.')
        if self.vartypes[boxend] != 'timing':
            raise TypeError(f'Column {boxend} in trialsdf is not registered as a timing. '
                            'boxstart and boxend need to refer to timing events in trialsdf.')

        if isinstance(height, str):
            if height in self.trialsdf.columns:
                height = self.trialsdf[height]
            else:
                raise KeyError(f'{height} is str not in columns of trialsdf')
        elif isinstance(height, pd.Series):
            if not all(height.index == self.trialsdf.index):
                raise IndexError('Indices of height series does not match trialsdf.')
        elif height is None:
            height = pd.Series(np.ones(len(self.trialsdf.index)), index=self.trialsdf.index)
        vecsizes = self.trialsdf['duration'].apply(self.binf)
        stind = self.trialsdf[boxstart].apply(self.binf)
        endind = self.trialsdf[boxend].apply(self.binf)
        stimvecs = []
        for i in self.trialsdf.index:
            bxcar = np.zeros(vecsizes[i])
            bxcar[stind[i]:endind[i] + 1] = height[i]
            stimvecs.append(bxcar)
        regressor = pd.Series(stimvecs, index=self.trialsdf.index)
        self.add_covariate(covlabel, regressor, None, cond, desc)
        return

    def add_covariate_raw(self, covlabel, raw,
                          cond=None, desc=''):
        stimlens = self.trialsdf.duration.apply(self.binf)
        if isinstance(raw, str):
            if raw not in self.trialsdf.columns:
                raise KeyError(f'String {raw} not found in columns of trialsdf. Strings must'
                               'refer to valid column names.')
            covseries = self.trialsdf[raw]
            if np.any(covseries.apply(len) != stimlens):
                raise IndexError(f'Some array shapes in {raw} do not match binned duration.')
            self.add_covariate(covlabel, covseries, None, cond=cond)

        if callable(raw):
            try:
                covseries = self.trialsdf.apply(raw, axis=1)
            except Exception:
                raise TypeError('Function for raw covariate generation did not run properly.'
                                'Make sure that the function passed takes in rows of trialsdf.')
            if np.any(covseries.apply(len) != stimlens):
                raise IndexError(f'Some array shapes in {raw} do not match binned duration.')
            self.add_covariate(covlabel, covseries, None, cond=cond)

        if isinstance(raw, pd.Series):
            if np.any(raw.index != self.trialsdf.index):
                raise IndexError('Indices of raw do not match indices of trialsdf.')
            if np.any(raw.apply(len) != stimlens):
                raise IndexError(f'Some array shapes in {raw} do not match binned duration.')
            self.add_covariate(covlabel, raw, None, cond=cond)

    def add_covariate(self, covlabel, regressor, bases,
                      offset=0, cond=None, desc=''):
        """
        Parent function to add covariates to model object. Takes a regressor in the form of a
        pandas Series object, a T x M array of M bases, and stores them for use in the design
        matrix generation.

        Parameters
        ----------
        covlabel : str
            Label for the covariate being added. Will be exposed, if possible, through
            (instance).(covlabel) attribute.
        regressor : pandas.Series
            Series in which each element is the value(s) of a regressor for a trial at that index.
            These will be convolved with the bases functions (if provided) to produce the
            components of the design matrix. *Regressor must be (T / dt) x 1 array for each trial*
        bases : numpy.array or None
            T x M array of M basis functions over T timesteps. Columns will be convolved with the
            elements of `regressor` to produce elements of the design matrix. If None, it is
            assumed a raw regressor is being used.
        offset : int, optional
            Offset of the regressor from the bases during convolution. Negative values indicate
            that the firing of the unit will be , by default 0
        cond : list or func, optional
            Condition for which to apply covariate. Either a list of trials which the covariate
            applies to, or a function of the form f(dataframerow) which returns a boolean,
            by default None
        desc : str, optional
            Description of the covariate for reference purposes, by default '' (empty)
        """
        if covlabel in self.covar:
            raise AttributeError(f'Covariate {covlabel} already exists in model.')
        self._compile_check()
        # Test for mismatch in length of regressor vs trials
        mismatch = np.zeros(len(self.trialsdf.index), dtype=bool)
        for i in self.trialsdf.index:
            currtr = self.trialsdf.loc[i]
            nT = self.binf(currtr.duration)
            if regressor.loc[i].shape[0] != nT:
                mismatch[i] = True

        if np.any(mismatch):
            raise ValueError('Length mismatch between regressor and trial on trials'
                             f'{np.argwhere(mismatch)}.')

        # Initialize containers for the covariate dicts
        if not hasattr(self, 'currcol'):
            self.currcol = 0
        if callable(cond):
            cond = self.trialsdf.index[self.trialsdf.apply(cond, axis=1)].to_numpy()
        if not all(regressor.index == self.trialsdf.index):
            raise IndexError('Indices of regressor and trials dataframes do not match.')

        cov = {'description': desc,
               'bases': bases,
               'valid_trials': cond if cond is not None else self.trialsdf.index,
               'offset': offset,
               'regressor': regressor,
               'dmcol_idx': np.arange(self.currcol, self.currcol + bases.shape[1])
               if bases is not None else self.currcol}
        if bases is None:
            self.currcol += 1
        else:
            self.currcol += bases.shape[1]

        self.covar[covlabel] = cov
        return

    def compile_design_matrix(self, dense=True):
        """
        Compiles design matrix for the current experiment based on the covariates which were added
        with the various NeuralGLM.add_covariate methods available. Can optionally compile a sparse
        design matrix using the scipy.sparse package, however that method may take longer depending
        on the degree of sparseness.

        Parameters
        ----------
        dense : bool, optional
            Whether or not to compute a dense design matrix or a sparse one, by default True
        """
        covars = self.covar
        # Go trial by trial and compose smaller design matrices
        miniDMs = []
        rowtrials = []
        for i, trial in self.trialsdf.iterrows():
            nT = self.binf(trial.duration)
            miniX = np.zeros((nT, self.currcol))
            rowlabs = np.ones((nT, 1), dtype=int) * i
            for cov in covars.values():
                sidx = cov['dmcol_idx']
                # Optionally use cond to filter out which trials to apply certain regressors,
                if i not in cov['valid_trials']:
                    continue
                stim = cov['regressor'][i]
                # Convolve Kernel or basis function with stimulus or regressor
                if cov['bases'] is None:
                    miniX[:, sidx] = stim
                else:
                    if len(stim.shape) == 1:
                        stim = stim.reshape(-1, 1)
                    miniX[:, sidx] = convbasis(stim, cov['bases'], self.binf(cov['offset']))
            # Sparsify convolved result and store in miniDMs
            if dense:
                miniDMs.append(miniX)
            else:
                miniDMs.append(sp.lil_matrix(miniX))
            rowtrials.append(rowlabs)
        if dense:
            dm = np.vstack(miniDMs)

        else:
            dm = sp.vstack(miniDMs).to_csc()
        trlabels = np.vstack(rowtrials)
        if hasattr(self, 'binnedspikes'):
            assert self.binnedspikes.shape[0] == dm.shape[0], "Oh shit. Indexing error."
        self.dm = dm
        self.trlabels = trlabels
        # self.dm = np.roll(dm, -1, axis=0)  # Fix weird +1 offset bug in design matrix
        self.compiled = True
        return

    def _compile_check(self):
        if self.compiled:
            warn('Design matrix was already compiled once. Be sure to compile again if adding'
                 ' additional covariates.')
        return


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

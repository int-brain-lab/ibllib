
"""
GLM fitting utilities based on NeuroGLM by Il Memming Park, Jonathan Pillow:

https://github.com/pillowlab/neuroGLM

Berk Gercek
International Brain Lab, 2020
"""
from warnings import warn, catch_warnings
import numpy as np
from numpy.linalg.linalg import LinAlgError
import pandas as pd
from brainbox.processing import bincount2D
from sklearn.linear_model import PoissonRegressor
import scipy.sparse as sp
import numba as nb
from numpy.matlib import repmat
from scipy.optimize import minimize
from scipy.special import xlogy
from tqdm import tqdm
import torch
from brainbox.modeling.poissonGLM import PoissonGLM


class NeuralGLM:
    """
    Generalized Linear Model which seeks to describe spiking activity as the output of a poisson
    process. Uses sklearn's GLM methods under the hood while providing useful routines for dealing
    with neural data
    """

    def __init__(self, trialsdf, spk_times, spk_clu, vartypes,
                 train=0.8, blocktrain=False, binwidth=0.02, mintrials=100, subset=False):
        """
        Construct GLM object using information about all trials, and the relevant spike times.
        Only ingests data, and further object methods must be called to describe kernels, gain
        terms, etc. as components of the model.

        Parameters
        ----------
        trialsdf: pandas.DataFrame
            DataFrame of trials in which each row contains all desired covariates of the model.
            e.g. contrast, stimulus type, etc. Not all columns will necessarily be fit.
            If a continuous covariate (e.g. wheel position, pupil diameter) is included, each entry
            of the column must be a nSamples x 2 array with samples in the first column and
            timestamps (relative to trial start) in the second position.
            *Must have \'trial_start\' and \'trial_end\' parameters which are times, in seconds.*
        spk_times: numpy.array of floats
            1-D array of times at which spiking events were detected, in seconds.
        spk_clu: numpy.array of integers
            1-D array of same shape as spk_times, with integer cluster IDs identifying which
            cluster a spike time belonged to.
        vartypes: dict
            Dict with column names in trialsdf as keys, values are the type of covariate the column
            contains. e.g. {'stimOn_times': 'timing', 'wheel', 'continuous', 'correct': 'value'}
            Valid values are:
                'timing' : A timestamp relative to trial start (e.g. stimulus onset)
                'continuous' : A continuous covariate sampled throughout the trial (e.g. eye pos)
                'value' : A single value for the given trial (e.g. contrast or difficulty)
        train: float
            Float in (0, 1] indicating proportion of data to use for training GLM vs testing
            (using the NeuralGLM.score method). Trials to keep will be randomly sampled.
        binwidth: float
            Width, in seconds, of the bins which will be used to count spikes. Defaults to 20ms.
        mintrials: int
            Minimum number of trials in which neurons fired a spike in order to be fit. Defaults
            to 100 trials.
        subset: bool
            Whether or not to perform model subsetting, in which the model is built iteratively
            from only the mean rate, up. This allows comparison of D^2 scores for sub-models which
            incorporate only some parameters, to see which regressors actually improve
            explainability. Default to False.

        Returns
        -------
        glm: object
            GLM object with methods for adding regressors and fitting
        """
        # Data checks #
        if not all([name in vartypes for name in trialsdf.columns]):
            raise KeyError("Some columns were not described in vartypes")
        if not all([value in ('timing', 'continuous', 'value') for value in vartypes.values()]):
            raise ValueError("Invalid values were passed in vartypes")
        if not len(spk_times) == len(spk_clu):
            raise IndexError("Spike times and cluster IDs are not same length")
        if not isinstance(train, float) and not train == 1:
            raise TypeError('train must be a float between 0 and 1')
        if not ((train > 0) & (train <= 1)):
            raise ValueError('train must be between 0 and 1')

        # Filter out cells which don't meet the criteria for minimum spiking, while doing trial
        # assignment
        self.vartypes = vartypes
        self.vartypes['duration'] = 'value'
        trialsdf = trialsdf.copy()  # Make sure we don't modify the original dataframe
        clu_ids = np.unique(spk_clu)
        trbounds = trialsdf[['trial_start', 'trial_end']]  # Get the start/end of trials
        # Initialize a Cells x Trials bool array to easily see how many trials a clu spiked
        trialspiking = np.zeros((trialsdf.index.max() + 1, clu_ids.max() + 1), dtype=bool)
        # Empty trial duration value to use later
        trialsdf['duration'] = np.nan
        # Iterate through each trial, and store the relevant spikes for that trial into a dict
        # Along with the cluster labels. This makes binning spikes and accessing spikes easier.
        spks = {}
        clu = {}
        st_endlast = 0
        timingvars = [col for col in trialsdf.columns if vartypes[col] == 'timing']
        for i, (start, end) in trbounds.iterrows():
            if any(np.isnan((start, end))):
                warn(f"NaN values found in trial start or end at trial number {i}. "
                     "Discarding trial.")
                trialsdf.drop(i, inplace=True)
                continue
            st_startind = np.searchsorted(spk_times[st_endlast:], start) + st_endlast
            st_endind = np.searchsorted(spk_times[st_endlast:], end, side='right') + st_endlast
            st_endlast = st_endind
            trial_clu = np.unique(spk_clu[st_startind:st_endind])
            trialspiking[i, trial_clu] = True
            spks[i] = spk_times[st_startind:st_endind] - start
            clu[i] = spk_clu[st_startind:st_endind]
            for col in timingvars:
                trialsdf.at[i, col] = np.round(trialsdf.at[i, col] - start, decimals=5)
            trialsdf.at[i, 'duration'] = end - start

        # Break the data into test and train sections for cross-validation
        if train == 1:
            print('Training fraction set to 1. Training on all data.')
            traininds = trialsdf.index
            testinds = trialsdf.index
        elif blocktrain:
            trainlen = int(np.floor(len(trialsdf) * train))
            traininds = trialsdf.index[:trainlen]
            testinds = trialsdf.index[trainlen:]
        else:
            trainlen = int(np.floor(len(trialsdf) * train))
            traininds = sorted(np.random.choice(trialsdf.index, trainlen, replace=False))
            testinds = trialsdf.index[~trialsdf.index.isin(traininds)]

        # Set model parameters to begin with
        self.spikes = spks
        self.clu = clu
        self.clu_ids = np.argwhere(np.sum(trialspiking, axis=0) > mintrials)
        self.binwidth = binwidth
        self.covar = {}
        self.trialsdf = trialsdf
        self.traininds = traininds
        self.testinds = testinds
        self.compiled = False
        self.subset = subset
        if len(self.clu_ids) == 0:
            raise UserWarning('No neuron fired a spike in a minimum number.')

        # Bin spikes
        self._bin_spike_trains()
        return

    def _bin_spike_trains(self):
        """
        Bins spike times passed to class at instantiation. Will not bin spike trains which did
        not meet the criteria for minimum number of spiking trials. Must be run before the
        NeuralGLM.fit() method is called.
        """
        spkarrs = []
        arrdiffs = []
        for i in self.trialsdf.index:
            duration = self.trialsdf.loc[i, 'duration']
            durmod = duration % self.binwidth
            if durmod > (self.binwidth / 2):
                duration = duration - (self.binwidth / 2)
            if len(self.spikes[i]) == 0:
                arr = np.zeros((self.binf(duration), len(self.clu_ids)))
                spkarrs.append(arr)
                continue
            spks = self.spikes[i]
            clu = self.clu[i]
            arr = bincount2D(spks, clu,
                             xbin=self.binwidth, ybin=self.clu_ids, xlim=[0, duration])[0]
            arrdiffs.append(arr.shape[1] - self.binf(duration))
            spkarrs.append(arr.T)
        y = np.vstack(spkarrs)
        if hasattr(self, 'dm'):
            assert y.shape[0] == self.dm.shape[0], "Oh shit. Indexing error."
        self.binnedspikes = y
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
        if self.compiled:
            warn('Design matrix was already compiled once. Be sure to compile again if adding'
                 ' additional covariates.')
        if deltaval is None:
            gainmod = False
        elif isinstance(deltaval, pd.Series):
            gainmod = True
        elif isinstance(deltaval, str) and deltaval in self.trialsdf.columns:
            gainmod = True
            deltaval = self.trialsdf[deltaval]
        else:
            raise TypeError(f'deltaval must be None or pandas series. {type(deltaval)} '
                            'was passed instead.')
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
        if self.compiled:
            warn('Design matrix was already compiled once. Be sure to compile again if adding'
                 ' additional covariates.')
        if boxstart not in self.trialsdf.columns or boxend not in self.trialsdf.columns:
            raise KeyError('boxstart or boxend not found in trialsdf columns.')
        if self.vartypes[boxstart] != 'timing':
            raise TypeError(f'Column {boxstart} in trialsdf is not registered as a timing. '
                            'boxstart and boxend need to refer to timining events in trialsdf.')
        if self.vartypes[boxend] != 'timing':
            raise TypeError(f'Column {boxend} in trialsdf is not registered as a timing. '
                            'boxstart and boxend need to refer to timining events in trialsdf.')

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
        if self.compiled:
            warn('Design matrix was already compiled once. Be sure to compile again if adding'
                 ' additional covariates.')
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

    def _fit_sklearn(self, dm, binned, alpha, cells=None, retvar=False, noncovwarn=True):
        """
        Fit a GLM using scikit-learn implementation of PoissonRegressor. Uses a regularization
        strength parameter alpha, which is the strength of ridge regularization term. When alpha
        is set to 0, this *should* in theory be the same as _fit_minimize, but in practice it is
        not and seems to exhibit some regularization still.

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
        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts')
        variances = pd.Series(index=cells, name='variances', dtype=object)
        nonconverged = []
        for cell in tqdm(cells, 'Fitting units:', leave=False):
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            cellbinned = binned[:, cell_idx]
            with catch_warnings(record=True) as w:
                fitobj = PoissonRegressor(alpha=alpha, max_iter=300).fit(dm,
                                                                         cellbinned)
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
            intercepts.at[cell] = fitobj.intercept_
        if noncovwarn:
            if len(nonconverged) != 0:
                warn(f'Fitting did not converge for some units: {nonconverged}')
        return coefs, intercepts, variances

    def _fit_pytorch(self, dm, binned, cells=None, retvar=False, epochs=500, optim='adam',
                     lr=1.0):
        """
        Fit the GLM using PyTorch on GPU(s). Regularization has not been applied yet.

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
        epochs : int
            The number of epochs to train the model
        optim : string
            The name of optimization method in pytorch
        lr : float
            Learning rate for the optimizer
        """
        if cells is None:
            cells = self.clu_ids.flatten()
        coefs = pd.Series(index=cells, name='coefficients', dtype=object)
        intercepts = pd.Series(index=cells, name='intercepts')
        variances = pd.Series(index=cells, name='variances', dtype=object)

        # CPU or GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # device = torch.device('cpu')
        in_features = dm.shape[1]
        out_features = binned.shape[1]
        glm = PoissonGLM(in_features, out_features).to(device)
        x = torch.as_tensor(dm, dtype=torch.float32, device=device)
        y = torch.as_tensor(binned, dtype=torch.float32, device=device)

        _, weight, bias = glm.fit(x, y, epochs=epochs, optim=optim, lr=lr)
        # Store parameters
        biasdm = np.pad(dm.copy(), ((0, 0), (1, 0)), 'constant', constant_values=1)
        for cell in cells:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            cellbinned = binned[:, cell_idx]
            coefs.at[cell] = weight[cell_idx, :]
            intercepts.at[cell] = bias[cell_idx]

            wts = np.concatenate([[bias[cell_idx]], weight[cell_idx, :]], axis=0)
            if retvar:
                wvar = np.diag(np.linalg.inv(dd_neglog(wts, biasdm, cellbinned)))
            else:
                wvar = np.ones((wts.shape[0], wts.shape[0])) * np.nan
            variances.at[cell] = wvar[1:]

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

    def fit(self, method='sklearn', alpha=0, singlepar_var=False, epochs=6000, optim='adam',
            lr=0.3):
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
        if optim != 'adam':
            epochs = 500
        # TODO: Make this optionally parallel across multiple cores of CPU
        # Initialize pd Series to store output coefficients and intercepts for fits
        trainmask = np.isin(self.trlabels, self.traininds).flatten()  # Mask for training data
        trainbinned = self.binnedspikes[trainmask]
        print(f'Condition of design matrix is {np.linalg.cond(self.dm[trainmask])}')

        if not self.subset:
            if method == 'sklearn':
                traindm = self.dm[trainmask]
                coefs, intercepts, variances = self._fit_sklearn(traindm, trainbinned, alpha,
                                                                 retvar=True)
            elif method == 'pytorch':
                traindm = self.dm[trainmask]
                coefs, intercepts, variances = self._fit_pytorch(traindm, trainbinned,
                                                                 retvar=True, epochs=epochs,
                                                                 optim=optim, lr=lr)
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
                                                                     retvar=singlepar_var)
                elif method == 'pytorch':
                    coefs, intercepts, variances = self._fit_pytorch(traindm, trainbinned,
                                                                     retvar=singlepar_var,
                                                                     epochs=epochs,
                                                                     optim=optim, lr=lr)
                else:
                    biasdm = np.pad(traindm.copy(), ((0, 0), (1, 0)), 'constant',
                                    constant_values=1)
                    coefs, intercepts, variances = self._fit_minimize(biasdm, trainbinned,
                                                                      retvar=singlepar_var)
                scores = self._score_submodel(coefs, intercepts, testdm, testbinned)
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
                    elif method == 'pytorch':
                        coefs, intercepts, variances = self._fit_pytorch(traindm, trainbinned,
                                                                         cells=currcells,
                                                                         retvar=retvar,
                                                                         epochs=epochs,
                                                                         optim=optim, lr=lr)
                    else:
                        biasdm = np.pad(traindm.copy(), ((0, 0), (1, 0)), 'constant',
                                        constant_values=1)
                        coefs, intercepts, variances = self._fit_minimize(biasdm, trainbinned,
                                                                          cells=currcells,
                                                                          retvar=retvar)
                    iscores.append(self._score_submodel(coefs, intercepts, testdm, testbinned))
                submodel_scores[f'{i}cov'] = pd.concat(iscores).sort_index()
            self.submodel_scores = submodel_scores
            self.coefs = coefs
            self.intercepts = intercepts
            self.variances = variances
            return

    def _score_submodel(self, weights, intercepts, dm, binned):
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
        for cell in weights.index:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            wt = weights.loc[cell].reshape(-1, 1)
            bias = intercepts.loc[cell]
            y = binned[:, cell_idx]
            pred = np.exp(dm @ wt + bias)
            null_pred = np.ones_like(pred) * np.mean(y)
            null_deviance = 2 * np.sum(xlogy(y, y / null_pred.flat) - y + null_pred.flat)
            with np.errstate(divide='ignore', invalid='ignore'):
                full_deviance = 2 * np.sum(xlogy(y, y / pred.flat) - y + pred.flat)
            d_sq = 1 - (full_deviance / null_deviance)
            scores.at[cell] = d_sq
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
            variances = self.variances.apply(lambda v: np.sum(v[winds] * bases, axis=1))
            offset = self.covar[var]['offset']
            tlen = bases.shape[0] * self.binwidth
            tstamps = np.linspace(0 + offset, tlen + offset, bases.shape[0])
            outputs[var] = pd.DataFrame(weights.values.tolist(),
                                        index=weights.index,
                                        columns=tstamps)
            varoutputs[var] = pd.DataFrame(variances.values.tolist(),
                                           index=weights.index,
                                           columns=tstamps)
        self.combined_weights = outputs
        self.combined_variances = varoutputs
        return outputs

    def score(self):
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
        scores = pd.Series(index=self.coefs.index)
        for cell in self.coefs.index:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            wt = self.coefs.loc[cell].reshape(-1, 1)
            bias = self.intercepts.loc[cell]
            y = self.binnedspikes[testmask, cell_idx]
            pred = np.exp(testdm @ wt + bias)
            null_pred = np.ones_like(pred) * np.mean(y)
            null_deviance = 2 * np.sum(xlogy(y, y / null_pred.flat) - y + null_pred.flat)
            with np.errstate(invalid='ignore', divide='ignore'):
                full_deviance = 2 * np.sum(xlogy(y, y / pred.flat) - y + pred.flat)
            d_sq = 1 - (full_deviance / null_deviance)
            scores.at[cell] = d_sq
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

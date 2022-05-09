
"""
GLM fitting utilities based on NeuroGLM by Il Memming Park, Jonathan Pillow:

https://github.com/pillowlab/neuroGLM

Berk Gercek
International Brain Lab, 2020
"""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.special import xlogy
from brainbox.processing import bincount2D
from .utils import neglog


class NeuralModel:
    """
    Parent class for multiple types of neural models. Contains core methods for extracting
    trial-relevant spike times and binning spikes, as well as making sure that the design matrix
    being used makes sense.
    """

    def __init__(self, design_matrix, spk_times, spk_clu,
                 binwidth=0.02, mintrials=100, stepwise=False):
        """
        Construct GLM object using information about all trials, and the relevant spike times.
        Only ingests data, and further object methods must be called to describe kernels, gain
        terms, etc. as components of the model.

        Parameters
        ----------
        design_matrix: brainbox.modeling.design_matrix.DesignMatrix
            Design matrix object which has already been compiled for use with neural data.
        spk_times: numpy.array of floats
            1-D array of times at which spiking events were detected, in seconds.
        spk_clu: numpy.array of integers
            1-D array of same shape as spk_times, with integer cluster IDs identifying which
            cluster a spike time belonged to.
        binwidth : float
            Size of bins to put spikes in to, in seconds.
        mintrials: int
            Minimum number of trials in which neurons fired a spike in order to be fit. Defaults
            to 100 trials.
        stepwise: bool
            Whether or not to perform stepwise regression, in which the model is built iteratively
            from only the mean rate, up. This allows comparison of D^2 scores for sub-models which
            incorporate only some parameters, to see which regressors actually improve
            explainability. Defaults to False.
        """
        # Data checks #
        if not len(spk_times) == len(spk_clu):
            raise IndexError("Spike times and cluster IDs are not same length")
        if not design_matrix.compiled:
            raise AttributeError('Design matrix object must be compiled before passing to fit')

        # Filter out cells which don't meet the criteria for minimum spiking, while doing trial
        # assignment
        base_df = design_matrix.base_df
        clu_ids = np.unique(spk_clu).flatten()
        trbounds = base_df[['trial_start', 'trial_end']]  # Get the start/end of trials
        # Initialize a Cells x Trials bool array to easily see how many trials a clu spiked
        trialspiking = np.zeros((base_df.index.max() + 1, clu_ids.max() + 1), dtype=bool)
        # Empty trial duration value to use later
        # Iterate through each trial, and store the relevant spikes for that trial into a dict
        # Along with the cluster labels. This makes binning spikes and accessing spikes easier.
        spks = {}
        clu = {}
        st_endlast = 0
        for i, (start, end) in trbounds.iterrows():
            st_startind = np.searchsorted(spk_times[st_endlast:], start) + st_endlast
            st_endind = np.searchsorted(spk_times[st_endlast:], end, side='right') + st_endlast
            st_endlast = st_endind
            trial_clu = np.unique(spk_clu[st_startind:st_endind])
            trialspiking[i, trial_clu] = True
            spks[i] = spk_times[st_startind:st_endind] - start
            clu[i] = spk_clu[st_startind:st_endind]

        # Set model parameters to begin with
        self.design = design_matrix
        self.spikes = spks
        self.clu = clu
        self.clu_ids = np.argwhere(np.sum(trialspiking, axis=0) > mintrials).flatten()
        self.stepwise = stepwise
        self.binwidth = binwidth

        if len(self.clu_ids) == 0:
            raise UserWarning('No neuron fired a spike in a minimum number.')

        # Bin spikes
        spkarrs, arrdiffs = [], []
        for i in self.design.trialsdf.index:
            duration = self.design.trialsdf.loc[i, 'duration']
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
        if hasattr(self.design, 'dm'):
            assert y.shape[0] == self.design.dm.shape[0], "Oh shit. Indexing error."
        self.binnedspikes = y

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
        for var in self.design.covar.keys():
            if self.design.covar[var]['bases'] is None:
                wind = self.design.covar[var]['dmcol_idx']
                outputs[var] = self.coefs.apply(lambda w: w[wind])
                continue
            winds = self.design.covar[var]['dmcol_idx']
            bases = self.design.covar[var]['bases']
            weights = self.coefs.apply(lambda w: np.sum(w[winds] * bases, axis=1))
            offset = self.design.covar[var]['offset']
            tlen = bases.shape[0] * self.binwidth
            tstamps = np.linspace(0 + offset, tlen + offset, bases.shape[0])
            outputs[var] = pd.DataFrame(weights.values.tolist(),
                                        index=weights.index,
                                        columns=tstamps)
        self.combined_weights = outputs
        return outputs

    def _scorer(self, wt, bias, dm, y):
        """
        Score a single target y
        """
        pred = self.link(dm @ wt + bias).flatten()
        if self.metric == 'dsq':
            null_pred = np.ones_like(pred) * np.mean(y)
            null_deviance = 2 * np.sum(xlogy(y, y / null_pred.flat) - y + null_pred.flat)
            with np.errstate(invalid='ignore', divide='ignore'):
                full_deviance = 2 * np.sum(xlogy(y, y / pred.flat) - y + pred.flat)
            return 1 - (full_deviance / null_deviance)
        elif self.metric == 'msespike':
            residuals = (y - pred) ** 2
            return residuals.sum() / y.sum()
        elif self.metric == 'rsq':
            return r2_score(y, pred)
        elif self.metric == 'nllspike':
            biasdm = np.pad(dm, ((0, 0), (1, 0)), constant_values=1)
            return -neglog(np.vstack((bias, wt)).flatten(), biasdm, y) / np.sum(y)
        else:
            raise AttributeError('No valid metric exists in the instance for use by _scorer()')

    def fit(self, train_idx=None, printcond=True):
        """
        Fit the current set of binned spikes as a function of the current design matrix. Requires
        NeuralGLM.bin_spike_trains and NeuralGLM.compile_design_matrix to be run first. Will store
        the fit weights to an internal variable. To access these fit weights in a pandas DataFrame
        use the NeuralGLM.combine_weights method.

        Parameters
        ----------
        train_idx : array-like of trial indices, optional
            List of which trials to use to train the model. Defaults to None, which indicates all
            indices in the trialsdf will be used (100% train)
        printcond : bool
            Whether or not to print the condition number of the design matrix. Defaults to True

        Returns
        -------
        coefs : list
            List of coefficients fit. Not recommended to use these for interpretation. Use
            the .combine_weights() method instead.
        intercepts : list
            List of intercepts (bias terms) fit.
        """
        # Input checks
        if train_idx is None:
            train_idx = self.design.trialsdf.index
        if not np.all(np.isin(train_idx, self.design.trialsdf.index)):
            raise IndexError('Not all train indices in the trials of design matrix')

        # Store training and test indices for self so that .score() method will know what to
        # operate on. If all data indices are in train indices, train and test are the same set.
        self.traininds = train_idx
        if not np.all(np.isin(self.design.trialsdf.index, train_idx)):
            self.testinds = self.design.trialsdf.index[~self.trialsdf.index.isin(train_idx)]
        else:
            self.testinds = train_idx

        # Mask for training data
        trainmask = np.isin(self.design.trlabels, train_idx).flatten()
        trainbinned = self.binnedspikes[trainmask]
        if printcond:
            print(f'Condition of design matrix is {np.linalg.cond(self.design[trainmask])}')

        traindm = self.design[trainmask]
        coefs, intercepts, = self._fit(traindm, trainbinned)
        self.coefs, self.intercepts = coefs, intercepts
        return

    def score(self, testinds=None):
        """
        Score model using chosen metric

        Returns
        -------
        pandas.Series
            Score using chosen metric (defined at instantiation) for each unit fit by the model.
        """
        if not hasattr(self, 'coefs'):
            raise AttributeError('Model has not been fit yet.')
        if testinds is None:
            testinds = self.testinds
        testmask = np.isin(self.design.trlabels, testinds).flatten()
        dm, binned = self.design[testmask, :], self.binnedspikes[testmask]

        scores = pd.Series(index=self.coefs.index, name='scores')
        for cell in self.coefs.index:
            cell_idx = np.argwhere(self.clu_ids == cell)[0, 0]
            wt = self.coefs.loc[cell].reshape(-1, 1)
            bias = self.intercepts.loc[cell]
            y = binned[:, cell_idx]
            scores.at[cell] = self._scorer(wt, bias, dm, y)
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

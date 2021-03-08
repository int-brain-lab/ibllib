
"""
GLM fitting utilities based on NeuroGLM by Il Memming Park, Jonathan Pillow:

https://github.com/pillowlab/neuroGLM

Berk Gercek
International Brain Lab, 2020
"""
import numpy as np
from brainbox.processing import bincount2D


class NeuralModel:
    """
    Parent class for multiple types of neural models. Contains core methods for extracting
    trial-relevant spike times and binning spikes, as well as making sure that the design matrix
    being used makes sense.
    """

    def __init__(self, design_matrix, spk_times, spk_clu,
                 train=0.8, blocktrain=False, mintrials=100, subset=False):
        """
        Construct GLM object using information about all trials, and the relevant spike times.
        Only ingests data, and further object methods must be called to describe kernels, gain
        terms, etc. as components of the model.

        Parameters
        ----------
        design_matrix: NeuralDesignMatrix
            Design matrix object which has already been compiled for use with neural data.
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
        if not len(spk_times) == len(spk_clu):
            raise IndexError("Spike times and cluster IDs are not same length")
        if not isinstance(train, float) and not train == 1:
            raise TypeError('train must be a float between 0 and 1')
        if not ((train > 0) & (train <= 1)):
            raise ValueError('train must be between 0 and 1')
        if not design_matrix.compiled:
            raise AttributeError('Design matrix object must be compiled before passing to fit')

        # Filter out cells which don't meet the criteria for minimum spiking, while doing trial
        # assignment
        trialsdf = design_matrix.trialsdf
        clu_ids = np.unique(spk_clu)
        trbounds = trialsdf[['trial_start', 'trial_end']]  # Get the start/end of trials
        # Initialize a Cells x Trials bool array to easily see how many trials a clu spiked
        trialspiking = np.zeros((trialsdf.index.max() + 1, clu_ids.max() + 1), dtype=bool)
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

        # Break the data into test and train sections for cross-validation
        if train == 1:
            print('Training fraction set to 1. Training on all data.')
            traininds = trialsdf.index
            testinds = trialsdf.index
        elif blocktrain:
            trainlen = int(np.floor(len(trialsdf) * train))
            testlen = len(trialsdf) - trainlen
            midpoint = len(trialsdf) // 2
            starttest = midpoint - (testlen // 2)
            endtest = midpoint + (testlen // 2)
            testinds = trialsdf.index[starttest:endtest]
            traininds = trialsdf.index[~np.isin(trialsdf.index, testinds)]
        else:
            trainlen = int(np.floor(len(trialsdf) * train))
            traininds = sorted(np.random.choice(trialsdf.index, trainlen, replace=False))
            testinds = trialsdf.index[~trialsdf.index.isin(traininds)]

        # Set model parameters to begin with
        self.design_matrix = design_matrix
        self.dm = design_matrix.dm
        self.covar = design_matrix.covar
        self.spikes = spks
        self.clu = clu
        self.clu_ids = np.argwhere(np.sum(trialspiking, axis=0) > mintrials)
        self.binwidth = design_matrix.binwidth
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

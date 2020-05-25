"""
GLM fitting utilities loosely based on NeuroGLM by Il Memming Park, Jonathan Pillow:

https://github.com/pillowlab/neuroGLM

International Brain Lab, 2020
"""
import numpy as np
import pandas as pd
from brainbox.core import Bunch
from .dataset import DataSet


class NeuralGLM:
    """
    Generalized Linear Model which seeks to describe spiking activity as the output of a poisson
    process. Uses sklearn's GLM methods under the hood while providing useful routines for dealing
    with neural data
    """
    def __init__(self, trialsdf, spk_times, spk_clu, vartypes, binwidth=0.02):
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
            `bb.processing.dfconstructor` can help with making this dataframe.
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
        binwidth: float
            Width, in seconds, of the bins which will be used to count spikes. Defaults to 20ms.

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

        self.clu_ids = np.unique(spk_clu)

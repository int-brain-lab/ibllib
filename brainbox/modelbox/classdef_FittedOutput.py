""" by Luigi Acerbi, Shan Shen and Anne Urai
International Brain Laboratory, 2019
"""

import pandas as pd
import numpy as np


class FittedOutput(self, data, model, method):
    """ Abstract class for the results of a model fit.
    Attributes:
        model # dictionary of model identifier (e.g. datajoint primary key of Model)
        data # dictionary of data identifier (e.g. datajoint primary key of DataSet)
        model_metrics # dictionary
    Methods
        simulate (calls Model.simulate with some parameter set)
        diagnose
        plot
        parameter_recovery
    """
    
    # define attributes
    self.data = data
    self.model = model
    self.method = method

    # determine which method is used to fit
    if self.method.name == 'maximum_likelihood_estimation':
        self = MaximumLikelihoodOutput


class MaximumLikelihoodOutput(FittedOutput):
    """
    Attributes
        starting_points # num_startingpoints x num_params (df?)
        loglikelihoods # num_startingpoints
        maximum_points # num_startingpoints x num_params
    """

class PosteriorOutput(FittedOutput):
    """
    Attributes
        posterior
    Methods
        draw_sample
    """

class MCMCPosteriorOutput(PosteriorOutput):
    """
    this class will most likely be a PyMC object
    """

""" by Luigi Acerbi, Shan Shen and Anne Urai
International Brain Laboratory, 2019
"""

import pandas as pd
import numpy as np

class DataSet:
    """ Object containing data that can be preprocessed.
    Inputs: dataframe # pandas
    Methods: preprocess
    """


class BehavioralData:

    def __init__(self, data, meta_data=dict()):
        """data should contain the following columns:
        stimulus_side       : values to be -1 or 1
        stimulus_strength   : non-negative contrast
        choice              : -1 and 1, nan for missed trials
        rewarded            : 0 and 1, including 0 contrast trials
        ---
        optional columns:
        correct             : 1 for correct, 0 for incorrect, nan for 0 contrast trials or missed trials
        reaction_time       : time diff of response time - stim on time
        prob_left_block     : probability (in block structure) of stimulus_side == -1
        trial_id            :
        session_id          :
        """

        self.trials_df = self.fill_dataframe(data)
        self.meta_data = meta_data

        # TODO: make it a dataframe with columns signed_contrast, total_trial_number, fraction_right
        self.summarized_data = self.trials_df.groupby('signed_contrast').agg(
            {'choice': 'count', 'choice_right': 'mean'}).reset_index()

    # TODO: to implement the function that fill in the columns that needed for later processing
    def fill_dataframe(data):
        data['choice_right'] = data['choice'].replace([-1, 0, 1], [0, np.nan, 1])
        return data

    def plot(self):
        pass


class Model:
    """Abstract class for defining models.
    Attributes:
        parameter_list # list of objects from Parameter class
        name # string
        description # string
        loglikelihood_function # function handle
    Methods:
        preprocess
        fit
        simulate
    """


class PsychometricFunction(Model):
    """Class for representing psychometric functions.
        Grab different functions from psignifit, etc
    """

    def __init__(self, model_name='erf_2lapses'):

        self.model_name = model_name
        if model_name == 'erf_2gammas':
            self.parameter_list = [
                Parameter(name='bias',
                          description='bias',
                          bounds_hard=[-100, 100],
                          range_plausible=[-50, 50]),
                Parameter(name='threshold',
                          description=r'Threshold $(\sigma)$',
                          bounds_hard=[0, 200],
                          range_plausible=[1, 50],
                          typical_value=15),
                Parameter(name='lapse_left',
                          description=r'Lapse left $(\lambda_L)$,
                          bounds_hard=[0, 0.5],
                          range_plausible=[0, 0.3],
                          typical_value=0.1),
                Parameter(name='lapse_right',
                          description=r'Lapse right $(\lambda_R)$',
                          bounds_hard=[0, 0.5],
                          range_plausible=[0, 0.3],
                          typical_value=0.1)]
            self.description = 'Psychometric function (erf, 2 lapses) for "right" (+1) responses.'
            self.psych_function = \
                lambda x, params: \
                params[2] + (1 - params[2] - params[3]) * (erf((x - params[0]) / params[1]) + 1) / 2

        def loglikelihood_function(self, data_set):

            # TODO: Anne
            prob_choice_right = self.psych_function(self.)
            l = - sum(nn*(choice_right_frac*np.log(sprobs)+(1-choice_right_frac)*np.log(1-probs)))


class Parameter:
    """
    Attributes:
        name # string
        description # string
        bounds # float (lower, upper, hard, plausible)
        typical_value # float
        parameterization # string
    """

    def __init__(name, description='',
                 bounds_hard, range_plausible=[],
                 typical_value=[], parameterization=[]):

        self.name = name
        self.description = description
        self.bounds_hard = bounds_hard

        if range_plausible:
            self.range_plausible = range_plausible
        else:
            self.range_plausible = bounds_hard

        if typical_value:
            self.typical_value = typical_value
        else:
            self.typical_value = (self.range_plausible[1] - self.range_plausible[0])/2.

        if paramterization:
            self.parameterization = parameterization
        else:
            self.parameterization = 'standard'


class FittingMethod:
    """ Abstract class with wrappers to e.g. skikit-learn functions
    Attributes:
        name # string
        algorithm # function handle
    Methods:
        fit
    """


class MaximumLikelihoodEstimation(FittingMethod):
    """ Maximum Likelihood Estimation
    """

class PosteriorEstimation(FittingMethod):
    """ Maximum Likelihood Estimation
    """


class FittedOutput:
    """ Abstract class for the results of a model fit.
    Name to be agreed on. FittedOutput/FittedResult/?
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


class MaximumLikelihoodOutput(FittedModel):
    """
    Attributes
        starting_points # num_startingpoints x num_params (df?)
        loglikelihoods # num_startingpoints
        maximum_points # num_startingpoints x num_params
    """

class PosteriorOutput(FittedModel):
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

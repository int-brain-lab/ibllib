""" by Luigi Acerbi, Shan Shen and Anne Urai
International Brain Laboratory, 2019
"""

import numpy as np
from parameter import Parameter
from scipy.special import erf
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.stats.proportion as sm


class Model:
    """Abstract class for defining models.
    Attributes:
        name # string
        description # string
        variable_names # variables that are expected in data.preproc_df
        parameter_list # list of objects from Parameter class
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

    def __init__(self, model_name='choice_erf_2lapses', data=[]):

        self.model_name = model_name
        # ================================================== #
        # STANDARD PSYCHOMETRIC FUNCTION FOR IBL ROOT TASK
        # ================================================== #

        if model_name == 'choice_erf_2lapses':
            self.description = 'Psychometric function (erf, 2 lapses) for "right" (+1) responses'
            self.variable_names = ['signed_stimulus', 'total_trial_number', 'fraction_right']

            # TODO: take parameter bounds and starting points from the data range
            self.parameter_list = [
                Parameter(name='bias',
                          description=r'Bias $(\mu)$',
                          bounds_hard=[-100, 100],
                          range_plausible=[-50, 50]),
                Parameter(name='threshold',
                          description=r'Threshold $(\sigma)$',
                          bounds_hard=[0, 200],
                          range_plausible=[1, 50],
                          typical_value=15),
                Parameter(name='lapse_left',
                          description=r'Lapse left $(\lambda_L)$',
                          bounds_hard=[0, 0.5],
                          range_plausible=[0, 0.3],
                          typical_value=0.1),
                Parameter(name='lapse_right',
                          description=r'Lapse right $(\lambda_R)$',
                          bounds_hard=[0, 0.5],
                          range_plausible=[0, 0.3],
                          typical_value=0.1)]
            self.function = \
                lambda x, params: params[2] + (1 - params[2] - params[3]) \
                * (erf((x - params[0]) / params[1]) + 1) / 2

        elif model_name == 'correct_weibull':
            pass

    # ================================================== #
    # LOGLIKELIHOOD FUNCTION
    # ================================================== #

    def loglikelihood_function(self, params, _model, _data):

        # evaluation of the model with these parameters
        probs = _model.function(_data.preproc_df.signed_stimulus, params)
        assert (max(probs) <= 1) or (min(probs) >= 0), 'Probs must be between 0 and 1'

        # avoid numerical problems
        probs[probs == 0] = np.finfo(float).eps
        probs[probs == 1] = 1 - np.finfo(float).eps

        # ll of data given parameters - use _data.preproc_df
        ll = sum(_data.preproc_df.total_trial_number *
                 (_data.preproc_df.fraction_right * np.log(probs) +
                  (1 - _data.preproc_df.fraction_right) * np.log(1 - probs)))
        return ll

    # ================================================== #
    # MODEL-DEPENDENT PREPROCESSING
    # ================================================== #

    def preprocess_data(self, data):

        if 'choice' in self.model_name:
            # make dataframe with columns signed_stimulus, total_trial_number, fraction_right
            print("Summarizing TrialData")
            data.preproc_df = data.trials_df.groupby('signed_stimulus').agg(
                {'choice': 'count', 'choice_right': 'mean'}).reset_index()
            data.preproc_df.rename(columns={'choice': 'total_trial_number',
                                   'choice_right': 'fraction_right'}, inplace=True)
        elif 'correct' in self.name:
            print('not implemented yet')
            pass

        # CHECK IF ALL VARIABLE NAMES WE NEED ARE PRESENT
        for v in self.variable_names:
            assert v in data.preproc_df.columns, ('preproc_df needs column %s' % v)

        return data

    # ================================================== #
    # PLOT (DEPENDS ON MODEL)
    # ================================================== #

    def plot(self, fittedoutput, plot_data=True, plot_fit=True, **kwargs):

        if 'choice' in fittedoutput.model.model_name:

            fig, ax = plt.subplots()
            # only plot preprocessed data
            if plot_data:
                assert hasattr(fittedoutput.data, 'preproc_df'), 'Call .preprocess() then .plot()'

                df = fittedoutput.data.preproc_df

                # TODO: BINOMIAL CONFIDENCE INTERVALS
                eb = sm.proportion_confint(df['fraction_right'] * df['total_trial_number'],
                                           df['total_trial_number'],
                                           alpha=0.05,
                                           method='wilson')

                ax.errorbar(df['signed_stimulus'], df['fraction_right'],
                            yerr=[- eb[0] + df['fraction_right'], eb[1] - df['fraction_right']],
                            linestyle='None', mew=0.5, marker='o')

            if plot_fit:
                x_vec = np.arange(min(fittedoutput.data.preproc_df.signed_stimulus),
                                  max(fittedoutput.data.preproc_df.signed_stimulus))
                sns.lineplot(x_vec, fittedoutput.model.function(x_vec,
                             fittedoutput.result['parameters']), ax=ax, **kwargs)

            # layout
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.set_xlabel('Signed stimulus')
            ax.set_ylabel('P(right)')

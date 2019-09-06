""" by Luigi Acerbi, Shan Shen and Anne Urai
International Brain Laboratory, 2019

#TODO: Anne
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython import embed as shell
sns.set_palette("gist_gray")  # palette for water types


class DataSet:
    """ Object containing data that can be preprocessed.
    Inputs: dataframe # pandas
    Methods: preprocess
    """


class TrialData:

    def __init__(self, data, meta_data=dict()):
        """data should contain the following columns:
        stimulus_side       : values to be -1 or 1
        stimulus_strength   : non-negative contrast
        choice              : -1 and 1, nan for missed trials
        rewarded            : 0 and 1, including 0 contrast trials
        ---
        optional columns:
        correct             : 1 for correct, 0 for incorrect, nan for 0 contrast or missed trials
        reaction_time       : time diff of response time - stim on time
        prob_left_block     : probability (in block structure) of stimulus_side == -1
        trial_id            :
        session_id          :
        """

        self.meta_data = meta_data
        self.trials_df = self.fill_dataframe(data)

        # make dataframe with columns signed_stimulus, total_trial_number, fraction_right
        # TODO: should this be part of the DataSet or the Model classdef?
        self.preproc_df = self.trials_df.groupby('signed_stimulus').agg(
            {'choice': 'count', 'choice_right': 'mean'}).reset_index()
        self.preproc_df.rename(columns={'choice': 'total_trial_number',
                               'choice_right': 'fraction_right'}, inplace=True)

    def fill_dataframe(self, data):
        data['signed_stimulus'] = data['stimulus_strength'] * data['stimulus_side']
        data['choice_right'] = data['choice'].replace([-1, 0, 1], [0, np.nan, 1])
        return data


""" by Luigi Acerbi, Shan Shen and Anne Urai
International Brain Laboratory, 2019
"""

# necessary modules
from ibl_pipeline import subject, acquisition, behavior
import numpy as np
# import pandas as pd
from IPython import embed as shell
import matplotlib.pyplot as plt


# import modelbox stuff manually for now
from dataset import TrialData
from model import PsychometricFunction
from fittingmethod import MaximumLikelihoodEstimation


# IMPORT SOME DATA FROM DATAJOINT
b = behavior.TrialSet.Trial * (acquisition.Session & 'task_protocol LIKE "%biased%"') \
    * (subject.Subject & 'subject_nickname="CSHL_015"') * subject.SubjectLab()

bdat = b.fetch(order_by='subject_nickname, session_start_time, trial_id',
               format='frame').reset_index()

# TODO: put this wrangle in a DJ table
bdat['signed_contrast'] = (bdat['trial_stim_contrast_right'] -
                           bdat['trial_stim_contrast_left']) * 100
bdat['signed_contrast'] = bdat.signed_contrast.astype(int)
bdat['stimulus_strength'] = np.abs(bdat['signed_contrast'])
bdat['choice'] = bdat['trial_response_choice'].map({'CCW': 1, 'No Go': np.nan, 'CW': -1})
bdat['rewarded'] = bdat['trial_feedback_type'].map({-1: 0, 1: 1})
bdat['stimulus_side'] = bdat['choice'].copy()
bdat.loc[bdat.rewarded == 0, 'stimulus_side'] = -1 * bdat.loc[bdat.rewarded == 0, 'stimulus_side']
bdat = bdat[['stimulus_side', 'stimulus_strength', 'choice', 'rewarded']]  # what do we need?

# create an instance of the DataSet class
df = TrialData(data=bdat)

# now create an instance of the PsychometricFunction class,
# let some parameters depend on the data
psychfunc = PsychometricFunction(model_name='erf_2lapses', data=df)
# preprocess the data for psychometric function fitting
df = psychfunc.preprocess(df)

# # define the method that we want to use for fitting
mle_fit = MaximumLikelihoodEstimation(data=df, model=psychfunc)
mle_fit.fit()

# plot the fitted psychometric function and then the data
mle_fit.plot()
plt.title('ModelBox''s first function fit')
plt.show()

# TODO: allow for averaging a list of mle_fit objects and show group-level stuff

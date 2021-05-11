import numpy as np
import brainbox.io.one as bbone
import brainbox.modeling.utils as mut
from brainbox.modeling.design_matrix import DesignMatrix
from brainbox.modeling.linear import LinearGLM
from brainbox.modeling.poisson import PoissonGLM
from oneibl import one

BINSIZE = 0.02
KERNLEN = 0.6
SHORT_KL = 0.4
NBASES = 10

one = one.ONE()

eid = '4d8c7767-981c-4347-8e5e-5d5fffe38534'

# Load in a dataframe of trial-by-trial data
trialsdf = bbone.load_trials_df(eid, one, maxlen=2., t_before=0.4, t_after=0.6, ret_wheel=True,
                                wheel_binsize=BINSIZE)

# Define what kind of data type each column is
vartypes = {
    'choice': 'value',
    'probabilityLeft': 'value',
    'feedbackType': 'value',
    'feedback_times': 'timing',
    'contrastLeft': 'value',
    'contrastRight': 'value',
    'goCue_times': 'timing',
    'stimOn_times': 'timing',
    'trial_start': 'timing', 'trial_end': 'timing',
    'wheel_velocity': 'continuous'
}

# The following is not a sensible model per se of the IBL task, but illustrates each regressor type

# Initialize design matrix
design = DesignMatrix(trialsdf, vartypes=vartypes, binwidth=BINSIZE)
# Build some basis functions
longbases = mut.full_rcos(KERNLEN, NBASES, design.binf)
shortbases = mut.full_rcos(SHORT_KL, NBASES, design.binf)

# The following are all timing regressors, i.e. modeling responses after a timing event

# Add regressors for Stim ON L/R
design.add_covariate_timing('stimL', 'stimOn_times', longbases,
                            cond=lambda tr: np.isfinite(tr.contrastLeft),
                            desc='Stimulus onset left side kernel')
design.add_covariate_timing('stimR', 'stimOn_times', longbases,
                            cond=lambda tr: np.isfinite(tr.contrastRight),
                            desc='Stimulus onset right side kernel')
# Regressor for feedback
design.add_covariate_timing('feedback', 'feedback_times', longbases,
                            desc='feedback kernel (any kind of feedback)')
# Regressing against continuous variables is similar. Note that because .add_covariate() is the
# Core function for adding regressors, it will need an explicit pd.Series to operate, and not a
# column name, for the second argument
design.add_covariate('wheel', trialsdf['wheel_velocity'], shortbases, offset=-SHORT_KL,
                     desc='Anti-causal regressor for wheel velocity')
# We can also regress while omitting basis functions:
design.add_covariate_raw('wheelraw', 'wheel_velocity', desc='Wheel velocity, no bases')
design.compile_design_matrix()

# Now let's load in some spikes and fit them
spikes, clusters = bbone.load_spike_sorting(eid, one, probe='probe00')
spk_times = spikes.probe00.times
spk_clu = spikes.probe00.clusters

# We will build a linear model and a poisson model:
lm = LinearGLM(design, spk_times, spk_clu, binwidth=BINSIZE)
pm = PoissonGLM(design, spk_times, spk_clu, binwidth=BINSIZE)

# Running the .fit() method is enough to start the fitting procedure:
lm.fit()
pm.fit()

# After which we can assess the score of each model on our data:
lm.score()
pm.score()

# Optionally, we could also run stepwise regression, also known as Sequential Feature Selection
sfs = mut.SequentialSelector(lm, n_features_to_select=3)
sfs.fit(progress=True)
sfs.scores_

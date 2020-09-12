from oneibl.one import ONE
import datetime
import numpy as np
from brainbox.core import Bunch
import ibl_pipeline.utils.psychofit as psy
one = ONE()

# need to get last three sessions of a subject
# download all the trials

# get sessions in last 10 days if < 3 then just get them all

subj = 'SWC_029'

if date is None:
    latest_sess = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    sess_minus_week = (datetime.date.today() - datetime.timedelta(days=8)).strftime("%Y-%m-%d")
    #compute from today
else:
    # compute from the date specified
    # Need some error handling for format of data
    assert()
    latest_sess = datetime.datetime.strptime(date, '%Y-%m-%d')
    sess_minus_week = (latest_sess - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
    latest_sess = date


sessions = one.alyx.rest('sessions', 'list', subject=subj, date_range=[sess_minus_week,
                         latest_sess], dataset_types='trials.intervals')

# If not enough sessions in the last week, then just fetch them all
if len(sessions) < 3:
    sessions = one.alyx.rest('sessions', 'list', subject=subj)

trials = Bunch()
n = 0
task_protocol = []
sess_dates = []
while len(trials) < 3:
    trials_ = one.load_object(sessions[n]['url'].split('/')[-1], 'trials')
    if trials_:
        task_protocol.append(re.search('tasks_(.*)Choice', sessions[n]['task_protocol']).group(1))
        sess_dates.append(sessions[n]['start_time'][:10])
        trials[sessions[n]['start_time'][:10]] = trials_

    n += 1

trials_all = Bunch()
for k in trials_.keys():
    trials_all[k] = np.concatenate(list(trials[kk][k] for kk in trials.keys()))


if np.all(np.array(task_protocol) == 'training'):

    signed_contrast = get_signed_contrast(trials_all)
    perf_easy = np.array([compute_performance_easy(trials[k]) for k in trials.keys()])
    n_trials = np.array([compute_n_trials(trials[k]) for k in trials.keys()])
    psych = compute_psychometric(trials_all, signed_contrast=signed_contrast)
    rt = compute_median_reaction_time(trials_all, signed_contrast=signed_contrast)

    if not np.any(signed_contrasts == 0):
        status = 'in training'
    else:
        if criterion_1b(psych, n_trials, perf_easy, rt):
            status = 'trained 1b'
        elif criterion_1a(psych, n_trials, perf_easy):
            status = 'trained 1a'
        else:
            status = 'in training'

# Case when there are < 3 biasedChoiceWorld sessions after reaching trained_1b criterion
if not ~np.all(np.array(task_protocol) == 'training') and \
        np.any(np.array(task_protocol) == 'training'):
    status = 'trained_1b'

# Case when there is biasedChoiceWorld or ephysChoiceWorld in last three sessions
if not np.any(np.array(task_protocol) == 'training'):
    # Check to see if any have been done on the ephys rig
    ephys_rig_sess = one.alyx.rest('sessions', 'list', subject=subj, date_range=[sess_dates[-1],
                                   sess_dates[0]], django='json__PYBPOD_BOARD__icontains,ephys')

    # We are still on training rig and so all sessions should be biased
    if len(ephys_rig_sess) == 0:
        assert(np.all(np.array(task_protocol) == 'biased'))
        signed_contrast = get_signed_contrast(trials_all)
        perf_easy = np.array([compute_performance_easy(trials[k]) for k in trials.keys()])
        n_trials = np.array([compute_n_trials(trials[k]) for k in trials.keys()])
        psych_20 = compute_psychometric(trials_all, signed_contrast=signed_contrast, block=0.2)
        psych_80 = compute_psychometric(trials_all, signed_contrast=signed_contrast, block=0.8)
        rt = compute_median_reaction_time(trials_all, signed_contrast=signed_contrast)

        if criterion_ephys(psych_20, psych_80, n_trials, perf_easy, rt):
            status = 'ready4ephysrig'
        else:
            status = 'trained1b'

    if len(ephys_rig_sess) < 3:

        ephys_sess_dates = [sess['start_time'][:10] for sess in ephys_rig_sess]
        assert(np.all(np.array([date in trials for date in ephys_sess_dates])))
        perf_easy = np.array([compute_performance_easy(trials[k]) for k in ephys_sess_dates])
        n_trials = np.array([compute_n_trials(trials[k]) for k in ephys_sess_dates])

        if criterion_delay(perf_easy, n_trials):
            status = 'ready4delay'
        else:
            status = 'ready4ephysrig'

    if len(ephys_rig_sess) >= 3:
        # See if any sessions have had
        perf_easy = np.array([compute_performance_easy(trials[k]) for k in ephys_sess_dates])
        n_trials = np.array([compute_n_trials(trials[k]) for k in ephys_sess_dates])
        psych_20 = compute_psychometric(trials_all, signed_contrast=signed_contrast, block=0.2)
        psych_80 = compute_psychometric(trials_all, signed_contrast=signed_contrast, block=0.8)
        rt = compute_median_reaction_time(trials_all, signed_contrast=signed_contrast)

        delay_sess = one.alyx.rest('sessions', 'list', subject=subj, date_range=[sess_dates[-1],
                                   sess_dates[0]], django='json__SESSION_START_DELAY_SEC__gte,900')

        if len(delay_sess) > 0 and criterion_ephys(psych_20, psych_80, n_trials, perf_easy, rt):
            status = 'ready4recording'
        elif criterion_delay(perf_easy, n_trials):
            status = 'ready4delay'
        else:
            status = 'ready4ephysrig'


def get_signed_contrast(trials):
    """Returns an array of signed contrasts in percent, where -ve values are on the left"""
    # Replace NaNs with zeros, stack and take the difference
    contrast = np.nan_to_num(np.c_[trials.contrastLeft, trials.contrastRight])
    return np.diff(contrast).flatten() * 100


def compute_performance_easy(trials):
    signed_contrast = get_signed_contrast(trials)
    easy_trials = np.where(np.abs(signed_contrast) >= 50)[0]
    return np.sum(trials.feedbackType[easy_trials] == 1)/easy_trials.shape[0]


def compute_n_trials(trials):
    return trials.choice.shape[0]


def compute_psychometric(trials, signed_contrast=None, block=None):
    if signed_contrast is None:
        signed_contrast = get_signed_contrast(trials)

    if block is None:
        block_idx = np.arange((trials.probabilityLeft.shape[0]))
    else:
        block_idx = np.where(trials.probabilityLeft == block)[0]

    contrasts, n_contrasts = np.unique(signed_contrast[block_idx], return_counts=True)
    rightward = trials.choice == -1
    # Calculate the proportion rightward for each contrast type
    prob_choose_right = np.vectorize(lambda x: np.mean(rightward[(x == signed_contrast[block_idx])]))(contrasts)

    psych, _ = psy.mle_fit_psycho(
        np.vstack([contrasts, n_contrasts, prob_choose_right]),
        P_model='erf_psycho_2gammas',
        parstart=np.array([np.mean(contrasts), 20., 0.05, 0.05]),
        parmin=np.array([np.min(contrasts), 0., 0., 0.]),
        parmax=np.array([np.max(contrasts), 100., 1, 1]))

    return psych


def compute_median_reaction_time(trials, stim_on_type='stimOn_times', signed_contrast=None):
    if not signed_contrast:
        signed_contrast = get_signed_contrast(trials)
    zero_trials = np.where(signed_contrast == 0)[0]
    reaction_time = np.median((trials.response_times - trials[stim_on_type])[signed_contrast == 0])
    return reaction_time


def criterion_1a(psych, n_trials, perf_easy):
    criterion = abs(psych[0]) < 16 and psych[1] < 19 and psych[2] < 0.2 and psych[3] < 0.2 and \
                np.all(n_trials) > 200 and np.all(perf_easy) > 0.8
    return criterion


def criterion_1b(psych, n_trials, perf_easy, rt):
    criterion = abs(psych[0]) < 10 and psych[1] < 20 and psych[2] < 0.1 and psych[3] < 0.1 and \
                np.all(n_trials) > 400 and np.all(perf_easy) > 0.9 and rt < 2
    return criterion


def criterion_ephys(psych_20, psych_80, n_trials, perf_easy, rt):
    criterion = psych_20[2] < 0.1 and psych_20[3] < 0.1 and psych_80[2] < 0.1 and psych_80[3] and \
                psych_20[0] - psych_80[0] > 5  and np.all(n_trials) > 400 and \
                np.all(perf_easy) > 0.9 and rt < 2
    return criterion


def criterion_delay(n_trials, perf_easy):
    criterion = np.any(n_trials) > 400 and np.any(perf_easy) > 0.9
    return criterion

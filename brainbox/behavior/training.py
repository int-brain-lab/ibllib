from one.api import ONE
import datetime
import re
import numpy as np
from iblutil.util import Bunch
import brainbox.behavior.pyschofit as psy
import logging
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

_logger = logging.getLogger('ibllib')

TRIALS_KEYS = ['contrastLeft',
               'contrastRight',
               'feedbackType',
               'probabilityLeft',
               'choice',
               'response_times',
               'stimOn_times']


def get_lab_training_status(lab, date=None, details=True, one=None):
    """
    Computes the training status of all alive and water restricted subjects in a specified lab

    :param lab: lab name (must match the name registered on Alyx)
    :type lab: string
    :param date: the date from which to compute training status from. If not specified will compute
    from the latest date with available data
    :type date: string of format 'YYYY-MM-DD'
    :param details: whether to display all information about training status computation e.g
    performance, number of trials, psychometric fit parameters
    :type details: bool
    :param one: instantiation of ONE class
    """
    one = one or ONE()
    subj_lab = one.alyx.rest('subjects', 'list', lab=lab, alive=True, water_restricted=True)
    subjects = [subj['nickname'] for subj in subj_lab]
    for subj in subjects:
        get_subject_training_status(subj, date=date, details=details, one=one)


def get_subject_training_status(subj, date=None, details=True, one=None):
    """
    Computes the training status of specified subject

    :param subj: subject nickname (must match the name registered on Alyx)
    :type subj: string
    :param date: the date from which to compute training status from. If not specified will compute
    from the latest date with available data
    :type date: string of format 'YYYY-MM-DD'
    :param details: whether to display all information about training status computation e.g
    performance, number of trials, psychometric fit parameters
    :type details: bool
    :param one: instantiation of ONE class
    """
    one = one or ONE()

    trials, task_protocol, ephys_sess, n_delay = get_sessions(subj, date=date, one=one)
    if not trials:
        return
    sess_dates = list(trials.keys())
    status, info = get_training_status(trials, task_protocol, ephys_sess, n_delay)

    if details:
        if np.any(info.get('psych')):
            display_status(subj, sess_dates, status, perf_easy=info.perf_easy,
                           n_trials=info.n_trials, psych=info.psych, rt=info.rt)
        elif np.any(info.get('psych_20')):
            display_status(subj, sess_dates, status, perf_easy=info.perf_easy,
                           n_trials=info.n_trials, psych_20=info.psych_20, psych_80=info.psych_80,
                           rt=info.rt)
    else:
        display_status(subj, sess_dates, status)


def get_sessions(subj, date=None, one=None):
    """
    Download and load in training data for a specfied subject. If a date is given it will load data
    from the three (or as many are available) previous sessions up to the specified date, if not it
    will load data from the last three training sessions that have data available

    :param subj: subject nickname (must match the name registered on Alyx)
    :type subj: string
    :param date: the date from which to compute training status from. If not specified will compute
    from the latest date with available data
    :type date: string of format 'YYYY-MM-DD'
    :param one: instantiation of ONE class
    :returns:
        - trials - dict of trials objects where each key is the session date
        - task_protocol - list of the task protocol used for each of the sessions
        - ephys_sess_data - list of dates where training was conducted on ephys rig. Empty list if
                            all sessions on training rig
        - n_delay - number of sessions on ephys rig that had delay prior to starting session
                    > 15min. Returns 0 is no sessions detected
    """
    one = one or ONE()

    if date is None:
        # compute from yesterday
        specified_date = (datetime.date.today() - datetime.timedelta(days=1))
        latest_sess = specified_date.strftime("%Y-%m-%d")
        latest_minus_week = (datetime.date.today() -
                             datetime.timedelta(days=8)).strftime("%Y-%m-%d")
    else:
        # compute from the date specified
        specified_date = datetime.datetime.strptime(date, '%Y-%m-%d')
        latest_minus_week = (specified_date - datetime.timedelta(days=7)).strftime("%Y-%m-%d")
        latest_sess = date

    sessions = one.alyx.rest('sessions', 'list', subject=subj, date_range=[latest_minus_week,
                             latest_sess], dataset_types='trials.goCueTrigger_times')

    # If not enough sessions in the last week, then just fetch them all
    if len(sessions) < 3:
        specified_date_plus = (specified_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        django_query = 'start_time__lte,' + specified_date_plus
        sessions = one.alyx.rest('sessions', 'list', subject=subj,
                                 dataset_types='trials.goCueTrigger_times', django=django_query)

        # If still 0 sessions then return with warning
        if len(sessions) == 0:
            _logger.warning(f"No training sessions detected for {subj}")
            return [None] * 4

    trials = Bunch()
    task_protocol = []
    sess_dates = []
    if len(sessions) < 3:
        for n, _ in enumerate(sessions):
            trials_ = one.load_object(sessions[n]['url'].split('/')[-1], 'trials')

            if trials_:
                task_protocol.append(re.search('tasks_(.*)Choice',
                                     sessions[n]['task_protocol']).group(1))
                sess_dates.append(sessions[n]['start_time'][:10])
                trials[sessions[n]['start_time'][:10]] = trials_

    else:
        n = 0
        while len(trials) < 3:
            trials_ = one.load_object(sessions[n]['url'].split('/')[-1], 'trials')

            if trials_:
                task_protocol.append(re.search('tasks_(.*)Choice',
                                     sessions[n]['task_protocol']).group(1))
                sess_dates.append(sessions[n]['start_time'][:10])
                trials[sessions[n]['start_time'][:10]] = trials_

            n += 1

    if not np.any(np.array(task_protocol) == 'training'):
        ephys_sess = one.alyx.rest('sessions', 'list', subject=subj,
                                   date_range=[sess_dates[-1], sess_dates[0]],
                                   django='json__PYBPOD_BOARD__icontains,ephys')
        if len(ephys_sess) > 0:
            ephys_sess_dates = [sess['start_time'][:10] for sess in ephys_sess]

            n_delay = len(one.alyx.rest('sessions', 'list', subject=subj,
                                        date_range=[sess_dates[-1], sess_dates[0]],
                                        django='json__SESSION_START_DELAY_SEC__gte,900'))
        else:
            ephys_sess_dates = []
            n_delay = 0
    else:
        ephys_sess_dates = []
        n_delay = 0

    return trials, task_protocol, ephys_sess_dates, n_delay


def get_training_status(trials, task_protocol, ephys_sess_dates, n_delay):
    """
    Compute training status of a subject from three consecutive training datasets

    :param trials: dict containing trials objects from three consective training sessions
    :type trials: Bunch
    :param task_protocol: task protocol used for the three training session, can be 'training',
    'biased' or 'ephys'
    :type task_protocol: list of strings
    :param ephys_sess_dates: dates of sessions conducted on ephys rig
    :type ephys_sess_dates: list of strings
    :param n_delay: number of sessions on ephys rig with delay before start > 15 min
    :type n_delay: int
    :returns:
        - status - training status of subject
        - info - Bunch containing performance metrics that decide training status e.g performance
                 on easy trials, number of trials, psychometric fit parameters, reaction time
    """

    info = Bunch()
    trials_all = concatenate_trials(trials)

    # Case when all sessions are trainingChoiceWorld
    if np.all(np.array(task_protocol) == 'training'):
        signed_contrast = get_signed_contrast(trials_all)
        (info.perf_easy, info.n_trials,
         info.psych, info.rt) = compute_training_info(trials, trials_all)
        if not np.any(signed_contrast == 0):
            status = 'in training'
        else:
            if criterion_1b(info.psych, info.n_trials, info.perf_easy, info.rt):
                status = 'trained 1b'
            elif criterion_1a(info.psych, info.n_trials, info.perf_easy):
                status = 'trained 1a'
            else:
                status = 'in training'

        return status, info

    # Case when there are < 3 biasedChoiceWorld sessions after reaching trained_1b criterion
    if ~np.all(np.array(task_protocol) == 'training') and \
            np.any(np.array(task_protocol) == 'training'):
        status = 'trained 1b'
        (info.perf_easy, info.n_trials,
         info.psych, info.rt) = compute_training_info(trials, trials_all)

        return status, info

    # Case when there is biasedChoiceWorld or ephysChoiceWorld in last three sessions
    if not np.any(np.array(task_protocol) == 'training'):

        (info.perf_easy, info.n_trials,
         info.psych_20, info.psych_80,
         info.rt) = compute_bias_info(trials, trials_all)
        # We are still on training rig and so all sessions should be biased
        if len(ephys_sess_dates) == 0:
            assert np.all(np.array(task_protocol) == 'biased')
            if criterion_ephys(info.psych_20, info.psych_80, info.n_trials, info.perf_easy,
                               info.rt):
                status = 'ready4ephysrig'
            else:
                status = 'trained 1b'

        elif len(ephys_sess_dates) < 3:
            assert all(date in trials for date in ephys_sess_dates)
            perf_ephys_easy = np.array([compute_performance_easy(trials[k]) for k in
                                        ephys_sess_dates])
            n_ephys_trials = np.array([compute_n_trials(trials[k]) for k in ephys_sess_dates])

            if criterion_delay(n_ephys_trials, perf_ephys_easy):
                status = 'ready4delay'
            else:
                status = 'ready4ephysrig'

        elif len(ephys_sess_dates) >= 3:
            if n_delay > 0 and \
                    criterion_ephys(info.psych_20, info.psych_80, info.n_trials, info.perf_easy,
                                    info.rt):
                status = 'ready4recording'
            elif criterion_delay(info.n_trials, info.perf_easy):
                status = 'ready4delay'
            else:
                status = 'ready4ephysrig'

        return status, info


def display_status(subj, sess_dates, status, perf_easy=None, n_trials=None, psych=None,
                   psych_20=None, psych_80=None, rt=None):
    """
    Display training status of subject to terminal

    :param subj: subject nickname
    :type subj: string
    :param sess_dates: training session dates used to determine training status
    :type sess_dates: list of strings
    :param status: training status of subject
    :type status: string
    :param perf_easy: performance on easy trials for each training sessions
    :type perf_easy: np.array
    :param n_trials: number of trials for each training sessions
    :type n_trials: np.array
    :param psych: parameters of psychometric curve fit to data from all training sessions
    :type psych: np.array - bias, threshold, lapse high, lapse low
    :param psych_20: parameters of psychometric curve fit to data in 20 (probability left) block
    from all training sessions
    :type psych_20: np.array - bias, threshold, lapse high, lapse low
    :param psych_80: parameters of psychometric curve fit to data in 80 (probability left) block
    from all training sessions
    :type psych_80: np.array - bias, threshold, lapse high, lapse low
    :param rt: median reaction time on zero contrast trials across all training sessions (if nan
    indicates no zero contrast stimuli in training sessions)
    """

    if perf_easy is None:
        print(f"\n{subj} : {status} \nSession dates=[{sess_dates[0]}, {sess_dates[1]}, "
              f"{sess_dates[2]}]")
    elif psych_20 is None:
        print(f"\n{subj} : {status} \nSession dates={[x for x in sess_dates]}, "
              f"Perf easy={[np.around(pe,2) for pe in perf_easy]}, "
              f"N trials={[nt for nt in n_trials]} "
              f"\nPsych fit over last 3 sessions: "
              f"bias={np.around(psych[0],2)}, thres={np.around(psych[1],2)}, "
              f"lapse_low={np.around(psych[2],2)}, lapse_high={np.around(psych[3],2)} "
              f"\nMedian reaction time at 0 contrast over last 3 sessions = "
              f"{np.around(rt,2)}")

    else:
        print(f"\n{subj} : {status} \nSession dates={[x for x in sess_dates]}, "
              f"Perf easy={[np.around(pe,2) for pe in perf_easy]}, "
              f"N trials={[nt for nt in n_trials]} "
              f"\nPsych fit over last 3 sessions (20): "
              f"bias={np.around(psych_20[0],2)}, thres={np.around(psych_20[1],2)}, "
              f"lapse_low={np.around(psych_20[2],2)}, lapse_high={np.around(psych_20[3],2)} "
              f"\nPsych fit over last 3 sessions (80): bias={np.around(psych_80[0],2)}, "
              f"thres={np.around(psych_80[1],2)}, lapse_low={np.around(psych_80[2],2)}, "
              f"lapse_high={np.around(psych_80[3],2)} "
              f"\nMedian reaction time at 0 contrast over last 3 sessions = "
              f"{np.around(rt, 2)}")


def concatenate_trials(trials):
    """
    Concatenate trials from different training sessions

    :param trials: dict containing trials objects from three consecutive training sessions,
    keys are session dates
    :type trials: Bunch
    :return: trials object with data concatenated over three training sessions
    :rtype: dict
    """
    trials_all = Bunch()
    for k in TRIALS_KEYS:
        trials_all[k] = np.concatenate(list(trials[kk][k] for kk in trials.keys()))

    return trials_all


def compute_training_info(trials, trials_all):
    """
    Compute all relevant performance metrics for when subject is on trainingChoiceWorld

    :param trials: dict containing trials objects from three consective training sessions,
    keys are session dates
    :type trials: Bunch
    :param trials_all: trials object with data concatenated over three training sessions
    :type trials_all: Bunch
    :returns:
        - perf_easy - performance of easy trials for each session
        - n_trials - number of trials in each session
        - psych - parameters for psychometric curve fit to all sessions
        - rt - median reaction time for zero contrast stimuli over all sessions
    """

    signed_contrast = get_signed_contrast(trials_all)
    perf_easy = np.array([compute_performance_easy(trials[k]) for k in trials.keys()])
    n_trials = np.array([compute_n_trials(trials[k]) for k in trials.keys()])
    psych = compute_psychometric(trials_all, signed_contrast=signed_contrast)
    rt = compute_median_reaction_time(trials_all, contrast=0, signed_contrast=signed_contrast)

    return perf_easy, n_trials, psych, rt


def compute_bias_info(trials, trials_all):
    """
    Compute all relevant performance metrics for when subject is on biasedChoiceWorld

    :param trials: dict containing trials objects from three consective training sessions,
    keys are session dates
    :type trials: Bunch
    :param trials_all: trials object with data concatenated over three training sessions
    :type trials_all: Bunch
    :returns:
        - perf_easy - performance of easy trials for each session
        - n_trials - number of trials in each session
        - psych_20 - parameters for psychometric curve fit to trials in 20 block over all sessions
        - psych_80 - parameters for psychometric curve fit to trials in 80 block over all sessions
        - rt - median reaction time for zero contrast stimuli over all sessions
    """

    signed_contrast = get_signed_contrast(trials_all)
    perf_easy = np.array([compute_performance_easy(trials[k]) for k in trials.keys()])
    n_trials = np.array([compute_n_trials(trials[k]) for k in trials.keys()])
    psych_20 = compute_psychometric(trials_all, signed_contrast=signed_contrast, block=0.2)
    psych_80 = compute_psychometric(trials_all, signed_contrast=signed_contrast, block=0.8)
    rt = compute_median_reaction_time(trials_all, contrast=0, signed_contrast=signed_contrast)

    return perf_easy, n_trials, psych_20, psych_80, rt


def get_signed_contrast(trials):
    """
    Compute signed contrast from trials object

    :param trials: trials object that must contain contrastLeft and contrastRight keys
    :type trials: dict
    returns: array of signed contrasts in percent, where -ve values are on the left
    """
    # Replace NaNs with zeros, stack and take the difference
    contrast = np.nan_to_num(np.c_[trials['contrastLeft'], trials['contrastRight']])
    return np.diff(contrast).flatten() * 100


def compute_performance_easy(trials):
    """
    Compute performance on easy trials (stimulus >= 50 %) from trials object

    :param trials: trials object that must contain contrastLeft, contrastRight and feedbackType
    keys
    :type trials: dict
    returns: float containing performance on easy contrast trials
    """
    signed_contrast = get_signed_contrast(trials)
    easy_trials = np.where(np.abs(signed_contrast) >= 50)[0]
    return np.sum(trials['feedbackType'][easy_trials] == 1) / easy_trials.shape[0]


def compute_performance(trials, signed_contrast=None, block=None, prob_right=False):
    """
    Compute performance on all trials at each contrast level from trials object

    :param trials: trials object that must contain contrastLeft, contrastRight and feedbackType
    keys
    :type trials: dict
    returns: float containing performance on easy contrast trials
    """
    if signed_contrast is None:
        signed_contrast = get_signed_contrast(trials)

    if block is None:
        block_idx = np.full(trials.probabilityLeft.shape, True, dtype=bool)
    else:
        block_idx = trials.probabilityLeft == block

    if not np.any(block_idx):
        return np.nan * np.zeros(3)

    contrasts, n_contrasts = np.unique(signed_contrast[block_idx], return_counts=True)

    if not prob_right:
        correct = trials.feedbackType == 1
        performance = np.vectorize(lambda x: np.mean(correct[(x == signed_contrast) & block_idx]))(contrasts)
    else:
        rightward = trials.choice == -1
        # Calculate the proportion rightward for each contrast type
        performance = np.vectorize(lambda x: np.mean(rightward[(x == signed_contrast) & block_idx]))(contrasts)

    return performance, contrasts, n_contrasts


def compute_n_trials(trials):
    """
    Compute number of trials in trials object

    :param trials: trials object
    :type trials: dict
    returns: int containing number of trials in session
    """
    return trials['choice'].shape[0]


def compute_psychometric(trials, signed_contrast=None, block=None, plotting=False):
    """
    Compute psychometric fit parameters for trials object

    :param trials: trials object that must contain contrastLeft, contrastRight and probabilityLeft
    :type trials: dict
    :param signed_contrast: array of signed contrasts in percent, where -ve values are on the left
    :type signed_contrast: np.array
    :param block: biased block can be either 0.2 or 0.8
    :type block: float
    :return: array of psychometric fit parameters - bias, threshold, lapse high, lapse low
    """

    if signed_contrast is None:
        signed_contrast = get_signed_contrast(trials)

    if block is None:
        block_idx = np.full(trials.probabilityLeft.shape, True, dtype=bool)
    else:
        block_idx = trials.probabilityLeft == block

    if not np.any(block_idx):
        return np.nan * np.zeros(4)

    prob_choose_right, contrasts, n_contrasts = compute_performance(trials, signed_contrast=signed_contrast, block=block,
                                                                    prob_right=True)

    if plotting:
        psych, _ = psy.mle_fit_psycho(
            np.vstack([contrasts, n_contrasts, prob_choose_right]),
            P_model='erf_psycho_2gammas',
            parstart=np.array([0., 40., 0.1, 0.1]),
            parmin=np.array([-50., 10., 0., 0.]),
            parmax=np.array([50., 50., 0.2, 0.2]),
            nfits=10)
    else:

        psych, _ = psy.mle_fit_psycho(
            np.vstack([contrasts, n_contrasts, prob_choose_right]),
            P_model='erf_psycho_2gammas',
            parstart=np.array([np.mean(contrasts), 20., 0.05, 0.05]),
            parmin=np.array([np.min(contrasts), 0., 0., 0.]),
            parmax=np.array([np.max(contrasts), 100., 1, 1]))

    return psych


def compute_median_reaction_time(trials, stim_on_type='stimOn_times', contrast=None, signed_contrast=None):
    """
    Compute median reaction time on zero contrast trials from trials object

    :param trials: trials object that must contain response_times and stimOn_times
    :type trials: dict
    :param stim_on_type: feedback from which to compute the reaction time. Default is stimOn_times
    i.e when stimulus is presented
    :type stim_on_type: string (must be a valid key in trials object)
    :param signed_contrast: array of signed contrasts in percent, where -ve values are on the left
    :type signed_contrast: np.array
    :return: float of median reaction time at zero contrast (returns nan if no zero contrast
    trials in trials object)
    """
    if signed_contrast is None:
        signed_contrast = get_signed_contrast(trials)

    if contrast is None:
        contrast_idx = np.full(trials.probabilityLeft.shape, True, dtype=bool)
    else:
        contrast_idx = signed_contrast == contrast

    if np.any(contrast_idx):
        reaction_time = np.nanmedian((trials.response_times - trials[stim_on_type])
                                     [contrast_idx])
    else:
        reaction_time = np.nan

    return reaction_time


def compute_reaction_time(trials, stim_on_type='stimOn_times', signed_contrast=None, block=None):
    """
    Compute median reaction time for all contrasts
    :param trials: trials object that must contain response_times and stimOn_times
    :param stim_on_type:
    :param signed_contrast:
    :param block:
    :return:
    """

    if signed_contrast is None:
        signed_contrast = get_signed_contrast(trials)

    if block is None:
        block_idx = np.full(trials.probabilityLeft.shape, True, dtype=bool)
    else:
        block_idx = trials.probabilityLeft == block

    contrasts, n_contrasts = np.unique(signed_contrast[block_idx], return_counts=True)
    reaction_time = np.vectorize(lambda x: np.nanmedian((trials.response_times - trials[stim_on_type])
                                                        [(x == signed_contrast) & block_idx]))(contrasts)

    return reaction_time, contrasts, n_contrasts


def criterion_1a(psych, n_trials, perf_easy):
    """
    Returns bool indicating whether criterion for trained_1a is met. All criteria documented here
    (https://figshare.com/articles/preprint/A_standardized_and_reproducible_method_to_measure_
    decision-making_in_mice_Appendix_2_IBL_protocol_for_mice_training/11634729)
    """

    criterion = (abs(psych[0]) < 16 and psych[1] < 19 and psych[2] < 0.2 and psych[3] < 0.2 and
                 np.all(n_trials > 200) and np.all(perf_easy > 0.8))
    return criterion


def criterion_1b(psych, n_trials, perf_easy, rt):
    """
    Returns bool indicating whether criterion for trained_1b is met.
    """
    criterion = (abs(psych[0]) < 10 and psych[1] < 20 and psych[2] < 0.1 and psych[3] < 0.1 and
                 np.all(n_trials > 400) and np.all(perf_easy > 0.9) and rt < 2)
    return criterion


def criterion_ephys(psych_20, psych_80, n_trials, perf_easy, rt):
    """
    Returns bool indicating whether criterion for ready4ephysrig or ready4recording is met.
    """
    criterion = (psych_20[2] < 0.1 and psych_20[3] < 0.1 and psych_80[2] < 0.1 and psych_80[3] and
                 psych_80[0] - psych_20[0] > 5 and np.all(n_trials > 400) and
                 np.all(perf_easy > 0.9) and rt < 2)
    return criterion


def criterion_delay(n_trials, perf_easy):
    """
    Returns bool indicating whether criterion for ready4delay is met.
    """
    criterion = np.any(n_trials > 400) and np.any(perf_easy > 0.9)
    return criterion


def plot_psychometric(trials, ax=None, title=None, **kwargs):
    """
    Function to plot pyschometric curve plots a la datajoint webpage
    :param trials:
    :return:
    """

    signed_contrast = get_signed_contrast(trials)
    contrasts_fit = np.arange(-100, 100)

    prob_right_50, contrasts_50, _ = compute_performance(trials, signed_contrast=signed_contrast, block=0.5, prob_right=True)
    pars_50 = compute_psychometric(trials, signed_contrast=signed_contrast, block=0.5, plotting=True)
    prob_right_fit_50 = psy.erf_psycho_2gammas(pars_50, contrasts_fit)

    prob_right_20, contrasts_20, _ = compute_performance(trials, signed_contrast=signed_contrast, block=0.2, prob_right=True)
    pars_20 = compute_psychometric(trials, signed_contrast=signed_contrast, block=0.2, plotting=True)
    prob_right_fit_20 = psy.erf_psycho_2gammas(pars_20, contrasts_fit)

    prob_right_80, contrasts_80, _ = compute_performance(trials, signed_contrast=signed_contrast, block=0.8, prob_right=True)
    pars_80 = compute_psychometric(trials, signed_contrast=signed_contrast, block=0.8, plotting=True)
    prob_right_fit_80 = psy.erf_psycho_2gammas(pars_80, contrasts_fit)

    cmap = sns.diverging_palette(20, 220, n=3, center="dark")

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    # TODO error bars

    fit_50 = ax.plot(contrasts_fit, prob_right_fit_50, color=cmap[1])
    data_50 = ax.scatter(contrasts_50, prob_right_50, color=cmap[1])
    fit_20 = ax.plot(contrasts_fit, prob_right_fit_20, color=cmap[0])
    data_20 = ax.scatter(contrasts_20, prob_right_20, color=cmap[0])
    fit_80 = ax.plot(contrasts_fit, prob_right_fit_80, color=cmap[2])
    data_80 = ax.scatter(contrasts_80, prob_right_80, color=cmap[2])
    ax.legend([fit_50[0], data_50, fit_20[0], data_20, fit_80[0], data_80],
              ['p_left=0.5 fit', 'p_left=0.5 data', 'p_left=0.2 fit', 'p_left=0.2 data', 'p_left=0.8 fit', 'p_left=0.8 data'],
              loc='upper left')
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Probability choosing right')
    ax.set_xlabel('Contrasts')
    if title:
        ax.set_title(title)

    return fig, ax


def plot_reaction_time(trials, ax=None, title=None, **kwargs):
    """
    Function to plot reaction time against contrast a la datajoint webpage (inversed for some reason??)
    :param trials:
    :return:
    """

    signed_contrast = get_signed_contrast(trials)
    reaction_50, contrasts_50, _ = compute_reaction_time(trials, signed_contrast=signed_contrast, block=0.5)
    reaction_20, contrasts_20, _ = compute_reaction_time(trials, signed_contrast=signed_contrast, block=0.2)
    reaction_80, contrasts_80, _ = compute_reaction_time(trials, signed_contrast=signed_contrast, block=0.8)

    cmap = sns.diverging_palette(20, 220, n=3, center="dark")

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    data_50 = ax.plot(contrasts_50, reaction_50, '-o', color=cmap[1])
    data_20 = ax.plot(contrasts_20, reaction_20, '-o', color=cmap[0])
    data_80 = ax.plot(contrasts_80, reaction_80, '-o', color=cmap[2])

    # TODO error bars

    ax.legend([data_50[0], data_20[0], data_80[0]],
              ['p_left=0.5 data', 'p_left=0.2 data', 'p_left=0.8 data'],
              loc='upper left')
    ax.set_ylabel('Reaction time (s)')
    ax.set_xlabel('Contrasts')

    if title:
        ax.set_title(title)

    return fig, ax


def plot_reaction_time_over_trials(trials, stim_on_type='stimOn_times', ax=None, title=None, **kwargs):
    """
    Function to plot reaction time with trial number a la datajoint webpage

    :param trials:
    :param stim_on_type:
    :param ax:
    :param title:
    :param kwargs:
    :return:
    """

    reaction_time = pd.DataFrame()
    reaction_time['reaction_time'] = trials.response_times - trials[stim_on_type]
    reaction_time.index = reaction_time.index + 1
    reaction_time_rolled = reaction_time['reaction_time'].rolling(window=10).median()
    reaction_time_rolled = reaction_time_rolled.where((pd.notnull(reaction_time_rolled)), None)
    reaction_time = reaction_time.where((pd.notnull(reaction_time)), None)

    if not ax:
        fig, ax = plt.subplots(**kwargs)
    else:
        fig = plt.gcf()

    ax.scatter(np.arange(len(reaction_time.values)), reaction_time.values, s=16, color='darkgray')
    ax.plot(np.arange(len(reaction_time_rolled.values)), reaction_time_rolled.values, color='k', linewidth=2)
    ax.set_yscale('log')
    ax.set_ylim(0.1, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_ylabel('Reaction time (s)')
    ax.set_xlabel('Trial number')
    if title:
        ax.set_title(title)

    return fig, ax

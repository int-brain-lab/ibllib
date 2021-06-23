import logging
from one.api import ONE
import datetime
import re
import numpy as np
from iblutil.util import Bunch
import brainbox.behavior.pyschofit as psy

logger = logging.getLogger('ibllib')


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
                             latest_sess], dataset_types='trials.intervals')

    # If not enough sessions in the last week, then just fetch them all
    if len(sessions) < 3:
        specified_date_plus = (specified_date + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        django_query = 'start_time__lte,' + specified_date_plus
        sessions = one.alyx.rest('sessions', 'list', subject=subj,
                                 dataset_types='trials.intervals', django=django_query)

        # If still 0 sessions then return with warning
        if len(sessions) == 0:
            logger.warning(f"No training sessions detected for {subj}")
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
            assert(np.all(np.array(task_protocol) == 'biased'))
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
        logger.info(f"\n{subj} : {status} \nSession dates=[{sess_dates[0]}, {sess_dates[1]}, "
                    f"{sess_dates[2]}]")
    elif psych_20 is None:
        logger.info(f"\n{subj} : {status} \nSession dates={[x for x in sess_dates]}, "
                    f"Perf easy={[np.around(pe,2) for pe in perf_easy]}, "
                    f"N trials={[nt for nt in n_trials]} "
                    f"\nPsych fit over last 3 sessions: "
                    f"bias={np.around(psych[0],2)}, thres={np.around(psych[1],2)}, "
                    f"lapse_low={np.around(psych[2],2)}, lapse_high={np.around(psych[3],2)} "
                    f"\nMedian reaction time at 0 contrast over last 3 sessions = "
                    f"{np.around(rt,2)}")

    else:
        logger.info(f"\n{subj} : {status} \nSession dates={[x for x in sess_dates]}, "
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

    :param trials: dict containing trials objects from three consective training sessions,
    keys are session dates
    :type trials: Bunch
    :return: trials object with data concatenated over three training sessions
    :rtype: dict
    """
    trials_all = Bunch()
    for k in trials[list(trials.keys())[0]].keys():
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
    rt = compute_median_reaction_time(trials_all, signed_contrast=signed_contrast)

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
    rt = compute_median_reaction_time(trials_all, signed_contrast=signed_contrast)

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


def compute_n_trials(trials):
    """
    Compute number of trials in trials object

    :param trials: trials object
    :type trials: dict
    returns: int containing number of trials in session
    """
    return trials['choice'].shape[0]


def compute_psychometric(trials, signed_contrast=None, block=None):
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

    contrasts, n_contrasts = np.unique(signed_contrast[block_idx], return_counts=True)
    rightward = trials.choice == -1
    # Calculate the proportion rightward for each contrast type
    prob_choose_right = np.vectorize(lambda x: np.mean(rightward[(x == signed_contrast) &
                                                                 block_idx]))(contrasts)

    psych, _ = psy.mle_fit_psycho(
        np.vstack([contrasts, n_contrasts, prob_choose_right]),
        P_model='erf_psycho_2gammas',
        parstart=np.array([np.mean(contrasts), 20., 0.05, 0.05]),
        parmin=np.array([np.min(contrasts), 0., 0., 0.]),
        parmax=np.array([np.max(contrasts), 100., 1, 1]))

    return psych


def compute_median_reaction_time(trials, stim_on_type='stimOn_times', signed_contrast=None):
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
    zero_trials = (trials.response_times - trials[stim_on_type])[signed_contrast == 0]
    if np.any(zero_trials):
        reaction_time = np.nanmedian((trials.response_times - trials[stim_on_type])
                                     [signed_contrast == 0])
    else:
        reaction_time = np.nan

    return reaction_time


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

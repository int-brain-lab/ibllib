from oneibl.one import ONE
import datetime
import re
import numpy as np
from brainbox.core import Bunch
import logging
import ibl_pipeline.utils.psychofit as psy

logger = logging.getLogger('ibllib')


def get_lab_training_status(lab, date=None, details=True, one=None):
    if not one:
        one = ONE()
    subj_lab = one.alyx.rest('subjects', 'list', lab=lab, alive=True, water_restricted=True)
    subjects = [subj['nickname'] for subj in subj_lab]
    for subj in subjects:
        get_subject_training_status(subj, date=date, details=details, one=one)


def get_subject_training_status(subj, date=None, details=True, one=None):
    if not one:
        one = ONE()

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
    if not one:
        one = ONE()

    if date is None:
        # compute from yesterday
        latest_sess = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
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
    info = Bunch()
    trials_all = concatenate_trials(trials)

    # Case when all sessions are trainingChoiceWorld
    if np.all(np.array(task_protocol) == 'training'):
        print('training=3')
        signed_contrast = get_signed_contrast(trials_all)
        (info.perf_easy, info.n_trials,
         info.psych, info.rt) = compute_training_info(trials, trials_all, signed_contrast)
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
        print('bias<3')
        status = 'trained 1b'
        signed_contrast = get_signed_contrast(trials_all)
        (info.perf_easy, info.n_trials,
         info.psych, info.rt) = compute_training_info(trials, trials_all, signed_contrast)

        return status, info

    # Case when there is biasedChoiceWorld or ephysChoiceWorld in last three sessions
    if not np.any(np.array(task_protocol) == 'training'):

        signed_contrast = get_signed_contrast(trials_all)
        (info.perf_easy, info.n_trials,
         info.psych_20, info.psych_80,
         info.rt) = compute_bias_info(trials, trials_all, signed_contrast)
        # We are still on training rig and so all sessions should be biased
        if len(ephys_sess_dates) == 0:
            print('ephys=0')
            assert(np.all(np.array(task_protocol) == 'biased'))
            if criterion_ephys(info.psych_20, info.psych_80, info.n_trials, info.perf_easy,
                               info.rt):
                status = 'ready4ephysrig'
            else:
                status = 'trained 1b'

        elif len(ephys_sess_dates) < 3:
            print('ephys<3')
            assert(np.all(np.array([date in trials for date in ephys_sess_dates])))
            perf_ephys_easy = np.array([compute_performance_easy(trials[k]) for k in
                                        ephys_sess_dates])
            n_ephys_trials = np.array([compute_n_trials(trials[k]) for k in ephys_sess_dates])

            if criterion_delay(n_ephys_trials, perf_ephys_easy):
                status = 'ready4delay'
            else:
                status = 'ready4ephysrig'

        elif len(ephys_sess_dates) >= 3:
            print('ephys>3')

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
    trials_all = Bunch()
    for k in trials[list(trials.keys())[0]].keys():
        trials_all[k] = np.concatenate(list(trials[kk][k] for kk in trials.keys()))

    return trials_all


def compute_training_info(trials, trials_all, signed_contrast):

    perf_easy = np.array([compute_performance_easy(trials[k]) for k in trials.keys()])
    n_trials = np.array([compute_n_trials(trials[k]) for k in trials.keys()])
    psych = compute_psychometric(trials_all, signed_contrast=signed_contrast)
    rt = compute_median_reaction_time(trials_all, signed_contrast=signed_contrast)

    return perf_easy, n_trials, psych, rt


def compute_bias_info(trials, trials_all, signed_contrast):

    perf_easy = np.array([compute_performance_easy(trials[k]) for k in trials.keys()])
    n_trials = np.array([compute_n_trials(trials[k]) for k in trials.keys()])
    psych_20 = compute_psychometric(trials_all, signed_contrast=signed_contrast, block=0.2)
    psych_80 = compute_psychometric(trials_all, signed_contrast=signed_contrast, block=0.8)
    rt = compute_median_reaction_time(trials_all, signed_contrast=signed_contrast)

    return perf_easy, n_trials, psych_20, psych_80, rt


def get_signed_contrast(trials):
    """Returns an array of signed contrasts in percent, where -ve values are on the left"""
    # Replace NaNs with zeros, stack and take the difference
    contrast = np.nan_to_num(np.c_[trials.contrastLeft, trials.contrastRight])
    return np.diff(contrast).flatten() * 100


def compute_performance_easy(trials):
    signed_contrast = get_signed_contrast(trials)
    easy_trials = np.where(np.abs(signed_contrast) >= 50)[0]
    return np.sum(trials.feedbackType[easy_trials] == 1) / easy_trials.shape[0]


def compute_n_trials(trials):
    return trials.choice.shape[0]


def compute_psychometric(trials, signed_contrast=None, block=None):
    if signed_contrast is None:
        signed_contrast = get_signed_contrast(trials)

    if block is None:
        block_idx = np.full(trials.probabilityLeft.shape, True, dtype=bool)
    else:
        block_idx = trials.probabilityLeft == block

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
    if signed_contrast is None:
        signed_contrast = get_signed_contrast(trials)
    zero_trials = (trials.response_times - trials[stim_on_type])[signed_contrast == 0]
    if np.any(zero_trials):
        reaction_time = np.median((trials.response_times - trials[stim_on_type])
                                  [signed_contrast == 0])
    else:
        reaction_time = np.nan

    return reaction_time


def criterion_1a(psych, n_trials, perf_easy):
    criterion = (abs(psych[0]) < 16 and psych[1] < 19 and psych[2] < 0.2 and psych[3] < 0.2 and
                 np.all(n_trials > 200) and np.all(perf_easy > 0.8))
    return criterion


def criterion_1b(psych, n_trials, perf_easy, rt):
    criterion = (abs(psych[0]) < 10 and psych[1] < 20 and psych[2] < 0.1 and psych[3] < 0.1 and
                 np.all(n_trials > 400) and np.all(perf_easy > 0.9) and rt < 2)
    return criterion


def criterion_ephys(psych_20, psych_80, n_trials, perf_easy, rt):
    criterion = (psych_20[2] < 0.1 and psych_20[3] < 0.1 and psych_80[2] < 0.1 and psych_80[3] and
                 psych_80[0] - psych_20[0] > 5 and np.all(n_trials > 400) and
                 np.all(perf_easy > 0.9) and rt < 2)
    return criterion


def criterion_delay(n_trials, perf_easy):
    criterion = np.any(n_trials > 400) and np.any(perf_easy > 0.9)
    return criterion



# sessions = one.alyx.rest('sessions', 'list', subject='SWC_054', date_range=['2020-08-21',
# '2020-09-04'])
#
# wanted_keys = ['choice', 'contrastLeft', 'contrastRight', 'feedbackType', 'probabilityLeft',
# 'response_times', 'stimOn_times']
#
# for n, _ in enumerate(sessions):
#    trials_orig = one.load_object(sessions[n]['url'].split('/')[-1], 'trials')
#    trials_ = Bunch(zip(wanted_keys, [trials_orig[k] for k in wanted_keys]))
#    if trials_:
#        trials_['task_protocol'] = re.search('tasks_(.*)Choice',
#                                       sessions[n]['task_protocol']).group(1)
#        task_protocol.append(re.search('tasks_(.*)Choice',
#                                       sessions[n]['task_protocol']).group(1))
#        sess_dates.append(sessions[n]['start_time'][:10])
#        trials[sessions[n]['start_time'][:10]] = trials_
#
# with open('trials_test.pickle', 'wb') as f:
#    pickle.dump(trials, f, protocol=pickle.HIGHEST_PROTOCOL)
#
# with open('trials_test.pickle', 'rb') as f:
#    trials_orig = pickle.load(f)

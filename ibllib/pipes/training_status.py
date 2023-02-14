import one.alf.io as alfio
from one.alf.exceptions import ALFObjectNotFound

from ibllib.io.raw_data_loaders import load_bpod
from ibllib.oneibl.registration import _get_session_times
from ibllib.io.extractors.base import get_pipeline, get_session_extractor_type

from ibllib.plots.snapshot import ReportSnapshot
from iblutil.numerical import ismember
from brainbox.behavior import training

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from datetime import datetime
import seaborn as sns


TRAINING_STATUS = {'not_computed': (-2, (0, 0, 0, 0)),
                   'habituation': (-1, (0, 0, 0, 0)),
                   'in training': (0, (0, 0, 0, 0)),
                   'trained 1a': (1, (195, 90, 80, 255)),
                   'trained 1b': (2, (255, 153, 20, 255)),
                   'ready4ephysrig': (3, (28, 20, 255, 255)),
                   'ready4delay': (4, (117, 117, 117, 255)),
                   'ready4recording': (5, (20, 255, 91, 255))}


def get_trials_task(session_path, one):
    # TODO this eventually needs to be updated for dynamic pipeline tasks
    pipeline = get_pipeline(session_path)
    if pipeline == 'training':
        from ibllib.pipes.training_preprocessing import TrainingTrials
        task = TrainingTrials(session_path, one=one)
    elif pipeline == 'ephys':
        from ibllib.pipes.ephys_preprocessing import EphysTrials
        task = EphysTrials(session_path, one=one)
    else:
        try:
            # try and look if there is a custom extractor in the personal projects extraction class
            import projects.base
            task_type = get_session_extractor_type(session_path)
            PipelineClass = projects.base.get_pipeline(task_type)
            pipeline = PipelineClass(session_path, one)
            trials_task_name = next(task for task in pipeline.tasks if 'Trials' in task)
            task = pipeline.tasks.get(trials_task_name)
        except Exception:
            task = None

    return task


def save_path(subj_path):
    return Path(subj_path).joinpath('training.csv')


def save_dataframe(df, subj_path):
    """
    Save training dataframe to disk
    :param df: dataframe to save
    :param subj_path: path to subject folder
    :return:
    """
    df.to_csv(save_path(subj_path), index=False)


def load_existing_dataframe(subj_path):
    """
    Load training dataframe from disk, if dataframe doesn't exist returns None
    :param subj_path: path to subject folder
    :return:
    """
    df_location = save_path(subj_path)
    if df_location.exists():
        return pd.read_csv(df_location)
    else:
        df_location.parent.mkdir(exist_ok=True, parents=True)
        return None


def load_trials(sess_path, one):
    """
    Load trials data for session. First attempts to load from local session path, if this fails will attempt to download via ONE,
    if this also fails, will then attempt to re-extraxt locally
    :param sess_path: session path
    :param one: ONE instance
    :return:
    """
    # try and load trials locally
    try:
        trials = alfio.load_object(sess_path.joinpath('alf'), 'trials')
        if 'probabilityLeft' not in trials.keys():
            raise ALFObjectNotFound
    except ALFObjectNotFound:
        try:
            # attempt to download trials using ONE
            trials = one.load_object(one.path2eid(sess_path), 'trials')
            if 'probabilityLeft' not in trials.keys():
                raise ALFObjectNotFound
        except Exception:
            try:
                task = get_trials_task(sess_path, one=one)
                if task is not None:
                    task.run()
                    trials = alfio.load_object(sess_path.joinpath('alf'), 'trials')
                    if 'probabilityLeft' not in trials.keys():
                        raise ALFObjectNotFound
                else:
                    trials = None
            except Exception:  # TODO how can i make this more specific
                trials = None
    return trials


def load_combined_trials(sess_paths, one):
    """
    Load and concatenate trials for multiple sessions. Used when we want to concatenate trials for two sessions on the same day
    :param sess_paths: list of paths to sessions
    :param one: ONE instance
    :return:
    """
    trials_dict = {}
    for sess_path in sess_paths:
        trials = load_trials(Path(sess_path), one)
        if trials is not None:
            trials_dict[Path(sess_path).stem] = load_trials(Path(sess_path), one)

    return training.concatenate_trials(trials_dict)


def get_latest_training_information(sess_path, one):
    """
    Extracts the latest training status
    :param sess_path:
    :param one:
    :return:
    """

    subj_path = sess_path.parent.parent
    df = load_existing_dataframe(subj_path)

    # Find the dates and associated session paths where we don't have data stored in our dataframe
    missing_dates = check_up_to_date(subj_path, df)

    # Iterate through the dates to fill up our training dataframe
    for _, grp in missing_dates.groupby('date'):
        sess_dicts = get_training_info_for_session(grp.session_path.values, one)
        if len(sess_dicts) == 0:
            continue

        for sess_dict in sess_dicts:
            if df is None:
                df = pd.DataFrame.from_dict(sess_dict)
            else:
                df = pd.concat([df, pd.DataFrame.from_dict(sess_dict)])

    # Sort values by date and reset the index
    df = df.sort_values('date')
    df = df.reset_index(drop=True)
    # Save our dataframe
    save_dataframe(df, subj_path)

    # Now go through the backlog and compute the training status for sessions. If for example one was missing as it is cumulative
    # we need to go through and compute all the back log
    # Find the earliest date in missing dates that we need to recompute the training status for
    missing_status = find_earliest_recompute_date(df.drop_duplicates('date').reset_index(drop=True))
    for date in missing_status:
        df = compute_training_status(df, date, one)
    save_dataframe(df, subj_path)

    return df


def find_earliest_recompute_date(df):
    """
    Find the earliest date that we need to compute the training status from. Training status depends on previous sessions
    so if a session was missing and now has been added we need to recompute everything from that date onwards
    :param df:
    :return:
    """
    missing_df = df[df['training_status'] == 'not_computed']
    if len(missing_df) == 0:
        return []
    missing_df = missing_df.sort_values('date')
    first_index = missing_df.index[0]

    return df[first_index:].date.values


def compute_training_status(df, compute_date, one):
    """
    Compute the training status for compute date based on training from that session and two previous days
    :param df: training dataframe
    :param compute_date: date to compute training on
    :param one: ONE instance
    :return:
    """

    # compute_date = str(one.path2ref(session_path)['date'])
    df_temp = df[df['date'] <= compute_date]
    df_temp = df_temp.drop_duplicates('session_path')
    df_temp.sort_values('date')

    dates = df_temp.date.values

    n_dates = np.min([3, len(dates)]).astype(int)
    compute_dates = dates[(-1 * n_dates):]

    assert compute_dates[-1] == compute_date

    df_temp_group = df_temp.groupby('date')

    trials = {}
    n_delay = 0
    ephys_sessions = []
    protocol = []
    status = []
    for date in compute_dates:

        df_date = df_temp_group.get_group(date)

        # If habituation skip
        if df_date.iloc[-1]['task_protocol'] == 'habituation':
            continue
        trials[df_date.iloc[-1]['date']] = load_combined_trials(df_date.session_path.values, one)
        protocol.append(df_date.iloc[-1]['task_protocol'])
        status.append(df_date.iloc[-1]['training_status'])
        if df_date.iloc[-1]['combined_n_delay'] >= 900:  # delay of 15 mins
            n_delay += 1
        if df_date.iloc[-1]['location'] == 'ephys_rig':
            ephys_sessions.append(df_date.iloc[-1]['date'])

    n_status = np.max([-2, -1 * len(status)])
    training_status, _ = training.get_training_status(trials, protocol, ephys_sessions, n_delay)
    training_status = pass_through_training_hierachy(training_status, status[n_status])
    df.loc[df['date'] == compute_date, 'training_status'] = training_status

    return df


def pass_through_training_hierachy(status_new, status_old):
    """
    Makes sure that the new training status is not less than the one from the previous day. e.g Subject cannot regress in
    performance
    :param status_new: latest training status
    :param status_old: previous training status
    :return:
    """

    if TRAINING_STATUS[status_old][0] > TRAINING_STATUS[status_new][0]:
        return status_old
    else:
        return status_new


def compute_session_duration_delay_location(sess_path, **kwargs):
    """
    Get meta information about task. Extracts session duration, delay before session start and location of session

    Parameters
    ----------
    sess_path : pathlib.Path, str
        The session path with the pattern subject/yyyy-mm-dd/nnn.
    task_collection : str
        The location within the session path directory of task settings and data.

    Returns
    -------
    int
        The session duration in minutes, rounded to the nearest minute.
    int
        The delay between session start time and the first trial in seconds.
    str {'ephys_rig', 'training_rig'}
        The location of the session.
    """
    md, sess_data = load_bpod(sess_path, **kwargs)
    start_time, end_time = _get_session_times(sess_path, md, sess_data)
    session_duration = int((end_time - start_time).total_seconds() / 60)

    session_delay = md.get('SESSION_START_DELAY_SEC', 0)

    if 'ephys' in md.get('PYBPOD_BOARD', None):
        session_location = 'ephys_rig'
    else:
        session_location = 'training_rig'

    return session_duration, session_delay, session_location


def get_training_info_for_session(session_paths, one):
    """
    Extract the training information needed for plots for each session
    :param session_paths: list of session paths on same date
    :param one: ONE instance
    :return:
    """

    # return list of dicts to add
    sess_dicts = []
    for session_path in session_paths:
        session_path = Path(session_path)
        sess_dict = {}
        sess_dict['date'] = str(one.path2ref(session_path)['date'])
        sess_dict['session_path'] = str(session_path)
        sess_dict['task_protocol'] = get_session_extractor_type(session_path)

        if sess_dict['task_protocol'] == 'habituation':
            nan_array = np.array([np.nan])
            sess_dict['performance'], sess_dict['contrasts'], _ = (nan_array, nan_array, np.nan)
            sess_dict['performance_easy'] = np.nan
            sess_dict['reaction_time'] = np.nan
            sess_dict['n_trials'] = np.nan
            sess_dict['sess_duration'] = np.nan
            sess_dict['n_delay'] = np.nan
            sess_dict['location'] = np.nan
            sess_dict['training_status'] = 'habituation'
            sess_dict['bias_50'], sess_dict['thres_50'], sess_dict['lapsehigh_50'], sess_dict['lapselow_50'] = \
                (np.nan, np.nan, np.nan, np.nan)
            sess_dict['bias_20'], sess_dict['thres_20'], sess_dict['lapsehigh_20'], sess_dict['lapselow_20'] = \
                (np.nan, np.nan, np.nan, np.nan)
            sess_dict['bias_80'], sess_dict['thres_80'], sess_dict['lapsehigh_80'], sess_dict['lapselow_80'] = \
                (np.nan, np.nan, np.nan, np.nan)

        else:
            # if we can't compute trials then we need to pass
            trials = load_trials(session_path, one)
            if trials is None:
                continue

            sess_dict['performance'], sess_dict['contrasts'], _ = training.compute_performance(trials, prob_right=True)
            if sess_dict['task_protocol'] == 'training':
                sess_dict['bias_50'], sess_dict['thres_50'], sess_dict['lapsehigh_50'], sess_dict['lapselow_50'] = \
                    training.compute_psychometric(trials)
                sess_dict['bias_20'], sess_dict['thres_20'], sess_dict['lapsehigh_20'], sess_dict['lapselow_20'] = \
                    (np.nan, np.nan, np.nan, np.nan)
                sess_dict['bias_80'], sess_dict['thres_80'], sess_dict['lapsehigh_80'], sess_dict['lapselow_80'] = \
                    (np.nan, np.nan, np.nan, np.nan)
            else:
                sess_dict['bias_50'], sess_dict['thres_50'], sess_dict['lapsehigh_50'], sess_dict['lapselow_50'] = \
                    training.compute_psychometric(trials, block=0.5)
                sess_dict['bias_20'], sess_dict['thres_20'], sess_dict['lapsehigh_20'], sess_dict['lapselow_20'] = \
                    training.compute_psychometric(trials, block=0.2)
                sess_dict['bias_80'], sess_dict['thres_80'], sess_dict['lapsehigh_80'], sess_dict['lapselow_80'] = \
                    training.compute_psychometric(trials, block=0.8)

            sess_dict['performance_easy'] = training.compute_performance_easy(trials)
            sess_dict['reaction_time'] = training.compute_median_reaction_time(trials)
            sess_dict['n_trials'] = training.compute_n_trials(trials)
            sess_dict['sess_duration'], sess_dict['n_delay'], sess_dict['location'] = \
                compute_session_duration_delay_location(session_path)
            sess_dict['training_status'] = 'not_computed'

        sess_dicts.append(sess_dict)

    protocols = [s['task_protocol'] for s in sess_dicts]

    if len(protocols) > 0 and len(set(protocols)) != 1:
        print(f'Different protocols on same date {sess_dicts[0]["date"]} : {protocols}')

    if len(sess_dicts) > 1 and len(set(protocols)) == 1:  # Only if all protocols are the same
        print(f'{len(sess_dicts)} sessions being combined for date {sess_dicts[0]["date"]}')
        combined_trials = load_combined_trials(session_paths, one)
        performance, contrasts, _ = training.compute_performance(combined_trials, prob_right=True)
        psychs = {}
        psychs['50'] = training.compute_psychometric(trials, block=0.5)
        psychs['20'] = training.compute_psychometric(trials, block=0.2)
        psychs['80'] = training.compute_psychometric(trials, block=0.8)

        performance_easy = training.compute_performance_easy(combined_trials)
        reaction_time = training.compute_median_reaction_time(combined_trials)
        n_trials = training.compute_n_trials(combined_trials)

        sess_duration = np.nansum([s['sess_duration'] for s in sess_dicts])
        n_delay = np.nanmax([s['n_delay'] for s in sess_dicts])

        for sess_dict in sess_dicts:
            sess_dict['combined_performance'] = performance
            sess_dict['combined_contrasts'] = contrasts
            sess_dict['combined_performance_easy'] = performance_easy
            sess_dict['combined_reaction_time'] = reaction_time
            sess_dict['combined_n_trials'] = n_trials
            sess_dict['combined_sess_duration'] = sess_duration
            sess_dict['combined_n_delay'] = n_delay

            for bias in [50, 20, 80]:
                sess_dict[f'combined_bias_{bias}'] = psychs[f'{bias}'][0]
                sess_dict[f'combined_thres_{bias}'] = psychs[f'{bias}'][1]
                sess_dict[f'combined_lapsehigh_{bias}'] = psychs[f'{bias}'][2]
                sess_dict[f'combined_lapselow_{bias}'] = psychs[f'{bias}'][3]

            # Case where two sessions on same day with different number of contrasts! Oh boy
            if sess_dict['combined_performance'].size != sess_dict['performance'].size:
                sess_dict['performance'] = \
                    np.r_[sess_dict['performance'],
                          np.full(sess_dict['combined_performance'].size - sess_dict['performance'].size, np.nan)]
                sess_dict['contrasts'] = \
                    np.r_[sess_dict['contrasts'],
                          np.full(sess_dict['combined_contrasts'].size - sess_dict['contrasts'].size, np.nan)]

    else:
        for sess_dict in sess_dicts:
            sess_dict['combined_performance'] = sess_dict['performance']
            sess_dict['combined_contrasts'] = sess_dict['contrasts']
            sess_dict['combined_performance_easy'] = sess_dict['performance_easy']
            sess_dict['combined_reaction_time'] = sess_dict['reaction_time']
            sess_dict['combined_n_trials'] = sess_dict['n_trials']
            sess_dict['combined_sess_duration'] = sess_dict['sess_duration']
            sess_dict['combined_n_delay'] = sess_dict['n_delay']

            for bias in [50, 20, 80]:
                sess_dict[f'combined_bias_{bias}'] = sess_dict[f'bias_{bias}']
                sess_dict[f'combined_thres_{bias}'] = sess_dict[f'thres_{bias}']
                sess_dict[f'combined_lapsehigh_{bias}'] = sess_dict[f'lapsehigh_{bias}']
                sess_dict[f'combined_lapselow_{bias}'] = sess_dict[f'lapselow_{bias}']

    return sess_dicts


def check_up_to_date(subj_path, df):
    """
    Check which sessions on local file system are missing from the computed training table.

    Parameters
    ----------
    subj_path : pathlib.Path
        The path to the subject's dated session folders.
    df : pandas.DataFrame
        The computed training table.

    Returns
    -------
    pandas.DataFrame
        A table of dates and session paths that are missing from the computed training table.
    """
    df_session = pd.DataFrame()

    for session in alfio.iter_sessions(subj_path):
        s_df = pd.DataFrame({'date': session.parts[-2], 'session_path': str(session)}, index=[0])
        df_session = pd.concat([df_session, s_df], ignore_index=True)

    if df is None or 'combined_thres_50' not in df.columns:
        return df_session
    else:
        # recorded_session_paths = df['session_path'].values
        isin, _ = ismember(df_session.date.unique(), df.date.unique())
        missing_dates = df_session.date.unique()[~isin]
        return df_session[df_session['date'].isin(missing_dates)].sort_values('date')


def plot_trial_count_and_session_duration(df, subject):

    df = df.drop_duplicates('date').reset_index(drop=True)

    y1 = {'column': 'combined_n_trials',
          'title': 'Trial counts',
          'lim': None,
          'color': 'k',
          'join': True}

    y2 = {'column': 'combined_sess_duration',
          'title': 'Session duration (mins)',
          'lim': None,
          'color': 'r',
          'log': False,
          'join': True}

    ax = plot_over_days(df, subject, y1, y2)

    return ax


def plot_performance_easy_median_reaction_time(df, subject):
    df = df.drop_duplicates('date').reset_index(drop=True)

    y1 = {'column': 'combined_performance_easy',
          'title': 'Performance on easy trials',
          'lim': [0, 1.05],
          'color': 'k',
          'join': True}

    y2 = {'column': 'combined_reaction_time',
          'title': 'Median reaction time (s)',
          'lim': [0.1, np.nanmax([10, np.nanmax(df.combined_reaction_time.values)])],
          'color': 'r',
          'log': True,
          'join': True}
    ax = plot_over_days(df, subject, y1, y2)

    return ax


def plot_fit_params(df, subject):
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    axs = axs.ravel()

    df = df.drop_duplicates('date').reset_index(drop=True)

    cmap = sns.diverging_palette(20, 220, n=3, center="dark")

    y50 = {'column': 'combined_bias_50',
           'title': 'Bias',
           'lim': [-100, 100],
           'color': cmap[1],
           'join': False}

    y80 = {'column': 'combined_bias_80',
           'title': 'Bias',
           'lim': [-100, 100],
           'color': cmap[2],
           'join': False}

    y20 = {'column': 'combined_bias_20',
           'title': 'Bias',
           'lim': [-100, 100],
           'color': cmap[0],
           'join': False}

    plot_over_days(df, subject, y50, ax=axs[0], legend=False, title=False)
    plot_over_days(df, subject, y80, ax=axs[0], legend=False, title=False)
    plot_over_days(df, subject, y20, ax=axs[0], legend=False, title=False)
    axs[0].axhline(16, linewidth=2, linestyle='--', color='k')
    axs[0].axhline(-16, linewidth=2, linestyle='--', color='k')

    y50['column'] = 'combined_thres_50'
    y50['title'] = 'Threshold'
    y50['lim'] = [0, 100]
    y80['column'] = 'combined_thres_20'
    y80['title'] = 'Threshold'
    y20['lim'] = [0, 100]
    y20['column'] = 'combined_thres_80'
    y20['title'] = 'Threshold'
    y80['lim'] = [0, 100]

    plot_over_days(df, subject, y50, ax=axs[1], legend=False, title=False)
    plot_over_days(df, subject, y80, ax=axs[1], legend=False, title=False)
    plot_over_days(df, subject, y20, ax=axs[1], legend=False, title=False)
    axs[1].axhline(19, linewidth=2, linestyle='--', color='k')

    y50['column'] = 'combined_lapselow_50'
    y50['title'] = 'Lapse Low'
    y50['lim'] = [0, 1]
    y80['column'] = 'combined_lapselow_20'
    y80['title'] = 'Lapse Low'
    y80['lim'] = [0, 1]
    y20['column'] = 'combined_lapselow_80'
    y20['title'] = 'Lapse Low'
    y20['lim'] = [0, 1]

    plot_over_days(df, subject, y50, ax=axs[2], legend=False, title=False)
    plot_over_days(df, subject, y80, ax=axs[2], legend=False, title=False)
    plot_over_days(df, subject, y20, ax=axs[2], legend=False, title=False)
    axs[2].axhline(0.2, linewidth=2, linestyle='--', color='k')

    y50['column'] = 'combined_lapsehigh_50'
    y50['title'] = 'Lapse High'
    y50['lim'] = [0, 1]
    y80['column'] = 'combined_lapsehigh_20'
    y80['title'] = 'Lapse High'
    y80['lim'] = [0, 1]
    y20['column'] = 'combined_lapsehigh_80'
    y20['title'] = 'Lapse High'
    y20['lim'] = [0, 1]

    plot_over_days(df, subject, y50, ax=axs[3], legend=False, title=False, training_lines=True)
    plot_over_days(df, subject, y80, ax=axs[3], legend=False, title=False, training_lines=False)
    plot_over_days(df, subject, y20, ax=axs[3], legend=False, title=False, training_lines=False)
    axs[3].axhline(0.2, linewidth=2, linestyle='--', color='k')

    fig.suptitle(f'{subject} {df.iloc[-1]["date"]}: {df.iloc[-1]["training_status"]}')
    lines, labels = axs[3].get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 0.1), fancybox=True, shadow=True, ncol=5)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='p=0.5', markerfacecolor=cmap[1], markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='p=0.2', markerfacecolor=cmap[0], markersize=8),
                       Line2D([0], [0], marker='o', color='w', label='p=0.8', markerfacecolor=cmap[2], markersize=8)]
    legend2 = plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.1, -0.2), fancybox=True, shadow=True)
    fig.add_artist(legend2)

    return axs


def plot_psychometric_curve(df, subject, one):
    df = df.drop_duplicates('date').reset_index(drop=True)
    sess_path = Path(df.iloc[-1]["session_path"])
    trials = load_trials(sess_path, one)

    fig, ax1 = plt.subplots(figsize=(8, 6))

    training.plot_psychometric(trials, ax=ax1, title=f'{subject} {df.iloc[-1]["date"]}: {df.iloc[-1]["training_status"]}')

    return ax1


def plot_over_days(df, subject, y1, y2=None, ax=None, legend=True, title=True, training_lines=True):

    if ax is None:
        fig, ax1 = plt.subplots(figsize=(12, 6))
    else:
        ax1 = ax

    dates = [datetime.strptime(dat, '%Y-%m-%d') for dat in df['date']]
    if y1['join']:
        ax1.plot(dates, df[y1['column']], color=y1['color'])
    ax1.scatter(dates, df[y1['column']], color=y1['color'])
    ax1.set_ylabel(y1['title'])
    ax1.set_ylim(y1['lim'])

    if y2 is not None:
        ax2 = ax1.twinx()
        if y2['join']:
            ax2.plot(dates, df[y2['column']], color=y2['color'])
        ax2.scatter(dates, df[y2['column']], color=y2['color'])
        ax2.set_ylabel(y2['title'])
        ax2.yaxis.label.set_color(y2['color'])
        ax2.tick_params(axis='y', colors=y2['color'])
        ax2.set_ylim(y2['lim'])
        if y2['log']:
            ax2.set_yscale('log')

        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)

    month_format = mdates.DateFormatter('%b %Y')
    month_locator = mdates.MonthLocator()
    ax1.xaxis.set_major_locator(month_locator)
    ax1.xaxis.set_major_formatter(month_format)
    week_locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
    ax1.xaxis.set_minor_locator(week_locator)
    ax1.grid(True, which='minor', axis='x', linestyle='--')

    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    if training_lines:
        ax1 = add_training_lines(df, ax1)

    if title:
        ax1.set_title(f'{subject} {df.iloc[-1]["date"]}: {df.iloc[-1]["training_status"]}')

    # Put a legend below current axis
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    if legend:
        ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
                   fancybox=True, shadow=True, ncol=5)

    return ax1


def add_training_lines(df, ax):

    status = df.drop_duplicates(subset='training_status', keep='first')
    for _, st in status.iterrows():
        if TRAINING_STATUS[st['training_status']][0] <= 0:
            continue

        ax.axvline(datetime.strptime(st['date'], '%Y-%m-%d'), linewidth=2,
                   color=np.array(TRAINING_STATUS[st['training_status']][1]) / 255, label=st['training_status'])

    return ax


def plot_heatmap_performance_over_days(df, subject):

    df = df.drop_duplicates(subset=['date', 'combined_contrasts'])
    df_perf = df.pivot(index=['date'], columns=['combined_contrasts'], values=['combined_performance']).sort_values(
        by='combined_contrasts', axis=1, ascending=False)
    df_perf.index = pd.to_datetime(df_perf.index)
    full_date_range = pd.date_range(start=df_perf.index.min(), end=df_perf.index.max(), freq="D")
    df_perf = df_perf.reindex(full_date_range, fill_value=np.nan)

    n_contrasts = len(df.combined_contrasts.unique())

    dates = df_perf.index.to_pydatetime()
    dnum = mdates.date2num(dates)
    if len(dnum) > 1:
        start = dnum[0] - (dnum[1] - dnum[0]) / 2.
        stop = dnum[-1] + (dnum[1] - dnum[0]) / 2.
    else:
        start = dnum[0] + 0.5
        stop = dnum[0] + 1.5

    extent = [start, stop, 0, n_contrasts]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    im = ax1.imshow(df_perf.T.values, extent=extent, aspect="auto", cmap='PuOr')

    month_format = mdates.DateFormatter('%b %Y')
    month_locator = mdates.MonthLocator()
    ax1.xaxis.set_major_locator(month_locator)
    ax1.xaxis.set_major_formatter(month_format)
    week_locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
    ax1.xaxis.set_minor_locator(week_locator)
    ax1.grid(True, which='minor', axis='x', linestyle='--')
    ax1.set_yticks(np.arange(0.5, n_contrasts + 0.5, 1))
    ax1.set_yticklabels(np.sort(df.combined_contrasts.unique()))
    ax1.set_ylabel('Contrast (%)')
    ax1.set_xlabel('Date')
    cbar = fig.colorbar(im)
    cbar.set_label('Rightward choice (%')
    ax1.set_title(f'{subject} {df.iloc[-1]["date"]}: {df.iloc[-1]["training_status"]}')

    ax1.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)

    return ax1


def make_plots(session_path, one, df=None, save=False, upload=False):
    subject = one.path2ref(session_path)['subject']
    subj_path = session_path.parent.parent

    df = load_existing_dataframe(subj_path) if df is None else df

    df = df[df['task_protocol'] != 'habituation']

    if len(df) == 0:
        return

    ax1 = plot_trial_count_and_session_duration(df, subject)
    ax2 = plot_performance_easy_median_reaction_time(df, subject)
    ax3 = plot_heatmap_performance_over_days(df, subject)
    ax4 = plot_fit_params(df, subject)
    ax5 = plot_psychometric_curve(df, subject, one)

    outputs = []
    if save:
        save_path = Path(subj_path)
        save_name = save_path.joinpath('subj_trial_count_session_duration.png')
        outputs.append(save_name)
        ax1.get_figure().savefig(save_name, bbox_inches='tight')

        save_name = save_path.joinpath('subj_performance_easy_reaction_time.png')
        outputs.append(save_name)
        ax2.get_figure().savefig(save_name, bbox_inches='tight')

        save_name = save_path.joinpath('subj_performance_heatmap.png')
        outputs.append(save_name)
        ax3.get_figure().savefig(save_name, bbox_inches='tight')

        save_name = save_path.joinpath('subj_psychometric_fit_params.png')
        outputs.append(save_name)
        ax4[0].get_figure().savefig(save_name, bbox_inches='tight')

        save_name = save_path.joinpath('subj_psychometric_curve.png')
        outputs.append(save_name)
        ax5.get_figure().savefig(save_name, bbox_inches='tight')

    if upload:
        subj = one.alyx.rest('subjects', 'list', nickname=subject)[0]
        snp = ReportSnapshot(session_path, subj['id'], content_type='subject', one=one)
        snp.outputs = outputs
        snp.register_images(widths=['orig'])

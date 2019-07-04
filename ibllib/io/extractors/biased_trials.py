#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Tuesday, February 12th 2019, 11:49:54 am
import numpy as np
import os
import ibllib.io.raw_data_loaders as raw
from ibllib.misc import version
from ibllib.io.extractors.training_trials import (  # noqa
    check_alf_folder, get_feedbackType, get_probabilityLeft,
    get_choice, get_rewardVolume, get_feedback_times, get_feedback_times_ge5,
    get_feedback_times_lt5, get_stimOn_times, get_stimOn_times_lt5,
    get_stimOn_times_ge5, get_intervals, get_response_times, get_iti_duration,
    get_goCueTrigger_times, get_goCueOnset_times, get_included_trials_ge5,
    get_included_trials_lt5, get_included_trials, get_stimOnTrigger_times)  # noqa


def get_contrastLR(session_path, save=False, data=False, settings=False):
    """
    Get left and right contrasts from raw datafile. Optionally, saves
    _ibl_trials.contrastLeft.npy and _ibl_trials.contrastRight.npy to alf folder.

    Uses signed_contrast to create left and right contrast vectors.

    :param session_path: absolute path of session folder
    :type session_path: str
    :param save: whether to save the corresponding alf file
                 to the alf folder, defaults to False
    :type save: bool, optional
    :return: numpy.ndarray
    :rtype: dtype('float64')
    """
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}

    contrastLeft = np.array([t['contrast'] if np.sign(
        t['position']) < 0 else np.nan for t in data])
    contrastRight = np.array([t['contrast'] if np.sign(
        t['position']) > 0 else np.nan for t in data])
    # save if needed
    check_alf_folder(session_path)
    if raw.save_bool(save, '_ibl_trials.contrastLeft.npy'):
        lpath = os.path.join(session_path, 'alf', '_ibl_trials.contrastLeft.npy')
        np.save(lpath, contrastLeft)

    if raw.save_bool(save, '_ibl_trials.contrastRight.npy'):
        rpath = os.path.join(session_path, 'alf', '_ibl_trials.contrastRight.npy')
        np.save(rpath, contrastRight)

    return (contrastLeft, contrastRight)


def extract_all(session_path, save=False, data=False, settings=False):
    if not data:
        data = raw.load_data(session_path)
    if not settings:
        settings = raw.load_settings(session_path)
    if settings is None or settings['IBLRIG_VERSION_TAG'] == '':
        settings = {'IBLRIG_VERSION_TAG': '100.0.0'}
    # Version check
    if version.ge(settings['IBLRIG_VERSION_TAG'], '5.0.0'):
        feedbackType = get_feedbackType(
            session_path, save=save, data=data, settings=settings)
        contrastLeft, contrastRight = get_contrastLR(
            session_path, save=save, data=data, settings=settings)
        probabilityLeft = get_probabilityLeft(
            session_path, save=save, data=data, settings=settings)
        choice = get_choice(
            session_path, save=save, data=data, settings=settings)
        rewardVolume = get_rewardVolume(
            session_path, save=save, data=data, settings=settings)
        feedback_times = get_feedback_times(
            session_path, save=save, data=data, settings=settings)
        stimOn_times = get_stimOn_times(
            session_path, save=save, data=data, settings=settings)
        stimOnTrigger_times = get_stimOnTrigger_times(
            session_path, save=save, data=data, settings=settings)
        intervals = get_intervals(
            session_path, save=save, data=data, settings=settings)
        response_times = get_response_times(
            session_path, save=save, data=data, settings=settings)
        go_cue_trig_times = get_goCueTrigger_times(
            session_path, save=save, data=data, settings=settings)
        go_cue_times = get_goCueOnset_times(
            session_path, save=save, data=data, settings=settings)
        included = get_included_trials(
            session_path, save=save, data=data, settings=settings)
        out = {'feedbackType': feedbackType,
               'contrastLeft': contrastLeft,
               'contrastRight': contrastRight,
               'probabilityLeft': probabilityLeft,
               'session_path': session_path,
               'choice': choice,
               'rewardVolume': rewardVolume,
               'feedback_times': feedback_times,
               'stimOnTrigger_times': stimOnTrigger_times,
               'stimOn_times': stimOn_times,
               'intervals': intervals,
               'response_times': response_times,
               'goCue_times': go_cue_times,
               'goCueTrigger_times': go_cue_trig_times,
               'included': included}

    else:
        feedbackType = get_feedbackType(session_path, save=save, data=data, settings=settings)
        contrastLeft, contrastRight = get_contrastLR(
            session_path, save=save, data=data, settings=settings)
        probabilityLeft = get_probabilityLeft(
            session_path, save=save, data=data, settings=settings)
        choice = get_choice(session_path, save=save, data=data, settings=settings)
        rewardVolume = get_rewardVolume(session_path, save=save, data=data, settings=settings)
        feedback_times = get_feedback_times(session_path, save=save, data=data, settings=settings)
        stimOn_times = get_stimOn_times(session_path, save=save, data=data, settings=settings)
        intervals = get_intervals(session_path, save=save, data=data, settings=settings)
        response_times = get_response_times(session_path, save=save, data=data, settings=settings)
        iti_dur = get_iti_duration(session_path, save=save, data=data, settings=settings)
        go_cue_trig_times = get_goCueTrigger_times(
            session_path, save=save, data=data, settings=settings)
        go_cue_times = get_goCueOnset_times(session_path, save=save, data=data, settings=settings)
        # Missing datasettypes
        # _ibl_trials.deadTime
        out = {'feedbackType': feedbackType,
               'contrastLeft': contrastLeft,
               'contrastRight': contrastRight,
               'probabilityLeft': probabilityLeft,
               'session_path': session_path,
               'choice': choice,
               'rewardVolume': rewardVolume,
               'feedback_times': feedback_times,
               'stimOn_times': stimOn_times,
               'intervals': intervals,
               'response_times': response_times,
               'iti_dur': iti_dur,
               'goCue_times': go_cue_times,
               'goCueTrigger_times': go_cue_trig_times}
    return out


if __name__ == "__main__":
    sess = '/home/nico/Projects/IBL/IBL-github/iblrig/scratch/test_iblrig_data/Subjects/ZM_1085/2019-02-12/002'  # noqa
    alf_data = extract_all(sess)
    print('.')

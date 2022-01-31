#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Friday, July 5th 2019, 11:46:37 am
from ibllib.io.flags import FLAG_FILE_NAMES


def assign_task(task_deck, session_path, task, **kwargs):
    """

    Parameters
    ----------
    task_deck :
    session_path
    task
    kwargs

    Returns
    -------

    """
    t = task(session_path, **kwargs)
    task_deck[t.name] = t

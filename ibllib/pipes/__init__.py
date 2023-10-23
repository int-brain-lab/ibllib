#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Friday, July 5th 2019, 11:46:37 am
from ibllib.io.flags import FLAG_FILE_NAMES


def assign_task(task_deck, session_path, task, **kwargs):
    """
    Assigns a task to a task deck with the task name as key.

    This is a convenience function when creating a large task deck.

    Parameters
    ----------
    task_deck : dict
        A dictionary of tasks to add to.
    session_path : str, pathlib.Path
        A session path to pass to the task.
    task : ibllib.pipes.tasks.Task
        A task class to instantiate and assign.
    **kwargs
        Optional keyword arguments to pass to the task.

    Examples
    --------
    >>> from ibllib.pipes.video_tasks import VideoCompress
    >>> task_deck = {}
    >>> session_path = './subject/2023-01-01/001'
    >>> assign_task(task_deck, session_path, VideoCompress, cameras=('left',))
    {'VideoCompress': <ibllib.pipes.video_tasks.VideoCompress object at 0x0000020461E762D0>}

    Using partial for convenience

    >>> from functools import partial
    >>> assign = partial(assign_task, task_deck, session_path)
    >>> assign(VideoCompress, cameras=('left',))
    """
    t = task(session_path, **kwargs)
    task_deck[t.name] = t

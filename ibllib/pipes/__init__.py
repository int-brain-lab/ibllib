"""IBL preprocessing pipeline.

This module concerns the data extraction and preprocessing for IBL data.  The lab servers routinely
call `local_server.job_creator` to search for new sessions to extract. The job creator registers
the new session to Alyx (i.e. creates a new session record on the database), if required, then
deduces a set of tasks (a.k.a. the pipeline [*]_) from the 'experiment.description' file at the
root of the session (see `dynamic_pipeline.make_pipeline`). If no file exists one is created,
inferring the acquisition hardware from the task protocol. The new session's pipeline tasks are
then registered for another process (or server) to query.

Another process calls `local_server.task_queue` to get a list of queued tasks from Alyx, then
`local_server.tasks_runner` to loop through tasks.  Each task is run by called
`tasks.run_alyx_task` with a dictionary of task information, including the Task class and its
parameters.

.. [*] A pipeline is a collection of tasks that depend on one another.  A pipeline consists of
   tasks associated with the same session path.  Unlike pipelines, tasks are represented in Alyx.
   A pipeline can be recreated given a list of task dictionaries.  The order is defined by the
   'parents' field of each task.

Notes
-----
All new tasks are subclasses of the base_tasks.DynamicTask class.  All others are defunct and shall
be removed in the future.
"""


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

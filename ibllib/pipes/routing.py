"""Map pipeline tasks to the Python environments that run them.

Tasks are routed by their Alyx ``executable`` string alone so that a task queue can be filtered
without importing the task classes. This means a server only ever imports the classes of the
environment it is running in.

The top-level package of an executable usually determines the environment (e.g. all ``mpci``
tasks run in the ``mpci`` env). The exceptions are ibllib tasks that have their own environment,
e.g. DLC and spike sorting. Any executable without a matching route is run in the base
environment (``None``).

The :attr:`ibllib.pipes.tasks.Task.env` class attribute must match the route of the task's
executable; this is checked when tasks are created on Alyx.

Examples
--------
>>> task_env('mpci.suite2p.task.MesoscopePreprocess')
'mpci'
>>> task_env('ibllib.pipes.video_tasks.DLC')
'dlc'
>>> task_env('ibllib.pipes.video_tasks.VideoCompress') is None
True
"""

from pathlib import Path

ROUTES = {
    'ibllib.pipes.video_tasks.DLC': 'dlc',
    'ibllib.pipes.video_tasks.LightningPose': 'litpose',
    'ibllib.pipes.video_tasks.LightningAction': 'litaction',
    'ibllib.pipes.ephys_tasks.SpikeSorting': 'iblsorter',
    'mpci': 'mpci',
    'mpci.chronic.roicat': 'roicat',
}
"""dict of str: Map of dotted executable prefix to environment label; the longest match wins."""

_PYTHON = Path.home() / 'Documents' / 'PYTHON'
ENV_PATHS = {
    'dlc': _PYTHON / 'envs' / 'dlcenv',
    'litpose': _PYTHON / 'envs' / 'litpose',
    'litaction': _PYTHON / 'envs' / 'litaction',
    'iblsorter': _PYTHON / 'SPIKE_SORTING' / 'ibl-sorter' / '.venv',
    'mpci': _PYTHON / 'envs' / 'suite2p',
    'roicat': _PYTHON / 'envs' / 'roicat',
}
"""dict of str: Map of environment label to the location of its virtual environment on a server."""


def task_env(executable, routes=None):
    """
    Return the environment label of a task executable.

    Parameters
    ----------
    executable : str
        A task executable, e.g. 'mpci.suite2p.task.MesoscopePreprocess'.
    routes : dict of str, optional
        A map of dotted executable prefix to environment label. Defaults to ROUTES.

    Returns
    -------
    str, None
        The environment label, or None if the task is run in the base environment.
    """
    routes = ROUTES if routes is None else routes
    matches = [k for k in routes if executable == k or executable.startswith(k.rstrip('.') + '.')]
    return routes[max(matches, key=len)] if matches else None


def env_path(env, env_paths=None):
    """
    Return the virtual environment location of an environment label.

    Parameters
    ----------
    env : str
        An environment label, e.g. 'mpci'.
    env_paths : dict of str, optional
        A map of environment label to virtual environment location. Defaults to ENV_PATHS.

    Returns
    -------
    pathlib.Path, None
        The virtual environment location, or None if the label is unknown.
    """
    env_paths = ENV_PATHS if env_paths is None else env_paths
    return Path(env_paths[env]) if env in env_paths else None


def env_python(env, env_paths=None):
    """
    Return the Python executable of an environment if it is installed.

    Parameters
    ----------
    env : str
        An environment label, e.g. 'mpci'.
    env_paths : dict of str, optional
        A map of environment label to virtual environment location. Defaults to ENV_PATHS.

    Returns
    -------
    pathlib.Path, None
        The Python executable, or None if the environment is not installed.
    """
    if not (path := env_path(env, env_paths)):
        return None
    python = path / 'bin' / 'python'
    return python if python.exists() else None


def installed_envs(env_paths=None):
    """
    Return the environment labels installed on this machine.

    Parameters
    ----------
    env_paths : dict of str, optional
        A map of environment label to virtual environment location. Defaults to ENV_PATHS.

    Returns
    -------
    list of str
        The installed environment labels, including None (the base environment).
    """
    env_paths = ENV_PATHS if env_paths is None else env_paths
    return [None, *sorted(k for k in env_paths if env_python(k, env_paths))]

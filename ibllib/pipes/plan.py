"""Plan pipeline tasks from other repositories in the environment that runs them.

Some pipeline tasks are defined in repositories that are installed in their own environment (e.g.
the mesoscope tasks in mpci). Instead of importing these repositories when creating a pipeline,
their tasks are planned by running a *planner* function within the task environment, which returns
plain-data :class:`ibllib.pipes.spec.TaskSpec` objects.

A planner is a function with the signature ``planner(session_path, context=None)`` that returns a
:class:`ibllib.pipes.tasks.Pipeline`, a map of task name to task, or a list of tasks, where each
task is a Task instance, TaskSpec or task dictionary. The context is a dictionary with the key
'tasks', a list of the other (core) pipeline task specs as dicts, whose names may be used as
parents.

The PLANNERS map determines which planner is called for a given acquisition description device.

Examples
--------
Plan the mesoscope tasks of a session from the command line (within the mpci environment)

>>> python -m ibllib.pipes.plan mpci.alyx.pipeline:plan /path/to/subject/2020-01-01/001 --output specs.json

Plan the mesoscope tasks from the base environment

>>> specs = plan_in_env('mpci.alyx.pipeline:plan', 'mpci', session_path)
"""

import argparse
import importlib
import importlib.util
import json
import logging
import subprocess
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

from ibllib.pipes.routing import env_python
from ibllib.pipes.spec import TaskSpec

_logger = logging.getLogger(__name__)

PLANNERS = {
    'mesoscope': ('mpci', 'mpci.alyx.pipeline:plan'),
}
"""dict of tuple: Map of acquisition description device to (environment label, planner target)."""


class PlannerError(Exception):
    """Failed to plan the tasks of an external repository."""


def load_planner(target):
    """
    Import a planner function.

    Parameters
    ----------
    target : str
        The planner function as 'module:function', e.g. 'mpci.alyx.pipeline:plan'.

    Returns
    -------
    function
        The planner function.
    """
    module, _, function = target.partition(':')
    return getattr(importlib.import_module(module), function)


def to_specs(tasks):
    """
    Convert the output of a planner function to a list of task specs.

    Parameters
    ----------
    tasks : ibllib.pipes.tasks.Pipeline, dict, list
        A pipeline, a map of task name to task, or a list of tasks, where each task is a Task
        instance, TaskSpec or task dictionary.

    Returns
    -------
    list of TaskSpec
        The task specs.
    """
    tasks = getattr(tasks, 'tasks', tasks)  # Pipeline -> tasks map
    tasks = tasks.values() if isinstance(tasks, dict) else tasks
    specs = []
    for t in tasks:
        if isinstance(t, TaskSpec):
            specs.append(t)
        elif isinstance(t, dict):
            specs.append(TaskSpec.from_dict(t))
        else:
            specs.append(t.to_spec())
    return specs


def plan(target, session_path, context=None):
    """
    Plan tasks in the current environment.

    Parameters
    ----------
    target : str
        The planner function as 'module:function', e.g. 'mpci.alyx.pipeline:plan'.
    session_path : str, pathlib.Path
        The session path.
    context : dict, optional
        The planning context, see module docstring.

    Returns
    -------
    list of TaskSpec
        The task specs.
    """
    return to_specs(load_planner(target)(Path(session_path), context=context))


def plan_in_env(target, env, session_path, context=None, env_paths=None, timeout=900):
    """
    Plan tasks in another environment by calling this module in a subprocess.

    Parameters
    ----------
    target : str
        The planner function as 'module:function', e.g. 'mpci.alyx.pipeline:plan'.
    env : str
        The environment label, e.g. 'mpci'.
    session_path : str, pathlib.Path
        The session path.
    context : dict, optional
        The planning context, see module docstring.
    env_paths : dict, optional
        A map of environment label to virtual environment location. Defaults to
        ibllib.pipes.routing.ENV_PATHS.
    timeout : float
        The maximum time in seconds to wait for the subprocess.

    Returns
    -------
    list of TaskSpec
        The task specs.

    Raises
    ------
    PlannerError
        The environment is not installed or the planner failed.
    """
    if not (python := env_python(env, env_paths)):
        raise PlannerError(f'Environment "{env}" not installed')
    with tempfile.TemporaryDirectory() as tmp:
        context_file, output_file = Path(tmp, 'context.json'), Path(tmp, 'specs.json')
        context_file.write_text(json.dumps(context or {}))
        cmd = [str(python), '-m', 'ibllib.pipes.plan', target, str(session_path)]
        cmd += ['--context', str(context_file), '--output', str(output_file)]
        _logger.info('Planning %s tasks in "%s" env', target, env)
        try:
            process = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as ex:
            raise PlannerError(f'{target} timed out after {timeout}s in "{env}" env') from ex
        if process.returncode != 0 or not output_file.exists():
            raise PlannerError(f'{target} failed in "{env}" env:\n{process.stderr[-5000:]}')
        return [TaskSpec.from_dict(d) for d in json.loads(output_file.read_text())]


def get_external_tasks(acquisition_description, session_path, context=None, planners=None, env_paths=None):
    """
    Plan the tasks of external repositories for the devices in an acquisition description.

    For each device in PLANNERS, the planner is called in its environment if installed, otherwise
    in the current environment if the planner module is importable (e.g. on a development machine).

    Parameters
    ----------
    acquisition_description : dict
        The acquisition description.
    session_path : str, pathlib.Path
        The session path.
    context : dict, optional
        The planning context, see module docstring.
    planners : dict, optional
        A map of device to (environment label, planner target). Defaults to PLANNERS.
    env_paths : dict, optional
        A map of environment label to virtual environment location. Defaults to
        ibllib.pipes.routing.ENV_PATHS.

    Returns
    -------
    collections.OrderedDict
        A map of task name to TaskSpec.
    dict
        A map of device to error message for any planners that failed.
    """
    planners = PLANNERS if planners is None else planners
    devices = acquisition_description.get('devices', {})
    specs, errors = OrderedDict(), {}
    for device, (env, target) in planners.items():
        if device not in devices:
            continue
        try:
            if env_python(env, env_paths):
                device_specs = plan_in_env(target, env, session_path, context=context, env_paths=env_paths)
            elif importlib.util.find_spec(target.split('.', 1)[0]):
                _logger.debug('"%s" env not installed; planning %s tasks in current env', env, device)
                device_specs = plan(target, session_path, context=context)
            else:
                raise PlannerError(f'Environment "{env}" not installed')
        except Exception as ex:
            _logger.error('Failed to plan %s tasks: %s', device, ex)
            errors[device] = str(ex)
            continue
        specs.update((s.name, s) for s in device_specs)
    return specs, errors


def main(argv=None):
    parser = argparse.ArgumentParser(description='Plan pipeline tasks and save them as JSON task specs.')
    parser.add_argument('target', help="The planner function as 'module:function', e.g. 'mpci.alyx.pipeline:plan'")
    parser.add_argument('session_path', type=Path, help='The session path.')
    parser.add_argument('--context', type=Path, help='A JSON file containing the planning context.')
    parser.add_argument('--output', type=Path, help='The JSON file to save the task specs to (default: stdout).')
    args = parser.parse_args(argv)
    context = json.loads(args.context.read_text()) if args.context else None
    specs = json.dumps([s.to_dict() for s in plan(args.target, args.session_path, context=context)])
    if args.output:
        args.output.write_text(specs)
    else:
        sys.stdout.write(specs)


if __name__ == '__main__':
    main()

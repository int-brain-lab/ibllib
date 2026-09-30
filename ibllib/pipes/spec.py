"""Plain-data pipeline task specifications.

A :class:`TaskSpec` holds everything required to create a task on Alyx: its name, executable,
run arguments, parent names and resources. Unlike a :class:`ibllib.pipes.tasks.Task` instance,
a spec can be created and posted without importing the task class, so pipelines can combine tasks
from repositories that are installed in different environments.

This module deliberately depends only on the standard library.

Examples
--------
Create a spec from a task instance

>>> spec = task.to_spec()

Specs are JSON serializable

>>> specs = [TaskSpec.from_dict(d) for d in json.loads(json.dumps([s.to_dict() for s in specs]))]

Sort specs so that parents come before their children and compute the task levels

>>> specs = sort_specs(specs)
"""

import importlib
from dataclasses import dataclass, field, asdict, fields, replace


def executable_name(task):
    """
    Return the executable name of a task as it should be stored in Alyx.

    When the class is created dynamically using the type() built-in function, the base class is
    returned so that the class can be re-instantiated from the Alyx record.

    Parameters
    ----------
    task : ibllib.pipes.tasks.Task
        A task instance.

    Returns
    -------
    str
        The full module plus class name.
    """
    if task.__module__ == 'abc':
        return f'{task.__class__.__base__.__module__}.{task.__class__.__base__.__name__}'
    return f'{task.__module__}.{task.name}'


@dataclass
class TaskSpec:
    """A plain-data specification of a pipeline task."""

    name: str
    """str: The task name, unique within a session (e.g. 'Trials_ChoiceWorldTrialsNidq_00')."""
    executable: str
    """str: The full module plus class name of the task (e.g. 'ibllib.pipes.video_tasks.DLC')."""
    arguments: dict = field(default_factory=dict)
    """dict: The keyword arguments passed to the task class constructor. Must be JSON serializable."""
    parents: list = field(default_factory=list)
    """list of str: The names of the parent tasks."""
    level: int = 0
    """int: The level in the pipeline hierarchy. Computed from the parents by :func:`sort_specs`."""
    priority: int = 30
    io_charge: int = 5
    gpu: int = 0
    cpu: int = 1
    ram: int = 4
    time_out_secs: int = 3600 * 2
    env: str = None
    """str: The environment label of the task class. Not stored on Alyx, see :mod:`ibllib.pipes.routing`."""
    job_size: str = 'small'
    """str: The job size of the task class. Not stored on Alyx."""

    @classmethod
    def from_task(cls, task):
        """
        Create a spec from a task instance.

        Parameters
        ----------
        task : ibllib.pipes.tasks.Task
            A task instance.

        Returns
        -------
        TaskSpec
            The task specification.
        """
        return cls(
            name=task.name,
            executable=executable_name(task),
            arguments=task.kwargs,
            parents=[p.name for p in task.parents],
            level=task.level,
            priority=task.priority,
            io_charge=task.io_charge,
            gpu=task.gpu,
            cpu=task.cpu,
            ram=task.ram,
            time_out_secs=task.time_out_secs,
            env=task.env,
            job_size=task.job_size,
        )

    @classmethod
    def from_dict(cls, d):
        """
        Create a spec from a dictionary.

        Accepts the output of :meth:`to_dict`, :meth:`to_alyx` and
        :meth:`ibllib.pipes.tasks.Pipeline.create_tasks_list_from_pipeline`. Unknown keys are
        ignored.

        Parameters
        ----------
        d : dict
            A task dictionary. The parents must be task names, not Alyx IDs.

        Returns
        -------
        TaskSpec
            The task specification.
        """
        d = dict(d)
        if 'time_out_sec' in d:  # Alyx task dict key
            d.setdefault('time_out_secs', d.pop('time_out_sec'))
        names = {f.name for f in fields(cls)}
        spec = cls(**{k: v for k, v in d.items() if k in names})
        spec.arguments = spec.arguments or {}
        spec.parents = list(spec.parents or [])
        return spec

    def to_dict(self):
        """dict: A JSON serializable dictionary of the spec."""
        return asdict(self)

    def to_alyx(self, parents=None, **kwargs):
        """
        Return the dictionary used to create the task on Alyx.

        Parameters
        ----------
        parents : list of str, optional
            The parent task Alyx IDs. Defaults to the parent names.
        kwargs
            Other fields to set, e.g. session, graph, module, data_repository.

        Returns
        -------
        dict
            An Alyx task dictionary.
        """
        return {
            'executable': self.executable,
            'priority': self.priority,
            'io_charge': self.io_charge,
            'gpu': self.gpu,
            'cpu': self.cpu,
            'ram': self.ram,
            'parents': self.parents if parents is None else parents,
            'level': self.level,
            'time_out_sec': self.time_out_secs,
            'status': 'Waiting',
            'log': None,
            'name': self.name,
            'arguments': self.arguments,
            **kwargs,
        }

    def instantiate(self, session_path, **kwargs):
        """
        Instantiate the task class. This imports the task module.

        Parameters
        ----------
        session_path : str, pathlib.Path
            The session path.
        kwargs
            Extra keyword arguments passed to the task constructor, e.g. one, location.

        Returns
        -------
        ibllib.pipes.tasks.Task
            A task instance (without parents).
        """
        module, name = self.executable.rsplit('.', 1)
        task_class = getattr(importlib.import_module(module), name)
        return task_class(session_path, **self.arguments, **kwargs)


def sort_specs(specs):
    """
    Sort task specs so that parents come before their children, and compute each task's level.

    The sort is stable: specs that are already in a valid order are not moved.

    Parameters
    ----------
    specs : iterable of TaskSpec
        The task specs. Each parent name must be the name of another spec.

    Returns
    -------
    list of TaskSpec
        The sorted task specs (copies, the input specs are not modified).

    Raises
    ------
    ValueError
        Duplicate task names, missing parents or circular dependencies.
    """
    specs = list(specs)
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f'Duplicate task names: {sorted({n for n in names if names.count(n) > 1})}')
    if missing := {p for s in specs for p in s.parents} - set(names):
        raise ValueError(f'Parent tasks not found: {sorted(missing)}')
    levels, out, pending = {}, [], specs
    while pending:
        # Take the first spec whose parents are all placed
        i = next((i for i, s in enumerate(pending) if all(p in levels for p in s.parents)), None)
        if i is None:
            raise ValueError(f'Circular task dependencies: {sorted(s.name for s in pending)}')
        spec = pending.pop(i)
        spec = replace(spec, level=max((levels[p] for p in spec.parents), default=-1) + 1)
        levels[spec.name] = spec.level
        out.append(spec)
    return out

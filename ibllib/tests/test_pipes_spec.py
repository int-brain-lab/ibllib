"""Tests for the ibllib.pipes.routing, ibllib.pipes.spec and ibllib.pipes.plan modules."""

import importlib
import inspect
import json
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import ibllib.pipes
from ibllib.pipes import routing, plan
from ibllib.pipes.spec import TaskSpec, sort_specs, executable_name
from ibllib.pipes.tasks import Task, Pipeline


class Task00(Task):
    priority = 90
    job_size = 'large'

    def _run(self, **_):
        pass


class Task01(Task):
    env = 'foo'

    def _run(self, **_):
        pass


def planner(session_path, context=None):
    """A planner for testing ibllib.pipes.plan (must be importable in a subprocess)."""
    if context and context.get('raise'):
        raise RuntimeError('planner failed')
    parents = [context['tasks'][0]['name']] if context and context.get('tasks') else []
    t0 = type('PlannedTask', (Task00,), {})(session_path, foo='bar', parents=[])
    return Pipeline(session_path=session_path, tasks={
        'PlannedTask': t0,
        'PlannedSpec': TaskSpec('PlannedSpec', 'mpci.foo.Bar', parents=['PlannedTask', *parents], env='mpci'),
    })


class TestRouting(unittest.TestCase):
    """Tests for the ibllib.pipes.routing module."""

    def test_task_env(self):
        self.assertEqual('mpci', routing.task_env('mpci.suite2p.task.MesoscopePreprocess'))
        self.assertEqual('roicat', routing.task_env('mpci.chronic.roicat.task.ROICaTTask'))  # longest match
        self.assertEqual('dlc', routing.task_env('ibllib.pipes.video_tasks.DLC'))
        self.assertIsNone(routing.task_env('ibllib.pipes.video_tasks.DLCFoo'))  # not a dotted prefix
        self.assertIsNone(routing.task_env('mpcifoo.task.Task'))
        self.assertIsNone(routing.task_env('ibllib.pipes.video_tasks.VideoCompress'))
        self.assertEqual('bar', routing.task_env('foo.Task', routes={'foo.': 'bar'}))

    def test_installed_envs(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_paths = {'foo': Path(tmp, 'foo'), 'bar': Path(tmp, 'bar'), 'baz': Path(tmp, 'baz')}
            for env in ('foo', 'bar'):
                env_paths[env].joinpath('bin').mkdir(parents=True)
                env_paths[env].joinpath('bin', 'python').touch()
            self.assertEqual([None, 'bar', 'foo'], routing.installed_envs(env_paths))
            self.assertEqual(env_paths['foo'] / 'bin' / 'python', routing.env_python('foo', env_paths))
            self.assertIsNone(routing.env_python('baz', env_paths))
            self.assertIsNone(routing.env_python('unknown', env_paths))

    def test_ibllib_task_envs(self):
        """Test each ibllib Task subclass env matches routing.task_env for its executable.

        If this fails, update ibllib.pipes.routing.ROUTES, otherwise the task will not be run.
        """
        n = 0
        for file in sorted(Path(ibllib.pipes.__file__).parent.glob('*tasks*.py')):
            module = importlib.import_module(f'ibllib.pipes.{file.stem}')
            for name, cls in inspect.getmembers(module, inspect.isclass):
                if issubclass(cls, Task) and cls.__module__ == module.__name__:
                    executable = f'{cls.__module__}.{name}'
                    with self.subTest(executable=executable):
                        self.assertEqual(cls.env, routing.task_env(executable))
                    n += 1
        self.assertGreater(n, 50)


class TestTaskSpec(unittest.TestCase):
    """Tests for the ibllib.pipes.spec module."""

    def setUp(self):
        self.session_path = Path('/subject/2020-01-01/001')
        self.t0 = type('Task00_foo', (Task00,), {})(self.session_path, foo='bar')
        self.t1 = Task01(self.session_path, parents=[self.t0])

    def test_from_task(self):
        spec = self.t0.to_spec()
        self.assertEqual('Task00_foo', spec.name)
        self.assertEqual('ibllib.tests.test_pipes_spec.Task00', spec.executable)  # dynamic class base
        self.assertEqual(executable_name(self.t0), spec.executable)
        self.assertEqual({'foo': 'bar'}, spec.arguments)
        self.assertEqual(([], 0, 90, 'large', None), (spec.parents, spec.level, spec.priority, spec.job_size, spec.env))
        spec = self.t1.to_spec()
        self.assertEqual('ibllib.tests.test_pipes_spec.Task01', spec.executable)
        self.assertEqual((['Task00_foo'], 1, 'foo'), (spec.parents, spec.level, spec.env))

    def test_dicts(self):
        spec = self.t1.to_spec()
        # Round trip through JSON
        self.assertEqual(spec, TaskSpec.from_dict(json.loads(json.dumps(spec.to_dict()))))
        # Alyx task dict
        d = spec.to_alyx(parents=['uuid'], session='eid', graph='Pipeline')
        self.assertEqual((['uuid'], 'eid', 'Pipeline', 'Waiting'), (d['parents'], d['session'], d['graph'], d['status']))
        self.assertEqual(spec.time_out_secs, d['time_out_secs'])
        self.assertEqual(spec.parents, spec.to_alyx()['parents'])
        # Alyx task dict (with parent names) to spec; env and job_size are not stored on Alyx
        d = spec.to_alyx(session='eid')
        d['time_out_secs'] = 10
        self.assertEqual(10, TaskSpec.from_dict(d).time_out_secs)
        # Legacy task dicts (e.g. pipeline_tasks.yaml fixtures) use the key 'time_out_sec'
        d['time_out_sec'] = d.pop('time_out_secs')
        self.assertEqual(TaskSpec.from_dict(d), TaskSpec(**{**spec.to_dict(), 'env': None, 'time_out_secs': 10}))

    def test_sort_specs(self):
        a, b, c, d = (TaskSpec(x, 'foo.Task') for x in 'abcd')
        b.parents, c.parents, d.parents = ['c'], ['a'], ['a', 'b']
        specs = sort_specs([a, b, c, d])
        self.assertEqual(['a', 'c', 'b', 'd'], [s.name for s in specs])
        self.assertEqual([0, 1, 2, 3], [s.level for s in specs])
        self.assertEqual(0, b.level, 'input specs should not be modified')
        # Already sorted specs are unchanged
        self.assertEqual(specs, sort_specs(specs))
        with self.assertRaises(ValueError, msg='duplicate names'):
            sort_specs([a, a])
        with self.assertRaises(ValueError, msg='missing parent'):
            sort_specs([b])
        a.parents = ['d']
        with self.assertRaises(ValueError, msg='circular'):
            sort_specs([a, b, c, d])

    def test_instantiate(self):
        task = self.t1.to_spec().instantiate(self.session_path, location='remote')
        self.assertIsInstance(task, Task01)
        self.assertEqual(('remote', self.session_path), (task.location, task.session_path))

    def test_pipeline_task_specs(self):
        """Test Pipeline methods with a mix of Task instances and specs."""
        spec = TaskSpec('Task02', 'mpci.foo.Task', parents=['Task01'], env='mpci')
        pipe = Pipeline(session_path=self.session_path, tasks={'Task02': spec, 'Task00_foo': self.t0, 'Task01': self.t1})
        with self.assertLogs('ibllib.pipes.tasks', 'WARNING') as log:
            specs = pipe.task_specs()
        # The Task01 env doesn't match its route
        self.assertEqual(1, len(log.records))
        self.assertIn('Task01', log.records[0].getMessage())
        self.assertEqual(['Task00_foo', 'Task01', 'Task02'], [s.name for s in specs])
        self.assertEqual([0, 1, 2], [s.level for s in specs])
        with self.assertLogs('ibllib.pipes.tasks', 'WARNING'):
            task_list = pipe.create_tasks_list_from_pipeline()
        self.assertEqual(['Task00_foo', 'Task01', 'Task02'], [t['name'] for t in task_list])
        self.assertEqual(['Task01'], task_list[-1]['parents'])
        self.assertEqual('Pipeline', task_list[-1]['graph'])
        # NB: env and job_size are not stored in Alyx task dicts
        with self.assertLogs('ibllib.pipes.tasks', 'WARNING'):
            from_dicts = pipe.task_specs(task_list)
        stored = lambda s: {k: v for k, v in s.to_dict().items() if k not in ('env', 'job_size')}  # noqa: E731
        self.assertEqual(list(map(stored, specs)), list(map(stored, from_dicts)))
        with self.assertLogs('ibllib.pipes.tasks', 'WARNING'):
            graph = pipe.make_graph(out_dir=tempfile.gettempdir(), show=False)
        for edge in ('root -> Task00_foo', 'Task00_foo -> Task01', 'Task01 -> Task02'):
            self.assertIn(edge, graph.source)

    def test_create_alyx_tasks(self):
        """Test Pipeline.create_alyx_tasks creates parents first and patches existing tasks."""
        spec = TaskSpec('Task02', 'mpci.foo.Task', parents=['Task00_foo'], env='mpci')
        one = mock.MagicMock()
        one.alyx.cache_mode = None
        pipe = Pipeline(session_path=self.session_path, one=one, eid='eid', tasks=OrderedDict(Task02=spec, Task00_foo=self.t0))
        existing = {'id': 'id_Task00_foo', 'name': 'Task00_foo', 'status': 'Waiting'}
        one.alyx.rest.side_effect = lambda *args, **kwargs: {
            'list': [existing],
            'partial_update': existing,
            'create': {'id': 'id_' + kwargs.get('data', {}).get('name', ''), **kwargs.get('data', {})},
        }[args[1]]
        tasks = pipe.create_alyx_tasks(rerun__status__in=['Waiting'])
        self.assertEqual(['Task00_foo', 'Task02'], [t['name'] for t in tasks])
        self.assertEqual(['id_Task00_foo'], tasks[1]['parents'])
        self.assertEqual(1, tasks[1]['level'])
        calls = [c.args[1] for c in one.alyx.rest.call_args_list]
        self.assertEqual(['list', 'partial_update', 'create'], calls)
        self.assertEqual(Task00.time_out_secs, one.alyx.rest.call_args.kwargs['data']['time_out_secs'])
        # Time outs greater than the Alyx field maximum raise before any tasks are created
        one.alyx.rest.reset_mock()
        pipe.tasks['Task02'] = TaskSpec('Task02', 'mpci.foo.Task', parents=['Task00_foo'], env='mpci', time_out_secs=32768)
        with self.assertRaises(AssertionError):
            pipe.create_alyx_tasks()
        self.assertEqual(['list'], [c.args[1] for c in one.alyx.rest.call_args_list])


class TestPlan(unittest.TestCase):
    """Tests for the ibllib.pipes.plan module."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.session_path = Path(tmp.name, 'subject', '2020-01-01', '001')
        self.target = f'{__name__}:planner'
        # An env that points to the current Python environment, for testing subprocess calls
        self.env_paths = {'test': Path(sys.executable).parents[1], 'missing': Path(tmp.name, 'missing')}
        self.description = {'devices': {'foo': {}}}

    def test_to_specs(self):
        pipe = planner(self.session_path)
        specs = plan.to_specs(pipe)
        self.assertEqual(['PlannedTask', 'PlannedSpec'], [s.name for s in specs])
        self.assertEqual(specs, plan.to_specs(pipe.tasks))
        self.assertEqual(specs, plan.to_specs(list(pipe.tasks.values())))
        self.assertEqual(specs, plan.to_specs([s.to_dict() for s in specs]))

    def test_plan(self):
        context = {'tasks': [{'name': 'CoreTask'}]}
        specs = plan.plan(self.target, self.session_path, context=context)
        self.assertEqual(['PlannedTask', 'CoreTask'], specs[1].parents)
        self.assertEqual({'foo': 'bar'}, specs[0].arguments)

    @unittest.skipIf(not Path(sys.executable).parents[1].joinpath('bin', 'python').exists(), 'not a venv layout')
    def test_plan_in_env(self):
        context = {'tasks': [{'name': 'CoreTask'}]}
        specs = plan.plan_in_env(self.target, 'test', self.session_path, context=context, env_paths=self.env_paths)
        self.assertEqual(plan.plan(self.target, self.session_path, context=context), specs)
        with self.assertRaises(plan.PlannerError) as ex:
            plan.plan_in_env(self.target, 'test', self.session_path, context={'raise': True}, env_paths=self.env_paths)
        self.assertIn('planner failed', str(ex.exception))
        with self.assertRaises(plan.PlannerError):
            plan.plan_in_env(self.target, 'missing', self.session_path, env_paths=self.env_paths)

    def test_get_external_tasks(self):
        planners = {'foo': ('missing', self.target), 'bar': ('missing', 'notapkg.foo:plan')}
        kwargs = {'planners': planners, 'env_paths': self.env_paths}
        # Env not installed so the planner is called in the current env (the planner is importable)
        with mock.patch('ibllib.pipes.plan.plan_in_env') as plan_in_env:
            specs, errors = plan.get_external_tasks(self.description, self.session_path, **kwargs)
            plan_in_env.assert_not_called()
        self.assertEqual({}, errors)
        self.assertEqual(['PlannedTask', 'PlannedSpec'], list(specs))
        # Env installed so the planner is called in a subprocess
        planners['foo'] = ('test', self.target)
        with mock.patch('ibllib.pipes.plan.plan_in_env', return_value=[TaskSpec('Foo', 'foo.Task')]) as plan_in_env:
            specs, errors = plan.get_external_tasks(self.description, self.session_path, **kwargs)
            plan_in_env.assert_called_once()
        self.assertEqual(['Foo'], list(specs))
        # Neither the env nor the planner are installed
        self.description['devices']['bar'] = {}
        with self.assertLogs('ibllib.pipes.plan', 'ERROR'), mock.patch('ibllib.pipes.plan.plan_in_env', return_value=[]):
            specs, errors = plan.get_external_tasks(self.description, self.session_path, **kwargs)
        self.assertEqual({'bar': 'Environment "missing" not installed'}, errors)
        # Planner errors are caught
        with self.assertLogs('ibllib.pipes.plan', 'ERROR'):
            specs, errors = plan.get_external_tasks(
                self.description, self.session_path, context={'raise': True}, planners={'foo': ('missing', self.target)}
            )
        self.assertEqual({'foo': 'planner failed'}, errors)
        # Devices not in the description are ignored
        specs, errors = plan.get_external_tasks({}, self.session_path, planners=planners)
        self.assertEqual(({}, {}), (dict(specs), errors))

    def test_main(self):
        with tempfile.TemporaryDirectory() as tmp:
            output, context = Path(tmp, 'specs.json'), Path(tmp, 'context.json')
            context.write_text(json.dumps({'tasks': [{'name': 'CoreTask'}]}))
            plan.main([self.target, str(self.session_path), '--context', str(context), '--output', str(output)])
            specs = [TaskSpec.from_dict(d) for d in json.loads(output.read_text())]
        self.assertEqual(['PlannedTask', 'CoreTask'], specs[1].parents)


if __name__ == '__main__':
    unittest.main()

import unittest
from unittest.runner import TextTestResult, TextTestRunner
import time
import os
from pathlib import Path
from functools import wraps
import logging
import json
import tempfile
import shutil
import warnings

from iblutil.io import params
from iblutil.util import ensure_list
from one.alf.path import get_session_path
from one.api import ONE

INTEGRATION_DATA_DIR = os.environ.get('INTEGRATION_DATA_DIR')
INTEGRATION_DATA_WRITABLE = os.environ.get('INTEGRATION_DATA_WRITABLE', '1') != '0'
"""When False (INTEGRATION_DATA_WRITABLE=0), IntegrationTest.setUpClass creates a writable
symlink mirror of required_files under a TemporaryDirectory so tests can write output files
without touching the read-only data mount (e.g. S3)."""
_logger = logging.getLogger('ibllib')


class TimeLoggingTestResult(TextTestResult):
    """A class to record test durations"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_timings = []
        self._test_started_at = time.time()

    def startTest(self, test):
        self._test_started_at = time.time()
        super().startTest(test)

    def addSuccess(self, test):
        elapsed = time.time() - self._test_started_at
        name = str(test)  # self.getDescription(test) # includes first line of docstring
        self.test_timings.append((name, elapsed))
        super().addSuccess(test)

    def getTestDurations(self) -> 'list[tuple[str, int]]':
        """Returns list of tests and their durations, in reverse duration order"""
        return sorted(self.test_timings, key=lambda x: x[1], reverse=True)


class TimeLoggingTestRunner(TextTestRunner):
    """A class that prints a list of the slowest tests to the output stream"""
    def __init__(self, slow_test_threshold=0.3, *args, **kwargs):
        self.slow_test_threshold = slow_test_threshold
        super().__init__(resultclass=TimeLoggingTestResult, *args, **kwargs)

    def run(self, test):
        result = super().run(test)
        self.stream.writeln(f'\nSlow Tests (>{self.slow_test_threshold:.03}s):\n')
        for name, elapsed in result.getTestDurations():
            if elapsed > self.slow_test_threshold:
                self.stream.writeln(f'({elapsed:.03}s) {name}')
        return result


@unittest.skipUnless(
    INTEGRATION_DATA_DIR and os.path.isdir(INTEGRATION_DATA_DIR),
    "Integration data not available (set INTEGRATION_DATA_DIR to enable).",
)
class IntegrationTest(unittest.TestCase):
    """Base class for tests that require S3 integration data.

    Subclass this for any test needing integration data. When
    INTEGRATION_DATA_DIR is unset or missing (e.g. an outside contributor
    running plain `python -m unittest discover`), these tests auto-skip,
    so the unit suite still runs cleanly.
    """

    required_files = []
    """An optional list of required files/folders to glob for, relative to `data_path`."""

    _writable_tempdir = None
    """Class-level TemporaryDirectory created by setUpClass when INTEGRATION_DATA_WRITABLE=0."""

    _writable_scope = 'class'
    """Controls when the writable symlink mirror is created (only used when INTEGRATION_DATA_WRITABLE=0).
    'class' (default): one mirror shared across all test methods, created once in setUpClass/tearDownClass.
    'test': a fresh mirror for each test method, created in setUp and torn down in tearDown.
    Use 'test' when test methods modify the session directory in ways that conflict across tests
    (e.g. multiple test methods that call backup_alf on the same session).
    Note: test classes using _writable_scope = 'test' must call super().setUp() / super().tearDown()."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data_path = cls.default_data_root()
        if not INTEGRATION_DATA_WRITABLE and cls.required_files and cls._writable_scope == 'class':
            _, cls._writable_tempdir = make_sym_links(map(cls.data_path.joinpath, cls.required_files))
            cls.data_path = Path(cls._writable_tempdir.name)
            
    @classmethod
    def tearDownClass(cls):
        if cls._writable_tempdir is not None:
            cls._writable_tempdir.cleanup()
            cls._writable_tempdir = None
        super().tearDownClass()

    def setUp(self):
        super().setUp()
        self.data_path = self.default_data_root()  # sync instance attr with class-level setting from setUpClass
        if not INTEGRATION_DATA_WRITABLE and self.required_files and self._writable_scope == 'test':
            _, self._writable_tempdir = make_sym_links(map(self.data_path.joinpath, self.required_files))
            self.data_path = Path(self._writable_tempdir.name)

    def tearDown(self):
        if getattr(self, '_writable_tempdir', None) is not None and self._writable_scope == 'test':
            self._writable_tempdir.cleanup()
            self._writable_tempdir = None
        super().tearDown()

    def __init__(self, *args, **kwargs):
        """A base class for locating integration test data
        Upon initialization, loads the path to the integration test data.  The path is loaded from
        the '.ibl_ci' parameter file's 'data_root' parameter, or the current working directory.
        The data root may be overridden with the `data_path` keyword arg.  The data path must be an
        existing directory containing a 'Subjects_init' folder.
        :param data_path: The data root path to the integration data directory
        """
        super().__init__(*args, **kwargs)

        self.data_path = self.default_data_root()
        if type(self)._writable_tempdir is None and self._writable_scope != 'test':
            data_present = (self.data_path.exists() and
                            self.data_path.is_dir() and
                            any(self.data_path.glob('Subjects_init')))
            if self.required_files:
                data_present &= all(map(self.data_path.glob, self.required_files))
            if not data_present:
                raise FileNotFoundError(f'Invalid data root folder {self.data_path.absolute()}\n\t'
                                        'must contain a "Subjects_init" folder.')

    @classmethod
    def default_data_root(cls):
        """Returns the path to the integration data.

        When INTEGRATION_DATA_WRITABLE=0 and setUpClass has created a writable mirror, returns
        the writable temp dir so all path construction automatically uses the writable location.
        Otherwise returns the configured data root (INTEGRATION_DATA_DIR or ibl_ci params).
        """
        if getattr(cls, '_writable_tempdir', None) is not None:
            return Path(cls._writable_tempdir.name)
        if INTEGRATION_DATA_DIR:
            return Path(INTEGRATION_DATA_DIR)
        return Path(params.read('ibl_ci', {'data_root': '.'}).data_root)

    def backup_alf(self, session_path):
        """Backup alf folder.

        Some extraction tests backup the ALF folder, extract the data into a new alf folder then
        compare the results. This function moves the original ALF folder to a backup location.
        """
        alf_path = session_path.joinpath('alf')
        bk_path = alf_path.parent / 'alf.bk'
        if alf_path.exists():
            # Back-up alf files and restore on teardown
            if bk_path.exists():  # if last cleanup failed
                warnings.warn(f'{bk_path} already exists; removing alf path')
                # assume backup is correct validation data and delete the alf folder
                shutil.rmtree(alf_path, ignore_errors=True)
            else:
                shutil.move(alf_path, bk_path)
            self.addCleanup(self.restore_alf, session_path)
        elif not bk_path.exists():
            raise FileNotFoundError(f'alf folder missing for session {session_path}')

    @staticmethod
    def restore_alf(session_path):
        """Restore backup alf folder.

        Some extraction tests backup the ALF folder, extract the data into a new alf folder then
        compare the results. This function moves the backed up folder back.
        """
        alf_path = session_path.joinpath('alf')
        bk_path = alf_path.parent / 'alf.bk'
        if alf_path.exists() and bk_path.exists():
            shutil.rmtree(alf_path, ignore_errors=True)
            shutil.move(str(bk_path), str(alf_path))


def list_current_sessions(one=None):
    """
    Get the set of session eids used in integration tests.  When writing new tests, this can be
    a useful way of choosing which sessions to use.

    :param one: An ONE object for fetching session eid from path
    :return: Set of integration session eids
    """
    def not_null(itr):
        return filter(lambda x: x is not None, itr)
    one = one or ONE()
    root = IntegrationTest.default_data_root()
    folders = set(get_session_path(x[0]) for x in os.walk(root))
    eids = not_null(one.path2eid(x) for x in not_null(folders))
    return set(eids)


def disable_log(level=logging.CRITICAL, restore_level=None, quiet=False):
    """
    Decorator to temporarily disable the log.
    :param level: The minimum logging level to disable
    :param restore_level: The logging level to restore
    :param quiet: If false the fact that the log is disabled will be printed
    :return:
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            logging.disable(level)
            if not quiet:
                print('**Log disabled for test**')
            output = func(self, *args, **kwargs)
            if not quiet:
                print('**Log re-enabled**')
            logging.disable(restore_level or logging.NOTSET)
            return output
        return wrapper
    return decorator


def _get_test_db():
    db_json = os.getenv('TEST_DB_CONFIG', None)
    if db_json:
        with open(db_json, 'r') as f:
            return json.load(f)
    else:
        return {
            'base_url': 'https://test.alyx.internationalbrainlab.org',
            'username': 'test_user',
            'password': 'TapetesBloc18',
            'silent': True
        }


def make_sym_links(source_path, extraction_path=None, fallback_to_copy=True):
    """
    This creates symlinks to a scratch directory to start an extraction while leaving the
    raw data untouched.
    :param source_path: location containing the extraction fixture
    :param extraction_path: (None) scratch location where the symlinks will end up,
    omitting the session parts example: "/tmp". If set to None, it will create a temporary
    directory using tempdir.
    :param fallback_to_copy: (True) if the symlink fails, it will copy the file instead.
    :return:
    """
    if return_tempdir := extraction_path is None:
        temp_dir = tempfile.TemporaryDirectory()
        extraction_path = Path(temp_dir.name)

    # Create the session path in the extraction path, preserving the relative path from the data root
    root = IntegrationTest.default_data_root()

    new_files = []
    return_list = not isinstance(source_path, (str, Path))
    for src in ensure_list(source_path):
        for f in [src] if src.is_file() else src.rglob('*.*'):
            new_file = extraction_path.joinpath(f.relative_to(root))
            new_files.append(new_file)
            if new_file.exists():
                continue
            new_file.parent.mkdir(exist_ok=True, parents=True)
            try:
                new_file.symlink_to(f)
            except OSError as e:
                if fallback_to_copy:
                    _logger.error(f'Error creating symlink: {e}')
                    shutil.copy(f, new_file)
                else:
                    raise e

    if not return_list:
        new_files = new_files[0] if new_files else None

    if return_tempdir:
        return new_files, temp_dir
    return new_files


TEST_DB = _get_test_db()

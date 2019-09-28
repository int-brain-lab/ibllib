# -*- coding: utf-8 -*-

"""ONE light."""


# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import csv
from collections import defaultdict
import json
import logging
import os.path as op
from pathlib import Path
import re
import urllib.parse

import click
import requests
# from tqdm import tqdm

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------------------------------

# Set a null handler on the root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# logger.addHandler(logging.NullHandler())


_logger_fmt = '%(asctime)s.%(msecs)03d [%(levelname)s] %(caller)s %(message)s'
_logger_date_fmt = '%H:%M:%S'


class _Formatter(logging.Formatter):
    def format(self, record):
        # Only keep the first character in the level name.
        record.levelname = record.levelname[0]
        filename = op.splitext(op.basename(record.pathname))[0]
        record.caller = '{:s}:{:d}'.format(filename, record.lineno).ljust(20)
        message = super(_Formatter, self).format(record)
        color_code = {'D': '90', 'I': '0', 'W': '33', 'E': '31'}.get(record.levelname, '7')
        message = '\33[%sm%s\33[0m' % (color_code, message)
        return message


def add_default_handler(level='INFO', logger=logger):
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = _Formatter(fmt=_logger_fmt, datefmt=_logger_date_fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


add_default_handler('DEBUG')


# -------------------------------------------------------------------------------------------------
# Utils
# -------------------------------------------------------------------------------------------------

class Bunch(dict):
    """A subclass of dictionary with an additional dot syntax."""

    def __init__(self, *args, **kwargs):
        super(Bunch, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def copy(self):
        """Return a new Bunch instance which is a copy of the current Bunch instance."""
        return Bunch(super(Bunch, self).copy())


def is_documented_by(original):
    def wrapper(target):
        target.__doc__ = original.__doc__
        return target
    return wrapper


# -------------------------------------------------------------------------------------------------
# File scanning and root file creation
# -------------------------------------------------------------------------------------------------

def read_root_file(path):
    with open(path) as f:
        for line in csv.reader(f, delimiter='\t'):
            yield line[0], line[1]


def write_root_file(path, iterator):
    with open(path, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        for items in iterator:
            writer.writerow(items)


def walk(root):
    """Iterate over all files found within a root directory."""
    for p in Path(root).rglob('*'):
        yield p


def is_session_dir(path):
    """Return whether a path is a session directory.

    Example of a session dir: `/path/to/root/mainenlab/Subjects/ZM_1150/2019-05-07/001/`

    """
    return path.is_dir() and path.parent.parent.parent.name == 'Subjects'


def is_file_in_session_dir(path):
    """Return whether a file path is within a session directory."""
    return not path.is_dir() and '/Subjects/' in str(path.parent.parent.parent)


def find_session_dirs(root):
    """Iterate over all session directories found in a root directory."""
    for p in walk(root):
        if is_session_dir(p):
            yield p


def search_session_files(root):
    """Iterate over all files within session directories found within a root directory."""
    for p in walk(root):
        if is_file_in_session_dir(p):
            yield p


def make_http_root_file(root, base_url, output):
    """Make a root TSV file for an HTTP server.

    Note: the session root directory needs to be the directory that contains
    the <lab> subdirectories, so that the relative file paths are correctly obtained.

    """
    relative_paths = (str(p.relative_to(root)) for p in search_session_files(root))
    write_root_file(output, ((rp, urllib.parse.urljoin(base_url, rp)) for rp in relative_paths))


def download_file(url, save_to, auth=None):
    """Download a file from HTTP and save it to a file.
    If Basic HTTP authentication is needed, pass `auth=(username, password)`.
    """
    save_to = Path(save_to)
    logger.info("Downloading %s to %s.", url, str(save_to.parent))
    response = requests.get(url, stream=True, auth=auth or None)
    response.raise_for_status()
    save_to.parent.mkdir(parents=True, exist_ok=True)
    with open(save_to, "wb") as f:
        for data in response.iter_content():
            f.write(data)


def default_download_dir():
    """Default download directory on the client computer, with {...} placeholders fields."""
    return '~/.one/data/{lab}/Subjects/{subject}/{date}/{number}/alf/'


def format_download_dir(session, download_dir):
    """Replace the placeholder fields in the download directory by the appropriate values for
    a given session."""
    session_info = _parse_session_path(session)
    download_dir = download_dir.format(**session_info)
    return Path(download_dir).expanduser()


def load_array(path):
    """Load a single file."""
    if str(path).endswith('.npy'):
        try:
            import numpy as np
            return np.load(path, mmap_mode='r')
        except ImportError:
            logger.warning("NumPy is not available.")
            return
        except ValueError as e:
            logger.error("Impossible to read %s.", path)
            raise e
    elif str(path).endswith('.tsv'):
        try:
            import pandas as pd
            return pd.read_csv(str(path), sep='\t')
        except ImportError:
            logger.warning("Pandas is not available.")
        except ValueError as e:
            logger.error("Impossible to read %s.", path)
            raise e
    raise NotImplementedError(path)


# -------------------------------------------------------------------------------------------------
# File path parsing
# -------------------------------------------------------------------------------------------------

def _pattern_to_regex(pattern):
    """Convert a path pattern with {...} into a regex."""
    return _compile(re.sub(r'\{(\w+)\}', r'(?P<\1>[a-zA-Z0-9\_\-\.]+)', pattern))


def _escape_for_regex(s):
    for char in '-_.':
        s = s.replace('%s' % char, '\\%s' % char)
    s = s.replace('*', r'[a-zA-Z0-9\_\-\.]+')
    return s


def _compile(r):
    r = r.replace('/', r'\/')
    return re.compile(r)


SESSION_PATTERN = r'^{lab}/Subjects/{subject}/{date}/{number}/$'
SESSION_REGEX = _pattern_to_regex(SESSION_PATTERN)

FILE_PATTERN = r'^{lab}/Subjects/{subject}/{date}/{number}/alf/{filename}$'
FILE_REGEX = _pattern_to_regex(FILE_PATTERN)


def _make_search_regex(**kwargs):
    """Make a regex for the search() function."""
    pattern = FILE_PATTERN

    for term in ('lab', 'subject', 'date', 'number', 'filename'):
        value = kwargs.get(term, None)
        if value:
            if not isinstance(value, str):
                value = '|'.join(value)
            value = _escape_for_regex(value)
            pattern = pattern.replace(r'{%s}' % term, '(?P<%s>%s)' % (term, value))
        else:
            pattern = pattern.replace(r'{%s}' % term, r'(?P<%s>[a-zA-Z0-9_\-\.]+)' % term)
    return _compile(pattern)


def _make_dataset_regex(session, dataset_type):
    """Make a regex for finding datasets of a given type in a session."""
    pattern = r'^{session}.+{dataset_type}$'

    # Dataset type.
    if '*' not in dataset_type:
        dataset_type += '*'
    if '*' in dataset_type:
        dataset_type = dataset_type

    # Escape the session.
    dataset_type = _escape_for_regex(dataset_type)
    session = _escape_for_regex(session)

    # Make the regex pattern.
    pattern = pattern.format(session=session, dataset_type=dataset_type)
    return _compile(pattern)


def _parse_session_path(session):
    """Parse a session path."""
    m = SESSION_REGEX.match(session)
    if not m:
        raise ValueError("The session path `%s` is invalid." % session)
    return {n: m.group(n) for n in ('lab', 'subject', 'date', 'number')}


def _parse_file_path(file_path):
    """Parse a file path."""
    m = FILE_REGEX.match(file_path)
    if not m:
        raise ValueError("The file path `%s` is invalid.", file_path)
    return {n: m.group(n) for n in ('lab', 'subject', 'date', 'number', 'filename')}


def _search(root_file_iterator, regex):
    """Search an iterator among tuples (file_path, url) using a regex."""
    for rel_path, url in root_file_iterator:
        m = regex.match(rel_path)
        if m:
            yield rel_path, url, m


# -------------------------------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------------------------------

def config_dir():
    """Path to the config directory."""
    return Path.home() / '.one/'


def config_file():
    """Path to the config file."""
    return config_dir() / 'config.json'


def get_config():
    """Return the config file dictionary."""
    # Create a default config file if there is none.
    path = config_file()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        write_config(**default_config())
    # Open the config file.
    with open(path, 'r') as f:
        return json.load(f)


def write_config(**kwargs):
    """Write some key-value pairs in the config file."""
    if config_file().exists():
        config = get_config()
    else:
        config = {}
    config.update(kwargs)
    with open(config_file(), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)


# -------------------------------------------------------------------------------------------------
# HTTP ONE class
# -------------------------------------------------------------------------------------------------

class HttpOne:
    def __init__(self, root_file=None, base_url=None, download_dir=None, auth=None):
        self.root_file = root_file
        self.base_url = base_url
        self.download_dir = download_dir or default_download_dir()
        self.auth = auth or None

    def _download_dataset(self, session, filename, url):
        save_to_dir = Path(format_download_dir(session, self.download_dir))
        save_to = save_to_dir / filename
        if not save_to.exists():
            download_file(url, save_to, auth=self.auth)
        else:
            logger.debug("Skip existing %s.", save_to)
        assert save_to.exists()
        return save_to

    def search(self, dataset_types, **kwargs):
        """Search all sessions that have all requested dataset types."""
        dataset_types = [(dst + '*' if '*' not in dst else dst) for dst in dataset_types]
        filter_kwargs = {
            'lab': kwargs.get('lab', None),
            'subject': kwargs.get('subject', None),
            'date': kwargs.get('date', None),
            'number': kwargs.get('number', None),
            'filename': dataset_types,
        }
        pattern = _make_search_regex(**filter_kwargs)

        # Find all sessions that have *all* requested dataset types.
        sessions = defaultdict(set)
        n_types = len(dataset_types)
        dtypes_substr = [dst.replace('*', '') for dst in dataset_types]
        for rel_path, url, m in _search(read_root_file(self.root_file), pattern):
            session = '/'.join(rel_path.split('/')[:5])
            # For each session candidate, we check which dataset types it has.
            for dt in dtypes_substr:
                if dt in rel_path:
                    sessions[session].add(dt)
        # The number of different dataset types of each session must be equal to the number
        # of the requested dataset types.
        return sorted(
            session for (session, dset_set) in sessions.items() if len(dset_set) >= n_types)

    def load_dataset(self, session, dataset_type, download_only=False):
        """Download and load a single dataset in a given session and with a given dataset type."""
        # Ensure session has a trailing slash.
        if not session.endswith('/'):
            session += '/'
        pattern = _make_dataset_regex(session, dataset_type)
        tup = next(_search(read_root_file(self.root_file), pattern))
        if not tup:
            raise ValueError("No `%s` file found in this session.", dataset_type)
        filename = tup[0].split('/')[-1]
        url = tup[1]
        out = self._download_dataset(session, filename, url)
        if not download_only:
            out = load_array(out)
        return out

    def load_object(self, session, obj, download_only=False):
        """Load all attributes of a given object."""
        # Ensure session has a trailing slash.
        if not session.endswith('/'):
            session += '/'
        # TODO: for now, load all existing dataset types. A set of default dataset types
        # could be specified for each object.
        pattern = _make_dataset_regex(session, obj)
        out = Bunch()
        for rel_path, url, m in _search(read_root_file(self.root_file), pattern):
            filename = rel_path.split('/')[-1]
            fs = filename.split('.')
            attr = fs[1]
            out[attr] = self._download_dataset(session, filename, url)
            if not download_only:
                out[attr] = load_array(out[attr])
        return out


_ONE_SINGLETON = None


def _make_http_one():
    """Create a new HttpOne instance based on the config file."""
    # Full config dict.
    config = get_config()
    # Get the config key-value pairs where the key starts with http_config_.
    # Config keys are [http_config_<x>] where <x> is: root_file, base_url, download_dir, auth
    kwargs = {
        k.replace('http_config_', ''): v
        for k, v in config.items() if k.startswith('http_config_')}
    auth = kwargs.get('auth') or None
    kwargs['auth'] = tuple(auth) if auth else None
    # This is then passed to the HttpOne() constructor.
    return HttpOne(**kwargs)


def get_one():
    """Get the singleton One instance, loading it from the config file, or using the singleton
    instance if it has already been instantiated."""
    if globals()['_ONE_SINGLETON'] is None:
        globals()['_ONE_SINGLETON'] = _make_http_one()
    one = globals()['_ONE_SINGLETON']
    assert one
    return one


def default_config():
    return {
        'http_config_root_file': '',
        'http_config_base_url': '',
        'http_config_download_dir': default_download_dir(),
        'http_config_auth': None,
    }


# -------------------------------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------------------------------

def search_terms():
    return ('lab', 'subject', 'date', 'number', 'dataset_types')


def set_download_dir(path):
    """Set the download directory. May contain fields like {lab}, {subject}, etc."""
    # Update the config file.
    write_config(http_config_download_dir=path)
    # Reload the HttpOne instance.
    _make_http_one()


@is_documented_by(HttpOne.search)
def search(dataset_types, **kwargs):
    return get_one().search(dataset_types, **kwargs)


@is_documented_by(HttpOne.load_object)
def load_object(session, obj, **kwargs):
    return get_one().load_object(session, obj, **kwargs)


@is_documented_by(HttpOne.load_dataset)
def load_dataset(session, dataset_type, **kwargs):
    return get_one().load_dataset(session, dataset_type, **kwargs)


# -------------------------------------------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------------------------------------------

@click.group()
def one():
    pass


@one.command('search')
@click.argument('dataset_types', nargs=-1)
def search_(dataset_types):
    # NOTE: underscore to avoid shadowing of public search() function.
    # TODO: other search options
    for session in search(dataset_types):
        click.echo(session)


@one.command()
@click.argument('session')
@click.argument('obj', required=False)
def download(session, obj=None):
    for file_path in load_object(session, obj or '*', download_only=True).values():
        click.echo(file_path)


@one.command()
@click.argument('root_dir')
def scan(root_dir):
    pass


@one.command()
def upload():
    pass


if __name__ == '__main__':
    one()

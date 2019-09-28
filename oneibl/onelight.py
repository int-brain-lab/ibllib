# -*- coding: utf-8 -*-

"""ONE light."""


# -------------------------------------------------------------------------------------------------
# Imports
# -------------------------------------------------------------------------------------------------

import csv
from collections import defaultdict
import hashlib
import json
import logging
from operator import itemgetter
import os.path as op
from pathlib import Path
import re
import urllib.parse

import click
import requests
from requests.exceptions import HTTPError
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
    for p in sorted(Path(root).rglob('*')):
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


def find_session_files(root):
    """Iterate over all files within session directories found within a root directory."""
    for p in walk(root):
        if is_file_in_session_dir(p):
            yield p


def make_http_root_file(root, base_url, output):
    """Make a root TSV file for an HTTP server.

    Note: the session root directory needs to be the directory that contains
    the <lab> subdirectories, so that the relative file paths are correctly obtained.

    """
    relative_paths = (str(p.relative_to(root)) for p in find_session_files(root))
    write_root_file(output, ((rp, urllib.parse.urljoin(base_url, rp)) for rp in relative_paths))


def download_file(url, save_to, auth=None):
    """Download a file from HTTP and save it to a file.
    If Basic HTTP authentication is needed, pass `auth=(username, password)`.
    """
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s.", url, str(save_to.parent))
    if 'figshare.com/' in url:
        data = figshare_request(url=url, binary=False)
        with open(save_to, "wb") as f:
            f.write(data)
    else:
        response = requests.get(url, stream=True, auth=auth or None)
        response.raise_for_status()
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
        raise ValueError("The file path `%s` is invalid." % file_path)
    return {n: m.group(n) for n in ('lab', 'subject', 'date', 'number', 'filename')}


def _get_file_rel_path(file_path):
    """Get the lab/Subjects/subject/... part of a file path."""
    file_path = str(file_path)
    # Find the relative part of the file path.
    i = file_path.index('/Subjects')
    i = file_path[:i].rindex('/') + 1
    return file_path[i:]


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
    def __init__(self, root_file=None, download_dir=None, auth=None):
        self.root_file = root_file
        self.download_dir = download_dir or default_download_dir()
        self.auth = auth or None

    def _download_dataset(self, session, filename, url, dry_run=False):
        save_to_dir = Path(format_download_dir(session, self.download_dir))
        save_to = save_to_dir / filename
        if not save_to.exists() and not dry_run:
            download_file(url, save_to, auth=self.auth)
            assert save_to.exists()
        else:
            logger.debug("Skip %s.", save_to)
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

    def load_object(self, session, obj, download_only=False, dry_run=False):
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
            out[attr] = self._download_dataset(session, filename, url, dry_run=dry_run)
            if not download_only:
                out[attr] = load_array(out[attr])
        return out


# -------------------------------------------------------------------------------------------------
# figshare
# -------------------------------------------------------------------------------------------------

_FIGSHARE_BASE_URL = 'https://api.figshare.com/v2/{endpoint}'


def figshare_request(endpoint=None, data=None, method='GET', url=None, binary=False):
    headers = {'Authorization': 'token ' + get_config().get('figshare_token', None)}
    if data is not None and not binary:
        data = json.dumps(data)
    response = requests.request(
        method, url or _FIGSHARE_BASE_URL.format(endpoint=endpoint), headers=headers, data=data)
    try:
        response.raise_for_status()
        try:
            data = json.loads(response.content)
        except ValueError:
            data = response.content
    except HTTPError as error:
        raise error
    return data


def figshare_files(article_id):
    """Iterate over all ALF files of a given figshare article."""
    files = figshare_request('account/articles/%s/files' % str(article_id))
    for file in files:
        path = file['name'].replace('~', '/')
        if FILE_REGEX.match(path):
            yield path, file['download_url']


def make_figshare_root_file(article_id, output):
    """Create a root file for a figshare article."""
    write_root_file(output, figshare_files(article_id))


def _get_file_check_data(file_name):
    chunk_size = 1048576
    with open(file_name, 'rb') as fin:
        md5 = hashlib.md5()
        size = 0
        data = fin.read(chunk_size)
        while data:
            size += len(data)
            md5.update(data)
            data = fin.read(chunk_size)
        return md5.hexdigest(), size


def figshare_upload_file(path, name, article_id, dry_run=False):
    """Upload a single file to figshare."""
    # see https://docs.figshare.com/#upload_files_example_upload_on_figshare
    assert Path(path).exists()
    logger.info("Uploading %s.%s", path, '' if not dry_run else ' --dry-run')
    if dry_run:
        return
    md5, size = _get_file_check_data(path)
    data = {'name': name, 'md5': md5, 'size': size}
    file_info = figshare_request(
        'account/articles/%s/files' % article_id, method='POST', data=data)
    file_info = figshare_request(url=file_info['location'])
    result = figshare_request(url=file_info.get('upload_url'))
    with open(path, 'rb') as stream:
        for part in result['parts']:
            udata = file_info.copy()
            udata.update(part)
            url = '{upload_url}/{partNo}'.format(**udata)
            stream.seek(part['startOffset'])
            data = stream.read(part['endOffset'] - part['startOffset'] + 1)
            figshare_request(url=url, method='PUT', data=data, binary=True)
    endpoint = 'account/articles/{}/files/{}'.format(article_id, file_info['id'])
    figshare_request(endpoint, method='POST')


def figshare_upload_dir(root_dir, article_id, dry_run=False):
    """Upload to figshare all session files found in a root directory."""
    root_dir = Path(root_dir)

    # Get existing files on figshare to avoid uploading them twice.
    existing_files = set(_[0] for _ in figshare_files(article_id))

    for p in find_session_files(root_dir):
        # Upload all found files.
        name = _get_file_rel_path(str(p))
        if name not in existing_files:
            figshare_upload_file(p, name.replace('/', '~'), article_id, dry_run=dry_run)

    if dry_run:
        return

    # At the end, create the root file.
    make_figshare_root_file(article_id, root_dir / '.one_root')

    # Upload the root file to figshare.
    figshare_upload_file(root_dir / '.one_root', '.one_root', article_id)


def find_figshare_root_file(article_id):
    """Download and return the local path to the ONE root file of a figshare article."""
    root_file = config_dir() / ('figshare/%s/.one_root' % article_id)
    if root_file.exists():
        return root_file
    # If the root file does not exist, find it on figshare.
    files = figshare_request('account/articles/%s/files' % article_id)
    for file in files:
        if file['name'] == '.one_root':
            download_file(file['download_url'], root_file)
            assert root_file.exists()
            return root_file


class FigshareOne(HttpOne):
    def __init__(self, article_id=None, download_dir=None):
        root_file = find_figshare_root_file(article_id)
        super(FigshareOne, self).__init__(root_file=root_file, download_dir=download_dir)


# -------------------------------------------------------------------------------------------------
# ONE singleton
# -------------------------------------------------------------------------------------------------

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


def _make_figshare_one():
    """Create a new FigshareOne instance based on the config file."""
    # Full config dict.
    # TODO: support for multiple ONE repositories
    config = get_config()
    token = config.get('figshare_token', None)
    if not token:
        return
    article_id = config.get('figshare_article_id', None)
    return FigshareOne(article_id=article_id, download_dir=config.get('download_dir', None))


def get_one():
    """Get the singleton One instance, loading it from the config file, or using the singleton
    instance if it has already been instantiated."""
    if globals()['_ONE_SINGLETON'] is None:
        globals()['_ONE_SINGLETON'] = _make_figshare_one() or _make_http_one()
    one = globals()['_ONE_SINGLETON']
    assert one
    return one


def default_config():
    return {
        'http_config_root_file': '',
        'http_config_base_url': '',
        'download_dir': default_download_dir(),
        'http_config_auth': None,
        'figshare_token': '',
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
    """Search for all sessions that have all requested dataset types."""
    # NOTE: underscore to avoid shadowing of public search() function.
    # TODO: other search options
    for session in search(dataset_types):
        click.echo(session)


@one.command()
@click.argument('session')
@click.argument('obj', required=False)
@click.option('--dry-run', is_flag=True)
def download(session, obj=None, dry_run=False):
    """Download files in a given session by specifying a filename pattern (object,
    object.attribute, possibly with wildcards)."""
    for file_path in load_object(
            session, obj or '*', download_only=True, dry_run=dry_run).values():
        click.echo(file_path)


@one.command()
@click.argument('root_dir')
@click.option('--sessions-only', default=False, is_flag=True)
def scan(root_dir, sessions_only=False):
    """Scan all session files locally."""
    for p in (find_session_files(root_dir) if not sessions_only else find_session_dirs(root_dir)):
        click.echo(str(p))


@one.command()
@click.argument('article_id')
def figscan(article_id):
    """Scan all session files on figshare."""
    for rel_path, url in sorted(figshare_files(article_id), key=itemgetter(0)):
        click.echo(rel_path)


@one.command()
@click.argument('root_dir')
@click.argument('article_id')
@click.option('--dry-run', is_flag=True)
def upload(root_dir, article_id, dry_run=False):
    """Upload a root directory to a figshare article."""
    figshare_upload_dir(root_dir, article_id, dry_run=dry_run)


if __name__ == '__main__':
    one()

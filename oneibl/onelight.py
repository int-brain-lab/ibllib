"""ONE light."""

import csv
from collections import defaultdict
from ftplib import FTP, error_perm
import hashlib
import json
import logging
import os
import os.path as op
from pathlib import Path
import re
import sys
import tempfile
import urllib.parse
import warnings

import click
import requests
from requests.exceptions import HTTPError

import alf.io

logger = logging.getLogger(__name__)

warnings.warn('`oneibl.onelight` will be removed in future version', DeprecationWarning)


# -------------------------------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------------------------------

# Set a null handler on the root logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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


add_default_handler(level='DEBUG' if '--debug' in sys.argv else 'INFO')


# -------------------------------------------------------------------------------------------------
# Global variables
# -------------------------------------------------------------------------------------------------

EXCLUDED_FILENAMES = ('.DS_Store', '.one_root')
_FIGSHARE_BASE_URL = 'https://api.figshare.com/v2/{endpoint}'
_CURRENT_REPOSITORY = None
_CURRENT_ONE = None

DEFAULT_CONFIG = {
    "download_dir": "~/.one/data/{repository}/{lab}/Subjects/{subject}/{date}/{number}/alf/",
    "session_pattern": "^{lab}/Subjects/{subject}/{date}/{number}/$",
    "file_pattern": "^{lab}/Subjects/{subject}/{date}/{number}/alf/{filename}$",
    "repositories": [
        {
            "name": "default",
            "root_dir": ".",
            "type": "local"
        }
    ],
    "current_repository": None,
}

DOWNLOAD_INSTRUCTIONS = '''

<h3>ONE interface</h3>

<p>The data is available via the ONE interface.
<a href="https://github.com/int-brain-lab/ibllib/tree/master/oneibl#one-light">
Installation instructions here.</a>
</p>

<p>To search and download this dataset:</p>

<blockquote>
from oneibl.onelight import ONE
one = ONE()
sessions = one.search(['trials'])  # search for all sessions that have a `trials` object
session = sessions[0]  # take the first session
trials = one.load_object(session, 'trials')  # load the trials object
print(trials.intervals)  # trials is a Bunch, values are NumPy arrays or pandas DataFrames
print(trials.goCue_times)
</blockquote>

'''


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
# Config
# -------------------------------------------------------------------------------------------------

def config_dir():
    """Path to the config directory."""
    return Path.home() / '.one/'


def repo_dir():
    """Path to the local directory of the repository."""
    return config_dir() / 'data' / repository().name


def config_file():
    """Path to the config file."""
    return config_dir() / 'config.json'


def default_config():
    """Return an empty configuration dictionary."""
    return DEFAULT_CONFIG


def get_config():
    """Return the config file dictionary."""
    # Create a default config file if there is none.
    path = config_file()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        set_config(default_config())
    # Open the config file.
    with open(path, 'r') as f:
        return json.load(f)


def set_config(config):
    """Set the config file."""
    with open(config_file(), 'w') as f:
        json.dump(config, f, indent=2, sort_keys=True)


def get_repo(name=None, config=None):
    """Get a repository by its name."""
    config = config or get_config()
    for r in config['repositories']:
        if name and r['name'] == name:
            return r
        if not name and config['current_repository'] == r['name']:
            return r


def update_repo(name=None, **kwargs):
    """Update a repository."""
    config = get_config()
    repo = get_repo(name=name, config=config)
    if repo:
        repo.update(kwargs)
    set_config(config)


def add_repo(name=None):
    """Interactive prompt to add a repository."""
    if not name:
        name = input('Choose a repository name? (leave empty for default) ') or 'default'
    config = get_config()
    repo = get_repo(name, config=config)
    if not repo:
        repo = Bunch(name=name, type=input("`local`, `ftp`, `http`, or `figshare`? "))
        config['repositories'].append(repo)
    if repo.type == 'http':
        repo.update(
            base_url=input("Root URL? "),
            login=input("Basic HTTP auth login? (leave empty if public) "),
            pwd=input("Basic HTTP auth password? (leave empty if public) "),
        )
    elif repo.type == 'ftp':
        repo.update(
            host=input("Host? "),
            port=input("Port? (21 by default)") or 21,
            ftp_login=input("Login? "),
            ftp_password=input("Password? "),
            remote_root=input("Remote path to the root directory? (`/` by default) ") or '/',
            base_url=input("Public URL? "),
        )
    elif repo.type == 'figshare':
        repo.update(
            article_id=_parse_article_id(input("figshare article public URL? ")),
            token=input(
                "[data uploaders only] "
                "Go to https://figshare.com/account/applications, generate a token, "
                "and copy-paste it here: "),
        )
    elif repo.type == 'local':
        repo.update(
            root_dir=input('root path? '),
        )
    set_config(config)
    return repo


def set_figshare_url(url):
    """Get or create a figshare repo with a given figshare URL."""
    config = get_config()
    article_id = _parse_article_id(url)
    for repo in config['repositories']:
        if repo['type'] == 'figshare' and repo['article_id'] == article_id:
            return set_repo(repo['name'])
    # Need to add a new repo.
    # Find a new unique name figshare_XXX.
    names = set(r['name'] for r in config['repositories'])
    name = None
    for i in range(100):
        n = 'figshare_%02d' % i
        if n not in names:
            name = n
            break
    assert name
    repo = Bunch(name=name, type='figshare', article_id=article_id)
    config['repositories'].append(repo)
    set_config(config)
    set_repo(name)


# -------------------------------------------------------------------------------------------------
# File scanning and root file creation
# -------------------------------------------------------------------------------------------------
def read_root_file(path):
    assert path
    with open(path) as f:
        for line in csv.reader(f, delimiter='\t'):
            # If single column, prepend the base URL.
            if len(line) == 1:
                rel_path = line[0]
                base_url = repository().get('base_url', None) or ''
                line = [rel_path, urllib.parse.urljoin(base_url, line[0])]
            assert len(line) == 2
            assert isinstance(line[0], str)
            assert isinstance(line[1], str)
            yield line[0], line[1]


def write_root_file(path, iterator):
    assert path
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
    if path.name in EXCLUDED_FILENAMES:
        return False
    return not path.is_dir() and '/Subjects/' in str(path.parent.parent.parent).replace('\\', '/')


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


# -------------------------------------------------------------------------------------------------
# Download and load
# -------------------------------------------------------------------------------------------------

def download_file(url, save_to, auth=None, log_level=logging.DEBUG):
    """Download a file from HTTP and save it to a file.
    If Basic HTTP authentication is needed, pass `auth=(username, password)`.
    """
    save_to = Path(save_to)
    save_to.parent.mkdir(parents=True, exist_ok=True)
    logger.log(log_level, "Downloading %s to %s.", url, str(save_to))
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
    return '~/.one/data/{repository}/{lab}/Subjects/{subject}/{date}/{number}/alf/'


def download_dir():
    """Return the download directory."""
    return get_config().get('download_dir', None)


def format_download_dir(session, download_dir):
    """Replace the placeholder fields in the download directory by the appropriate values for
    a given session."""
    session_info = _parse_session_path(session)
    session_info['repository'] = repository().name
    download_dir = download_dir.format(**session_info)
    return Path(download_dir).expanduser()


def load_array(path):
    """Load a single file."""
    path = str(path)
    if path.endswith('.npy'):
        try:
            import numpy as np
            mmap_mode = 'r' if op.getsize(path) > 1e8 else None
            return np.load(path, mmap_mode=mmap_mode)
        except ImportError:
            logger.warning("NumPy is not available.")
            return
        except ValueError as e:
            logger.error("Impossible to read %s.", path)
            raise e
    elif path.endswith('.tsv'):
        try:
            import pandas as pd
            return pd.read_csv(path, sep='\t')
        except ImportError:
            logger.warning("Pandas is not available.")
        except ValueError as e:
            logger.error("Impossible to read %s.", path)
            raise e
    raise NotImplementedError(path)


# -------------------------------------------------------------------------------------------------
# File path parsing
# -------------------------------------------------------------------------------------------------

def _session_regex():
    return _pattern_to_regex(get_config()['session_pattern'])


def _file_regex():
    return _pattern_to_regex(get_config()['file_pattern'])


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


def _make_search_regex(**kwargs):
    """Make a regex for the search() function."""
    pattern = get_config()['file_pattern']

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


def _make_dataset_regex(session, dataset_type=None):
    """Make a regex for finding datasets of a given type in a session."""
    pattern = r'^{session}.+{dataset_type}$'

    # Dataset type.
    dataset_type = dataset_type or ''
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
    m = _session_regex().match(session)
    if not m:
        raise ValueError("The session path `%s` is invalid." % session)
    return {n: m.group(n) for n in ('lab', 'subject', 'date', 'number')}


def _parse_file_path(file_path):
    """Parse a file path."""
    m = _file_regex().match(file_path)
    if not m:
        raise ValueError("The file path `%s` is invalid." % file_path)
    return {n: m.group(n) for n in ('lab', 'subject', 'date', 'number', 'filename')}


def _get_file_rel_path(file_path):
    """Get the lab/Subjects/subject/... part of a file path."""
    file_path = str(file_path).replace('\\', '/')
    # Find the relative part of the file path.
    i = file_path.index('/Subjects')
    if '/' not in file_path[:i]:
        return file_path
    i = file_path[:i].rindex('/') + 1
    return file_path[i:]


def _search(root_file_iterator, regex):
    """Search an iterator among tuples (file_path, url) using a regex."""
    for rel_path, url in root_file_iterator:
        m = regex.match(rel_path)
        if m:
            yield rel_path, url, m


# -------------------------------------------------------------------------------------------------
# FTP Uploader
# -------------------------------------------------------------------------------------------------

class FtpUploader:
    def __init__(self, host, login=None, password=None, port=21, remote_root=None):
        self.host = host
        self.login = login
        self.password = password
        self.port = port
        self.remote_root = remote_root or '/'
        self._ftp = FTP()
        self._fr = None
        self._writer = None
        self.connect()

    def connect(self):
        # FTP connect.
        self._ftp.connect(self.host, self.port)
        self._ftp.login(self.login, self.password)
        logger.debug(self._ftp.getwelcome())
        # Go to the root directory.
        for n in self.remote_root.split('/'):
            n = n.strip()
            if n:
                logger.debug("Enter %s.", n)
                self._ftp.cwd(n)

    def upload(self, root_dir, base_dir=None):
        root_dir = Path(root_dir)
        base_dir = base_dir or root_dir
        # Write the .one_root file iteratively.
        if self._writer is None:
            self._fr = open(root_dir / '.one_root', 'w')
            self._writer = csv.writer(self._fr, delimiter='\t')
        for name in sorted(os.listdir(root_dir)):
            path = Path(op.join(root_dir, name))
            rel_path = path.relative_to(base_dir)
            if op.isfile(path) and is_file_in_session_dir(path):
                logger.debug("Upload %s.", path)
                self._writer.writerow([rel_path])
                with open(path, 'rb') as f:
                    self._ftp.storbinary('STOR ' + name, f)
            elif op.isdir(path):
                try:
                    logger.debug("Create FTP dir %s.", name)
                    self._ftp.mkd(name)
                except error_perm as e:
                    if not e.args[0].startswith('550'):
                        raise
                self._ftp.cwd(name)
                self.upload(path, base_dir=base_dir)
                self._ftp.cwd("..")
        # End: close the file and the FTP connection.
        if base_dir == root_dir:
            with open(root_dir / '.one_root', 'rb') as f:
                self._ftp.storbinary('STOR .one_root', f)
            self._fr.close()
            self._ftp.quit()


# -------------------------------------------------------------------------------------------------
# HTTP ONE
# -------------------------------------------------------------------------------------------------

class HttpOne:
    def __init__(self, root_file=None, download_dir=None, auth=None):
        assert root_file
        if not Path(root_file).exists():
            raise ValueError("Root file %s could not be found.", root_file)
        self.root_file = root_file
        self.download_dir = download_dir or default_download_dir()
        self.auth = auth

    def _iter_files(self):
        """Iterator over tuples (relative_path, full_path). It is up to the base class
        to implement the method for downloading the data."""
        yield from read_root_file(self.root_file)

    def _download_dataset(self, session, filename, url, dry_run=False):
        save_to_dir = Path(format_download_dir(session, self.download_dir))
        save_to = save_to_dir / filename
        if not save_to.exists():
            if not dry_run:
                download_file(url, save_to, auth=self.auth, log_level=logging.INFO)
                assert save_to.exists()
        else:
            logger.debug("Skip existing %s.", save_to)
        return save_to

    def list(self, session):
        """List all dataset types found in a session."""
        if not session.endswith('/'):
            session += '/'
        out = []
        for rel_path, _ in self._iter_files():
            if rel_path.startswith(session):
                out.append('.'.join(op.basename(rel_path).split('.')[:2]))
        return sorted(out)

    def search(self, dataset_types=(), **kwargs):
        """Search all sessions that have all requested dataset types."""
        if not dataset_types:
            # All sessions.
            return sorted(
                set('/'.join(_[0].split('/')[:5]) for _ in self._iter_files()))
        if isinstance(dataset_types, str):
            dataset_types = [dataset_types]
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
        for rel_path, url, m in _search(self._iter_files(), pattern):
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
        tup = next(_search(self._iter_files(), pattern))
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
        for rel_path, url, m in _search(self._iter_files(), pattern):
            filename = rel_path.split('/')[-1]
            fs = filename.split('.')
            attr = fs[1]
            out[attr] = self._download_dataset(session, filename, url, dry_run=dry_run)
            if not download_only and not dry_run:
                out[attr] = load_array(out[attr])
        alf.io.check_dimensions(out)
        return out


class LocalOne(HttpOne):
    def __init__(self, root_dir):
        assert root_dir
        root_dir = Path(root_dir).expanduser().resolve()
        if not root_dir.exists():
            raise ValueError("Root dir %s could not be found." % root_dir)
        if not root_dir.is_dir():
            raise ValueError("Root dir %s is not a directory." % root_dir)
        self.root_dir = root_dir
        self.relative_paths = list(
            (str(_get_file_rel_path(p)), str(p)) for p in find_session_files(root_dir))

    def _iter_files(self):
        """Iterator over tuples (relative_path, full_path)."""
        yield from self.relative_paths

    def _download_dataset(self, session, filename, url, dry_run=False):
        return url


# -------------------------------------------------------------------------------------------------
# figshare uploader
# -------------------------------------------------------------------------------------------------

def _parse_article_id(url):
    if url.endswith('/'):
        url = url[:-1]
    return int(url.split('/')[-1])


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


def figshare_request(
        endpoint=None, data=None, method='GET', url=None,
        binary=False, error_level=logging.ERROR):
    """Perform a REST request against the figshare API."""
    headers = {'Authorization': 'token ' + repository().get('token', '')}
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
        logger.log(error_level, error)
        raise error
    return data


class FigshareUploader:
    def __init__(self, article_id):
        assert article_id
        self.article_id = int(article_id)

    def _req(self, endpoint=None, private=True, **kwargs):
        return figshare_request('%sarticles/%d%s' % (
            'account/' if private else '',
            self.article_id,
            '/' + endpoint if endpoint else ''), **kwargs)

    def _get(self, endpoint=None, private=True, **kwargs):
        return self._req(endpoint, private=private, **kwargs)

    def _post(self, endpoint=None, private=True, **kwargs):
        return self._req(endpoint, private=private, method='POST', **kwargs)

    def iter_files(self, private=True):
        """Iterate over all ALF files of a given figshare article."""
        r = _file_regex()
        for file in self._get('files', private=private):
            path = file['name'].replace('~', '/')
            if r.match(path):
                yield path, file['download_url']

    def _make_root_file(self, output):
        """Create a root file for a figshare article."""
        write_root_file(output, self.iter_files())

    def _upload(self, path, name, dry_run=False):
        """Upload a single file to figshare."""
        # see https://docs.figshare.com/#upload_files_example_upload_on_figshare
        assert Path(path).exists()
        logger.info("Uploading %s%s", path, '' if not dry_run else ' --dry-run')
        if dry_run:
            return
        md5, size = _get_file_check_data(path)
        data = {'name': name, 'md5': md5, 'size': size}
        file_info = self._post('files', data=data)
        file_info = self._get(url=file_info['location'])
        result = self._get(url=file_info.get('upload_url'))
        with open(path, 'rb') as stream:
            for part in result['parts']:
                udata = file_info.copy()
                udata.update(part)
                url = '{upload_url}/{partNo}'.format(**udata)
                stream.seek(part['startOffset'])
                data = stream.read(part['endOffset'] - part['startOffset'] + 1)
                self._req(url=url, method='PUT', data=data, binary=True)
        self._post('files/%s' % file_info['id'])
        return file_info['id'], file_info['download_url']

    def _publish(self):
        logger.debug("Publishing new version for article %d." % self.article_id)
        self._post('publish')

    def _update_description(self):
        """Append the ONE interface doc at the end of the article's description."""
        description = self._get('').get('description', '')
        if 'ONE interface' not in description:
            description += DOWNLOAD_INSTRUCTIONS.replace('\n', '<br>')
            logger.debug("Updating description of article %d." % self.article_id)
            self._req('', method='PUT', data={'description': description})

    def _delete(self, *file_ids, pattern=None):
        file_ids = list(file_ids)
        # Find the files to delete based on a regex pattern.
        if not file_ids:
            r = re.compile(pattern)
            for f in self._get('files'):
                if r.match(f['name']):
                    file_ids.append(f['id'])
        # Delete all specified files.
        for file_id in file_ids:
            logger.debug("Delete file %s.", file_id)
            self._req('files/%s' % file_id, method='DELETE', error_level=logging.DEBUG)

    def _find_root_file(self, private=True, use_cache=True):
        """Download and return the local path to the ONE root file of a figshare article."""
        root_file = repo_dir() / '.one_root'
        if use_cache and root_file.exists():
            return root_file
        # If the root file does not exist, find it on figshare.
        for file in self._get('files', private=private):
            if file['name'] == '.one_root':
                download_file(file['download_url'], root_file)
                assert root_file.exists()
                return root_file

    def _update_root_file(self, root_dir):
        """Create and upload the root file, replacing any existing one."""
        # At the end, create the root file.
        root_file_path = root_dir / '.one_root'
        self._make_root_file(root_file_path)
        assert root_file_path.exists()

        # Delete the old .one_root files
        logger.debug("Deleting old versions of .one_root.")
        self._delete(pattern=r'.+\.one_root')

        # Upload the new root file to figshare.
        root_file_id = self._upload(root_file_path, '.one_root')[0]

        # Update the root file id in the config file.
        update_repo(root_file_id=root_file_id)

    def upload_dir(self, root_dir, dry_run=False, limit=None):
        """Upload to figshare all session files found in a root directory."""
        root_dir = Path(root_dir)

        # Get existing files on figshare to avoid uploading them twice.
        existing_files = set(_[0] for _ in self.iter_files())

        uploaded = []
        for p in find_session_files(root_dir):
            if limit and len(uploaded) >= limit:
                break
            # Upload all found files.
            name = _get_file_rel_path(str(p))
            if name not in existing_files:
                try:
                    up = self._upload(p, name.replace('/', '~'), dry_run=dry_run)
                except Exception:
                    break
                uploaded.append(up)
        if dry_run:
            logger.debug("Skip uploading.")
            return
        logger.info("Uploaded %d new files.", len(uploaded))

        # Create and upload the root file.
        self._update_root_file(root_dir)

        # Add the download instructions in the description.
        self._update_description()

        # Validate all changes.
        # self._publish()

    def _remove_duplicates(self):
        """Remove duplicate files."""
        existing = set()
        for rel_path, _ in self.iter_files(private=True):
            if rel_path in existing:
                self._delete(pattern=_escape_for_regex(rel_path.replace('/', '~')))
                existing.add(rel_path)

    def clean_publish(self):
        """Clean up and publish the figshare article."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Remove duplicates.
            self._remove_duplicates()
            # Update the ONE root.
            self._update_root_file(Path(tmpdir))
            # Publish.
            self._publish()


# -------------------------------------------------------------------------------------------------
# figshare client
# -------------------------------------------------------------------------------------------------

class FigshareOne(HttpOne):
    def __init__(self, article_id=None, download_dir=None, private=False):
        # NOTE: we don't use the cache if ever the data changes remotely.
        # NOTE: we use the public version here as the client may not have access to the private
        # version of the figshare article.
        root_file = FigshareUploader(article_id)._find_root_file(private=private, use_cache=False)
        if not root_file:
            raise ValueError(
                "No ONE root file could be found in figshare article %d." % article_id)
        super(FigshareOne, self).__init__(root_file=root_file, download_dir=download_dir)


# -------------------------------------------------------------------------------------------------
# ONE singleton
# -------------------------------------------------------------------------------------------------

def set_repo(name=None):
    """Set the current repository."""
    config = get_config()
    name = name or config.get('current_repository', None) or 'default'
    repos = config.get('repositories', [])
    if not repos:
        logger.error("No repository configured!")
        add_repo(name)
        return set_repo(name)
    for repo in repos:
        repo = Bunch(repo)
        if repo.name == name:
            globals()['_CURRENT_REPOSITORY'] = repo
            break
    config['current_repository'] = repo.name
    set_config(config)
    logger.debug("Current repository is: `%s`.", repo.name)
    return repo


def repository():
    """Get the current repository."""
    if not globals()['_CURRENT_REPOSITORY']:
        globals()['_CURRENT_REPOSITORY'] = set_repo()
    return Bunch(globals()['_CURRENT_REPOSITORY'])


def _make_http_one(repo):
    """Create a new HttpOne instance based on the config file."""
    # Optional authentication.
    auth = (repo.login, repo.get('password', None)) if 'login' in repo else None
    # Download the root file from the HTTP server.
    root_file = repo_dir() / '.one_root'
    if not root_file.exists():
        root_url = urllib.parse.urljoin(repo.base_url, '.one_root')
        download_file(root_url, root_file, auth=auth)
    assert root_file.exists()
    return HttpOne(root_file=root_file, download_dir=download_dir(), auth=auth)


def _make_figshare_one(repo, private=False):
    """Create a new FigshareOne instance based on the config file."""
    return FigshareOne(article_id=repo.article_id, download_dir=download_dir(), private=private)


def _make_local_one(repo):
    """Create a new LocalOne instance based on the config file."""
    return LocalOne(repo.root_dir)


def get_one(private=False):
    """Get the singleton One instance, loading it from the config file, or using the singleton
    instance if it has already been instantiated."""
    if globals()['_CURRENT_ONE'] is not None:
        return globals()['_CURRENT_ONE']
    repo = repository()
    if repo.type in ('http', 'ftp'):
        globals()['_CURRENT_ONE'] = _make_http_one(repo)
    elif repo.type == 'figshare':
        globals()['_CURRENT_ONE'] = _make_figshare_one(repo, private=private)
    elif repo.type == 'local':
        globals()['_CURRENT_ONE'] = _make_local_one(repo)
    else:
        raise NotImplementedError(repo.type)
    return globals()['_CURRENT_ONE']


# -------------------------------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------------------------------

class ONE(object):
    def set_local_dir(self, name=None, **kwargs):
        update_repo(name=name, **kwargs)

    def set_figshare_url(self, url):
        set_figshare_url(url)

    def search_terms(self, ):
        return ('lab', 'subject', 'date', 'number', 'dataset_types')

    def set_download_dir(self, path):
        """Set the download directory. May contain fields like {lab}, {subject}, etc."""
        # Update the config dictionary.
        config = get_config()
        config['download_dir'] = str(path)
        set_config(config)
        # Update the current ONE instance.
        if globals()['_CURRENT_ONE']:
            globals()['_CURRENT_ONE'].download_dir = path

    @is_documented_by(HttpOne.search)
    def search(self, dataset_types, private=False, **kwargs):
        return get_one(private=private).search(dataset_types, **kwargs)

    @is_documented_by(HttpOne.list)
    def list(self, session):
        return get_one().list(session)

    @is_documented_by(HttpOne.load_object)
    def load_object(self, session, obj=None, **kwargs):
        return get_one().load_object(session, obj, **kwargs)

    @is_documented_by(HttpOne.load_dataset)
    def load_dataset(self, session, dataset_type, **kwargs):
        return get_one().load_dataset(session, dataset_type, **kwargs)


# -------------------------------------------------------------------------------------------------
# Command-line interface
# -------------------------------------------------------------------------------------------------

@click.group()
def one():
    """ONE light command-line tool for searching, downloading, and uploading data to an FTP server,
    or to figshare."""
    pass


@one.command('repo')
@click.argument('name', required=False)
def repo(name=None):
    """Show the existing repos, or set the current repo."""
    if not name:
        repos = get_config().get('repositories', [])
        for repo in repos:
            repo = Bunch(repo)
            is_current = '*' if repo.name == repository().name else ''
            click.echo(
                f'{is_current}{repo.name}: {repo.type} '
                f'{repo.get("base_url", repo.get("article_id", ""))}')
    else:
        set_repo(name)


@one.command('add_repo')
@click.argument('name', required=False)
def add_repo_(name=None):
    """Add a new repository and prompt for its configuration info."""
    repo = add_repo(name)
    set_repo(repo.name)


@one.command('search')
@click.argument('dataset_types', nargs=-1)
@click.option('--private', is_flag=True)
@is_documented_by(ONE.search)
def search_(dataset_types, private=False):
    # NOTE: underscore to avoid shadowing of public search() function.
    # TODO: other search options
    for session in ONE().search(dataset_types, private=private):
        click.echo(session)


@one.command('list')
@click.argument('session')
@is_documented_by(ONE.list)
def list_cli(session):
    for dataset_type in ONE().list(session):
        click.echo(dataset_type)


@one.command()
@click.argument('session')
@click.argument('obj', required=False)
@click.option('--dry-run', is_flag=True)
def download(session, obj=None, dry_run=False):
    """Download files in a given session."""
    for file_path in ONE().load_object(
            session, obj or '*', download_only=True, dry_run=dry_run).values():
        click.echo(file_path)


@one.command()
@click.argument('root_dir')
@click.option('--sessions-only', default=False, is_flag=True)
def scan(root_dir, sessions_only=False):
    """Scan all files locally."""
    for p in (find_session_files(root_dir) if not sessions_only else find_session_dirs(root_dir)):
        click.echo(str(p))


@one.command()
@click.argument('root_dir')
@click.option('--limit', type=int, default=0, help="maximum number of files to upload")
@click.option('--dry-run', is_flag=True)
def upload(root_dir, limit=0, dry_run=False):
    """Upload a root directory to a figshare article."""
    repo = repository()
    if repo.type == 'figshare':
        FigshareUploader(repo.article_id).upload_dir(root_dir, limit=limit, dry_run=dry_run)
    elif repo.type == 'ftp':
        fu = FtpUploader(
            repo.host, login=repo.ftp_login, password=repo.ftp_password, port=repo.port or 21,
            remote_root=repo.remote_root)
        fu.upload(root_dir)
    else:
        raise NotImplementedError("Upload only possible for figshare repositories.")


@one.command('clean_publish')
@is_documented_by(FigshareUploader.clean_publish)
def clean_publish():
    repo = repository()
    if repo.type != 'figshare':
        raise NotImplementedError("Upload only possible for figshare repositories.")
    FigshareUploader(repo.article_id).clean_publish()

"""
python one_iblrig.py extract /path/to/my/session/
python one_iblrig.py register /path/to/my/session/
python one_iblrig.py extract /path/to/my/session/ --dry=True
python one_iblrig.py register /path/to/my/session/ --dry True
"""

import logging
import argparse
from pathlib import Path

from alf import extract_session
from oneibl.registration import RegistrationClient
from oneibl.one import ONE

logger = logging.getLogger('ibllib')
# set the logging level to paranoid
logger.setLevel('INFO')


def extract(root_data_folder, dry=False):
    extract_session.bulk(root_data_folder, dry=dry)


def register(root_data_folder, dry=False):
    # registration part
    one = ONE(base_url='https://alyx.internationalbrainlab.org')
    rc = RegistrationClient(one=one)
    rc.register_sync(root_data_folder, dry=dry)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('action', help='Action: extract or register ')
    parser.add_argument('folder', help='A Folder containing a session')
    parser.add_argument('--dry', help='Dry Run', required=False, default=False, type=bool)
    args = parser.parse_args()  # returns data from the options specified (echo)
    assert(Path(args.folder).exists())
    if args.action == 'extract':
        extract(args.folder, dry=args.dry)
    if args.action == 'register':
        register(args.folder, dry=args.dry)
    print('done')

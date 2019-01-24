import logging

from alf import extract_session
from oneibl.registration import RegistrationClient
from oneibl.one import ONE

# set the logging level to paranoid
logger = logging.getLogger('ibllib')
logger.setLevel('INFO')

# extraction part
ROOT_DATA_FOLDER = '/datadisk/FlatIron/wittenlab/Subjects'
extract_session.bulk(ROOT_DATA_FOLDER)

# # registration part
one = ONE(base_url='https://dev.alyx.internationalbrainlab.org')
rc = RegistrationClient(one=one)
rc.register_sync(ROOT_DATA_FOLDER)

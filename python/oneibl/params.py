import oneibl.params_secret as sec
import os

# BASE_URL = "http://localhost:8000"
# BASE_URL = 'https://dev.alyx.internationalbrainlab.org'
BASE_URL = r'https://test.alyx.internationalbrainlab.org'

ALYX_LOGIN  = r'test_user'
ALYX_PWD = sec.ALYX_PWD

HTTP_DATA_SERVER = r'http://ibl.flatironinstitute.com'
HTTP_DATA_SERVER_LOGIN = r'iblmember'
HTTP_DATA_SERVER_PWD = sec.HTTP_DATA_SERVER_PWD  # password for data server

# if empty it will download in the user download directory
CACHE_DIR = r'/home/nico/Downloads/FlatIron'
if CACHE_DIR and not os.path.isdir(CACHE_DIR):
    os.mkdir(CACHE_DIR)

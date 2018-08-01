import oneibl.params_secret as sec
# import one_ibl.params as par

# BASE_URL = "https://alyx.cortexlab.net"
# BASE_URL = "https://alyx-dev.cortexlab.net"
BASE_URL = "http://localhost:8000"

ALYX_LOGIN = 'olivier'
ALYX_PWD = sec.ALYX_PWD

HTTP_DATA_SERVER = r'http://ibl.flatironinstitute.org/cortexlab'
HTTP_DATA_SERVER_LOGIN = 'ibl'
HTTP_DATA_SERVER_PWD = sec.HTTP_DATA_SERVER_PWD  # password for data server

CACHE_DIR = '/home/owinter/Downloads/FlatIronCache'  # if empty it will download in the user download directory

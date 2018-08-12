## Single session with an URL
from oneibl.one import ONE
myone = ONE() # need to instantiate the class to have the API. But the methods may be static in other implementations

eid = 'http://localhost:8000/sessions/698361f6-b7d0-447d-a25d-42afdef7a0da'
# eid = 'http://localhost:8000/sessions/2afc2091-d554-4a77-9f29-24008ae92ee0'
sess = myone._alyxClient.get(eid)

## Implements the list
from oneibl.misc import pprint




##
dataset_types = []  # 30 sessions
users=['']      # a list
subject='' # a string
url = '/sessions?'  # session query
date_range = ['2018-03-01', '2018-03-31']


## TESTS THAT WORK
import json
json_file = '/home/owinter/Downloads/2018-04-17_1_MW53_parameters.5b32fbd5-3749-4920-9811-38cdd62b8ab3.json'
with open(json_file) as f:
    data = json.load(f)


##
##
import getpass
from oneibl.utils import AlyxClient
from oneibl.misc import pprint, isostr2date, date2isostr
import pandas
import oneibl.params as par
import numpy as np

ac = AlyxClient(username=par.ALYX_LOGIN , password=par.sec.ALYX_PWD)


##
sessions_table = np.load('/home/owinter/Documents/IBL/sessions_2018_07_05.pkl')
users_set = np.unique(sessions_table.users);
nusers = np.array([len(l) for l in users_set])

##
users_set[nusers==3]

# we should probably have a lab field in the database
##
## Single session with an URL
import oneibl.one
import importlib
importlib.reload(oneibl.one)
from oneibl.one import ONE
from ibllib.misc import pprint

myone = ONE() # need to instantiate the class

ac = myone._alyxClient

## FIXME: comment faire un AND
ses = ac.get('/sessions?users=Morgane') # 183 results: this is not a strict lookup (ie. can't return only Morgane's sessions)
ses = ac.get('/sessions?users=Morgane,miles,armin&users=armin') # 20 results: en fait ce n'est pas un AND mais bien un OR

ses = ac.get('/sessions?users=Morgane,miles,armin&users=nick') # 20 results: en fait ce n'est pas un AND mais bien un OR




np.unique([s['users'] for s in ses])

## Sheduler
ses = ac.get('/sessions?date_range=2018-03-01,2018-04-01&users=miles')
# NB: pour l'implémentation bien faire attention à mettre la date de fin plus un jour pour que le dernier jour soit inclu
dmin = min([isostr2date(s['start_time']+'.00') for s in ses])
dmax = max([isostr2date(s[  'end_time']+'.00') for s in ses])

print(np.unique([s['users'] for s in ses]))


##
myone._alyxClient.authenticate(username='dfia',password= 'asdf', base_url='http://localhost:8000')
myone._alyxClient._token

##
from oneibl.one import ONE
myone = ONE() # need to instantiate the class

subject = {'nickname':'toto',
           'responsible_user':'olivier',
           'protocol_number': '1',
            'project':'<Project_test_IBL>',
           'genotype': []}

subject = {'nickname':'toto',
           'responsible_user':'olivier',
           'protocol_number': '1',
            'project':'<Project_test_IBL>'}

r = myone._alyxClient.post('/subjects', data=subject)

##
session =  {'subject': 'clns0730',
            'procedures': ['Behavior training/tasks'],
            'narrative': 'auto-generated session',
            'start_time': '2018-05-18T12:12:12',
            'type': 'Base',
            'number':'1',
            'users': ['olivier']}


r = myone._alyxClient.post('/sessions', data=session)

##
r = myone._alyxClient.get('/data-repositories')
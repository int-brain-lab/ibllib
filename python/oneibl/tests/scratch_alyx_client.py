##
from oneibl.webclient import AlyxClient
import os
import pandas
# small library of useful functions
import json
import datetime
isostr2date = lambda isostr:  datetime.datetime.strptime(isostr, '%Y-%m-%dT%H:%M:%S.%f')
date2isostr = lambda adate:   datetime.datetime.isoformat(adate)
pprint = lambda my_dict: print(json.dumps(my_dict, indent=5))

# Init connection to the database
ac = AlyxClient()


## Implement complex requests through the REST API

# https://django-filter.readthedocs.io/en/1.1.0/guide/usage.html

## downloads a big pile of stuff to look at off-line (takes a little while even for a local request)
# subjects_table = ac.get('/subjects')
# sessions_table = ac.get('/sessions')
def get_full_table(url):
    t = time.time()
    table = ac.get(url)
    print(time.time() - t)
    out_chem = "/home/owinter/Documents/IBL"
    # csv_time = datetime.datetime.now().strftime('%Y_%m_%d')
    csv_time = '2018_07_05'
    table = pandas.DataFrame(table)
    fname = out_chem + os.sep + url[1:] + '_' + csv_time
    table.to_csv(fname + '.csv')
    table.to_pickle(fname + '.pkl')
    return table

# datasets_table = get_full_table('/datasets') # 42 secs
# subjects_table = get_full_table('/subjects') # 2 secs
# sessions_table = get_full_table('/sessions') # 51 secs
##
users = ac.get('/users')
##
subj = ac.get('/subjects')
subj = ac.get('/subjects?responsible_user=Hamish')

##
sess = ac.get('/sessions/742e9031-1c34-47e8-a7fe-0351173f78bb') # returns a dict
sess = ac.get('/sessions?starts_after=2018-03-24') # ca c'est long:
a = [datetime.datetime.strptime((s['start_time']), '%Y-%m-%dT%H:%M:%S') for s in sess]
print(len(sess), min(a), max(a))
##
import time

start = time.time()
sess = ac.get('/sessions?dataset_types=cwGoCue.times,cwFeedback.type') #83 secs,
end = time.time()
print(end - start)
start = time.time()
sess = ac.get('/sessions?dataset_types=expDefinition,Parameters,wheel.timestamps') # 3.5 secs, 32 records
end = time.time()
print(end - start)


##
dset = ac.get('/datasets?created_datetime_lte=2018-01-01T00:00:00.00000') # returns a list
dset = ac.get('/datasets?created_datetime_lte=2018-01-01') # returns a list
dset = ac.get('/datasets?pk=1') # return a dict
dset = ac.get('/datasets?session=ad167732-bc95-4064-a483-0186927e18e0')
dset = ac.get('/datasets?created_by=Hamish')
dset = ac.get('/datasets?username=Hamish')


##
dtypes  = ac.get('/dataset-types')





##
dset = ac.get('/datasets?created_by=Hamish&dataset_type=Block&created_datetime_lte=2018-01-01')
set([d['dataset_type'] for d in dset])
a = [isostr2date(d['created_datetime']) for d in dset]
print(len(dset), min(a), max(a))

##
dtyp = ac.get('/dataset-types')

dset = ac.get('/datasets?dataset_type=cwResponse.times')
dset = ac.get('/datasets?dataset_type=cwResponse.choice')

##

# https://docs.djangoproject.com/en/2.0/topics/db/queries/
from data.models import Dataset
from data.serializers import DatasetSerializer

from django.db.models import Max, Min

print(Dataset.objects.all().count())

dsets = Dataset.objects.filter(created_by__username='Hamish')
print(dsets.count())

dsets = Dataset.objects.filter(dataset_type__name='Block')
print(dsets.count())

dsets = Dataset.objects.filter(created_by__username='Hamish')
dsets = dsets.filter(dataset_type__name='Block')
print(dsets.count())


# OrderedUser.objects.values()
dsets.aggregate(Max("created_datetime"))
dsets.aggregate(Min("created_datetime"))
##

from rest_framework.request import Request
from rest_framework.test import APIRequestFactory

factory = APIRequestFactory()
request = factory.get('/',)

serializer_context = {
    'request': Request(request),
}

sz = DatasetSerializer(dsets, context=serializer_context, many=True)
sz.data


import datetime
start_date = datetime.date(2005, 1, 1)
end_date = datetime.date(2005, 3, 31)


##



# Init connection to the database
ac = AlyxClient()
##
rep = ac.get('/sessions?dataset_types=cwGoCue.times,cwFeedback.type')

##
rep = ac.get('/sessions?dataset_types=expDefinition,Parameters,wheel.timestamps')


## FIXME: this should result in an empty set or an error, not in a long query
# rep = ac.get('/sessions?bimbamboom')





## Init
from oneibl.one import ONE
myone = ONE() # need to instantiate the class to have the API.

## Load #1
dataset_types = ['clusters.templateWaveforms', 'clusters.probes', 'clusters.depths']
eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
wf, pr, d = myone.load(eid, dataset_types=dataset_types)

## Load #2
my_data = myone.load(eid, dataset_types=dataset_types, dclass_output=True)
from ibllib.misc import pprint
pprint(my_data.local_path)
pprint(my_data.dataset_type)

## Load everything
eid, ses_info = myone.search(subject='flowers')
my_data = myone.load(eid[0])
pprint(my_data.dataset_type)

## Load
eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
dataset_types = ['clusters.probes', 'thisDataset.IveJustMadeUp', 'clusters.depths']
t, empty, cl = myone.load(eid, dataset_types=dataset_types)

## List #1
myone.list(table='dataset-types', verbose=True)

## List #2
list_types , dtypes = myone.list(table=['dataset-types','users'])
pprint(list_types)
pprint(dtypes)

## Search users
eid, ses_info = myone.search(users=['olivier'])
pprint(ses_info)

eid, ses_info = myone.search(users=['nbonacchi', 'olivier'])

## Search by date
eid = myone.search(users='olivier', date_range=['2018-08-24', '2018-08-24'])

## Init
from oneibl.one import ONE
one = ONE(base_url='https://test.alyx.internationalbrainlab.org', username='test_user',
          password='TapetesBloc18')
## Info about a session
eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
list_of_datasets = one.list(eid)

## More Info about a session
d = one.session_data_info(eid)
print(d)

## Load #1
dataset_types = ['clusters.templateWaveforms', 'clusters.probes', 'clusters.depths']
eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
wf, pr, d = one.load(eid, dataset_types=dataset_types)

## Load #2
my_data = one.load(eid, dataset_types=dataset_types, dclass_output=True)
from ibllib.misc import pprint
pprint(my_data.local_path)
pprint(my_data.dataset_type)

## Load everything
eid, ses_info = one.search(subjects='flowers')
my_data = one.load(eid[0])
pprint(my_data.dataset_type)

## Load
eid = '86e27228-8708-48d8-96ed-9aa61ab951db'
dataset_types = ['clusters.probes', 'thisDataset.IveJustMadeUp', 'clusters.depths']
t, empty, cl = one.load(eid, dataset_types=dataset_types)

## List #1
one.ls_dataset_types()
one.ls_users()
one.ls_subjects()

## List #2
list_types , dtypes = one.ls(table=['dataset-types','users'])
pprint(list_types)
pprint(dtypes)

## Search users
eid, ses_info = one.search(users=['olivier'])
pprint(ses_info)

eid, ses_info = one.search(users=['nbonacchi', 'olivier'])

## Search by date
eid = one.search(users='olivier', date_range=['2018-08-24', '2018-08-24'])

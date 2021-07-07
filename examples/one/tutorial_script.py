## Init
from one.api import ONE

from pprint import pprint

one = ONE(base_url='https://test.alyx.internationalbrainlab.org',
          username='test_user',
          password='TapetesBloc18')

## Find an experiment
eid = one.search(users='olivier', date_range=['2018-08-24', '2018-08-24'])
pprint(eid)
pprint(one.search_terms)

## List dataset types for a session
eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
one.list_datasets(eid)

## List all datasets for a given collection
one.list_datasets(eid, collection='alf/probe00')

## List all datasets
one.list_datasets()

## Load object
eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
probe_label = 'probe00'
clusters = one.load_object(eid, 'clusters', collection=f'alf/{probe_label}')

## Load one dataset
depths = one.load_dataset(eid, 'clusters.depths.npy', collection=f'alf/{probe_label}')

## Load datasets
eid = 'cf264653-2deb-44cb-aa84-89b82507028a'
dsets = ['clusters.probes.npy', 'thisDataset.IveJustMadeUp', 'clusters.depths.npy']
t, empty, cl = one.load_datasets(eid, dsets,
                                 collections=f'alf/{probe_label}',
                                 assert_present=False)
## Load everything
eid = one.search(subjects='flowers')
all_datasets = one.list_datasets(eid, details=True)
files = one._download_datasets(all_datasets)
pprint(files)

# FIXME Doesn't work yet
## Search users
eid = one.search(users='olivier', query_type='remote')
eid = one.search(users=['nbonacchi', 'olivier'], query_type='remote')

# with details
eid, session_details = one.search(user=['test_user', 'olivier'], details=True)
pprint(session_details)

## Search by date
eid, details = one.search(date_range=['2018-08-24', '2018-08-24'], details=True)

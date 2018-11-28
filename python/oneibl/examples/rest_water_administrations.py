from ibllib.misc import pprint
from oneibl.one import ONE

one = ONE(base_url='https://dev.alyx.internationalbrainlab.org')

# list all water administrations
wa = one.alyx.rest('water-administrations', 'list')

# to list administrations for one subject, it is better to use the subjects endpoint
sub_info = one.alyx.rest('subjects', 'read', 'ZM_346')
pprint(sub_info['water_administrations'])

# this is how to programmatically create a water administration

wa_ = {
    'subject': 'ZM_346',
    'date_time': '2018-11-25T12:34',
    'water_administered': 25,
    'water_type': 'Water 10% Sucrose',
    'user': 'olivier',
    'session': 'f4b13ba2-1308-4673-820e-0e0a3e0f2d73',
    'adlib': True}

# do not use the example on anything else than alyx-dev
if one.alyx._base_url != 'https://dev.alyx.internationalbrainlab.org':
    rep = one.alyx.rest('water-administrations', 'create', wa_)

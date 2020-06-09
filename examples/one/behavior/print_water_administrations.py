'''
Print water administration values from behavior data downloaded via ONE.
'''
#  Author: Olivier Winter

from ibllib.misc import pprint
from oneibl.one import ONE

one = ONE(base_url='https://dev.alyx.internationalbrainlab.org')

# -- Get saved water administration --
# List all water administrations
wa = one.alyx.rest('water-administrations', 'list')

# To list administrations for one subject, it is better to use the subjects endpoint
subject_details = one.alyx.rest('subjects', 'read', 'ZM_346')
pprint(subject_details['water_administrations'][0:2])  # Print the first 2 water admin.

'''
Register histology session for example mouse
Note, we have defined: start_time = sample_imaging_date
'''
# Author: Gaelle Chapuis

import datetime
from oneibl.one import ONE
import ibllib.time

# Test first on dev alyx for example
one = ONE(base_url='https://dev.alyx.internationalbrainlab.org')

subject = 'CSHL028'  # example
TASK_PROTOCOL = 'SWC_Histology_Serial2P_v0.0.1'

# Date-Time of imaging (example), change as needed
sample_imaging_date = datetime.datet(2020, 2, 1)  # Format: y - m - d
sample_reception_date = datetime.date(2020, 4, 1)

json_note = {'sample_reception_date': ibllib.time.date2isostr(sample_reception_date)[:10]}
# Saving only the date

ses_ = {'subject': subject,
        'users': ['steven.west'],
        'location': 'serial2P_01',
        'procedures': ['Histology'],
        'lab': 'mrsicflogellab',
        # 'project': project['name'],
        # 'type': 'Experiment',
        'task_protocol': TASK_PROTOCOL,
        'number': 1,
        'start_time': ibllib.time.date2isostr(sample_imaging_date)[:10],  # Saving only the date
        # 'end_time': ibllib.time.date2isostr(end_time) if end_time else None,
        # 'n_correct_trials': n_correct_trials,
        # 'n_trials': n_trials,
        'json': json_note
        }
session = one.alyx.rest('sessions', 'create', data=ses_)

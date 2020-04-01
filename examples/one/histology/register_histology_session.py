'''
Register histology session for example mouse
'''
# Author: Gaelle Chapuis

import datetime
from oneibl.one import ONE

# Test first on dev alyx for example
one = ONE(base_url='https://dev.alyx.internationalbrainlab.org')

subject = 'CSHL028' # example
TASK_PROTOCOL = 'SWC_Histology_Serial2P_v0.0.1'
# Date-Time of registration
start_time = datetime.datetime.now()

# Date-Time of imaging (example)
img_date = datetime.datetime(2020, 4, 1, 17, 28, 55, 536948)
json_note = {'imaging_time' = img_date}  # TODO Does not work written like this

ses_ = {'subject': subject,
        'users': 'steven.west',
        'location': 'serial2P_01',
        'procedures': 'Histology',
        'lab': 'mrsicflogellab',
        # 'project': project['name'],
        # 'type': 'Experiment',
        'task_protocol': TASK_PROTOCOL,
        'number': 1,
        'start_time': ibllib.time.date2isostr(start_time),
        # 'end_time': ibllib.time.date2isostr(end_time) if end_time else None,
        # 'n_correct_trials': n_correct_trials,
        #'n_trials': n_trials,
        'json': json_note
        }
session = one.alyx.rest('sessions', 'create', data=ses_)
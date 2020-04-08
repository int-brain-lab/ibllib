'''
Register histology session for example mouse
Note, we have defined: start_time = sample_imaging_date
'''
# Author: Gaelle Chapuis, Steven J. West

import datetime
from oneibl.one import ONE
import ibllib.time
import numpy as np
import json
from json import JSONEncoder

# override deault method of JSONEncoder to implement custom NumPy JSON serialization.
# see https://pynative.com/python-serialize-numpy-ndarray-into-json/
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Test first on dev alyx for example
one = ONE(base_url='https://dev.alyx.internationalbrainlab.org')

subject = 'CSHL028'  # example
TASK_PROTOCOL = 'SWC_Histology_Serial2P_v0.0.1'

# Date-Time of imaging (example), change as needed
sample_imaging_date = datetime.date(2020, 2, 1)  # Format: y - m - d
sample_reception_date = datetime.date(2020, 4, 1)

# create elastix afffine transform numpy array:
array = np.zeros((4, 4)) #UPDATE with correct transform!
elastix_affine_transform = {"elastix_affine_transform": array}

json_note = {
        'sample_reception_date': ibllib.time.date2isostr(sample_reception_date),
        'elastix_affine_transform': array,
        'tilt': 0,
        'yaw': 0,
        'roll': 0,
        'dv_scale': 1,
        'ap_scale': 1,
        'ml_scale': 1
}

# use dump() to properly encode np array:
json_note = json.dumps(json_note, cls=NumpyArrayEncoder)

ses_ = {
        'subject': subject,
        'users': ['steven.west'],
        'location': 'serial2P_01',
        'procedures': ['Histology'],
        'lab': 'mrsicflogellab',
        # 'project': project['name'],
        # 'type': 'Experiment',
        'task_protocol': TASK_PROTOCOL,
        'number': 1,
        'start_time': ibllib.time.date2isostr(sample_imaging_date),  # Saving only the date
        # 'end_time': ibllib.time.date2isostr(end_time) if end_time else None,
        # 'n_correct_trials': n_correct_trials,
        # 'n_trials': n_trials,
        'json': json_note
}

# overwrites the session if it already exists
ses_date = ibllib.time.date2isostr(sample_imaging_date)[:10]
ses = one.alyx.rest('sessions', 'list', subject=subject, number=1,
                    date_range=[ses_date, ses_date])
if len(ses) > 0:
        one.alyx.rest('sessions', 'delete', ses[0]['url'])

session = one.alyx.rest('sessions', 'create', data=ses_)


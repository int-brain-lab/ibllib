"""
Get subject training status via ONE
===================================
Use ONE to get the training status of a chosen subject or all subjects within a lab.
Training status is computed based on performance over latest 3 sessions (default) or last 3
sessions before a specified date.
"""
from one.api import ONE

import brainbox.behavior.training as training
one = ONE(silent=True)
# Get training status of a specific subject
training.get_subject_training_status('SWC_055', one=one)

# Get training status of a specific subject on a chosen date
training.get_subject_training_status('SWC_055', date='2020-09-01', one=one)

# Get training status of all mice within a lab
# (N.B. only looks for alive and water restricted subjects)
training.get_lab_training_status('danlab', one=one)

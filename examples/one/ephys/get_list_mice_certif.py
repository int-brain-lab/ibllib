'''
Get list of subjects associated to the certification recording project.
'''
# Author: Gaelle Chapuis

from oneibl.one import ONE
one = ONE()

eid, det = one.search(project='ibl_certif_neuropix_recording', details=True)
sub = [p['subject'] for p in det]
"""
Get first pass map sessions
===========================
Use ONE to get information about sessions included in first pass map

"""

from oneibl.one import ONE
one = ONE()

first_pass_map_sessions = one.alyx.rest('sessions', 'list', project='ibl_neuropixel_brainwide_01')
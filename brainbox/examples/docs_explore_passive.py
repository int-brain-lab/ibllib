"""
Explore Passive Data
====================
This example shows how to load in passive data for a session and plot the receptive field map over
depth as well as some task aligned activity
"""

eid = ''

##################################################################################################
# Look at receptive field part of passive protocol

rf_map = bbone.load_rf_map_data(eid, one=one)

# Now pass this into brainbox passive functions. First we want to arrange our rfmap data into
rf_stim_times, rf_stim_pos, rf_stim_frames = passive.g


'''
Register on Alyx the set of tracked traces (after histology) for a given mouse.

All your tracks should be in a single folder, and the files names should follow the nomenclature
{yyyy-mm-dd}_{SubjectName}_{SessionNumber}_{ProbeLabel}_pts.csv

Edit the variable 'path_tracks'(set it either to your local directory [example given here],
either to the Google folder if synched to your machine).

To check if the registration worked, go to the admin interface
> experiments > trajectory estimates > search for the subject

If you want to test first, use  ALYX_URL = "https://dev.alyx.internationalbrainlab.org"
And check the data appears on:
https://dev.alyx.internationalbrainlab.org/admin/experiments/trajectoryestimate/?

When you feel confident you can upload without error,
change to the   ALYX_URL = "https://alyx.internationalbrainlab.org"
and re-run.
'''
# Author: Olivier Winter

from ibllib.pipes import histology
from oneibl.one import ONE

# ======== EDIT FOR USERS ====

# Edit so as to reflect the directory containing your electrode tracks
path_tracks = "/Users/gaelle/Downloads/00_to_add"

ALYX_URL = "https://dev.alyx.internationalbrainlab.org"  # FOR TESTING
# ALYX_URL = "https://alyx.internationalbrainlab.org"  # UNCOMMENT WHEN READY

# ======== DO NOT EDIT BELOW ====
one = ONE(base_url=ALYX_URL)
histology.register_track_files(path_tracks=path_tracks, one=one, overwrite=True)

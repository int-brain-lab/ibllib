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
set EXAMPLE_OVERWRITE = False ,
change to the   ALYX_URL = "https://alyx.internationalbrainlab.org"
and re-run.

With EXAMPLE_OVERWRITE = True, the script downloads an example dataset and runs
the registration (used for automatic testing of the example).
'''
# Author: Olivier, Gaelle

from ibllib.pipes import histology
from oneibl.one import ONE
from pathlib import Path

# ======== EDIT FOR USERS ====

# Edit so as to reflect the directory containing your electrode tracks
path_tracks = "/Users/gaelle/Downloads/Flatiron/examples/00_to_add"


EXAMPLE_OVERWRITE = True  # Put to False when wanting to run the script on your data

ALYX_URL = "https://dev.alyx.internationalbrainlab.org"  # FOR TESTING
# ALYX_URL = "https://alyx.internationalbrainlab.org"  # UNCOMMENT WHEN READY

# ======== DO NOT EDIT BELOW ====
one = ONE(base_url=ALYX_URL)

if EXAMPLE_OVERWRITE:
    # TODO Olivier : Function to download examples folder
    cachepath = Path(one._par.CACHE_DIR)
    path_tracks = cachepath.joinpath('examples', 'histology', 'tracks_to_add')

histology.register_track_files(path_tracks=path_tracks, one=one, overwrite=True)
histology.detect_missing_histology_tracks(path_tracks=path_tracks, one=one)

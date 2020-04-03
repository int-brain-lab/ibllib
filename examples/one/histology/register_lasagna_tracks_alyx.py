'''
Register on Alyx the set of tracked traces (after histology) for a given mouse.

All your tracks should be in a single folder, and the files names should follow the nomenclature
{yyyy-mm-dd}_{SubjectName}_{SessionNumber}_{ProbeLabel}_pts.csv

Edit the variable 'path_tracks'(set it either to your local directory [example given here],
either to the Google folder if synched to your machine).

To check if the registration worked, go to the admin interface
> experiments > trajectory estimates > search for the subject

If you want to test first, use  ALYX_URL = "https://dev.alyx.internationalbrainlab.org"
And check on:
https://dev.alyx.internationalbrainlab.org/admin/experiments/trajectoryestimate/?

When you feel confident you can upload without error,
change to the   ALYX_URL = "https://alyx.internationalbrainlab.org"

'''
# Author: Olivier Winter
from pathlib import Path
from ibllib.pipes import histology

from oneibl.one import ONE

# ======== EDIT FOR USERS ====

# Edit so as to reflect the directory containing your electrode tracks
path_tracks = "/Users/gaelle/Downloads/electrodetracks_lic3"

ALYX_URL = "https://dev.alyx.internationalbrainlab.org"  # FOR TESTING
# ALYX_URL = "https://alyx.internationalbrainlab.org"  # UNCOMMENT WHEN READY


# ======== DO NOT EDIT BELOW ====
# ALYX_URL = "http://localhost:8000"
glob_pattern = "*_probe*_pts*.csv"

one = ONE(base_url=ALYX_URL)

path_tracks = Path(path_tracks)
assert path_tracks.exists()


def parse_filename(track_file):
    tmp = track_file.name.split('_')
    inumber = [i for i, s in enumerate(tmp) if s.isdigit and len(s) == 3][-1]
    search_filter = {'date': tmp[0], 'experiment_number': int(tmp[inumber]),
                     'name': '_'.join(tmp[inumber+1:-1]),
                     'subject': '_'.join(tmp[1:inumber])}
    return search_filter


for track_file in path_tracks.rglob(glob_pattern):
    # '{yyyy-mm-dd}}_{nickname}_{session_number}_{probe_label}_pts.csv'
    # beware: there may be underscores in the subject nickname
    print(track_file)
    search_filter = parse_filename(track_file)
    probe = one.alyx.rest('insertions', 'list', **search_filter)
    if len(probe) == 0:
        eid = one.search(subject=search_filter['subject'], date_range=search_filter['date'],
                         number=search_filter['experiment_number'])
        if len(eid) == 0:
            raise Exception("No session found")
        insertion = {'session': eid[0],
                     'name': search_filter['name']}
        probe = one.alyx.rest('insertions', 'create', data=insertion)
    elif len(probe) == 1:
        probe = probe[0]
    else:
        raise ValueError("Multiple probes found.")
    probe_id = probe['id']
    xyz_picks = histology.load_track_csv(track_file)
    try:
        histology.register_track(probe_id, xyz_picks, one=one)
    except Exception as e:
        pass


def test_filename_parser():
    tdata = [
        {'input': Path("/gna/electrode_tracks_SWC_014/2019-12-12_SWC_014_001_probe01_fit.csv"),
          'output': {'date': '2019-12-12', 'experiment_number': 1, 'name': 'probe01',
                     'subject': 'SWC_014'}},
        {'input': Path("/gna/datadisk/Data/Histology/"
                       "tracks/ZM_2407/2019-11-06_ZM_2407_001_probe_00_pts.csv"),
         'output': {'date': '2019-11-06', 'experiment_number': 1, 'name': 'probe_00',
                    'subject': 'ZM_2407'}},
        {'input': Path("/gna/2019-12-06_KS023_001_probe01_pts.csv"),
         'output': {'date': '2019-12-06', 'experiment_number': 1, 'name': 'probe01',
                    'subject': 'KS023'}},
        ]
    for t in tdata:
        track_file = t['input']
        assert t['output'] == parse_filename(track_file)

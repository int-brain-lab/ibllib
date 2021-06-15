from oneibl.one import ONE
from ibllib.atlas import atlas

from brainbox.io.one import load_channels_from_insertion
one = ONE()
pid = "8413c5c6-b42b-4ec6-b751-881a54413628"
ba = atlas.AllenAtlas()

xyz = load_channels_from_insertion(one.alyx.rest('insertions', 'read', id=pid), one=one, ba=ba)

from oneibl.one import ONE

one = ONE()

# query sessions that have histology available
ses = one.alyx.rest('sessions', 'list', histology=True)
# the generic way
ses = one.alyx.rest('sessions', 'list',
                    django="subject__actions_sessions__procedures__name,Histology")

# query channel locations (hierarchical, will fetch everything below)
ses = one.alyx.rest('sessions', 'list', atlas_id=500)
ses = one.alyx.rest('sessions', 'list', atlas_acronym="MO")
ses = one.alyx.rest('sessions', 'list', atlas_name="Somatomotor areas")

from ibllib.io import globus as gl


def test_globus_1():
    assert gl.globus_client_id()

    g = gl.Globus()
    assert g.ls('flatiron')

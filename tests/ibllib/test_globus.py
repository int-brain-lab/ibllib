from ibllib.io import globus as gl


def test_globus_1():
    assert gl.globus_client_id()

    g = gl.Globus()
    assert g.ls('test')

    assert not g.file_exists('test', 'test/NOT_EXISTS')
    assert g.file_exists('test', 'test/README')

    assert g.files_exist('test', 'test/', ['NOT_EXISTS.txt', 'README']) == [False, True]

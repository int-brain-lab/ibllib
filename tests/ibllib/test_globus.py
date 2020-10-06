from ibllib.io import globus as gl
from pytest import fixture


@fixture
def g():
    return gl.Globus()


def test_globus_1():
    assert gl.globus_client_id()

    g = gl.Globus()
    assert g.ls('test')


def test_globus_2(g):
    assert not g.file_exists('test', 'test/NOT_EXISTS')
    assert g.file_exists('test', 'test/README')


def test_globus_3(g):
    assert g.dir_contains_files('test', 'test/', ['NOT_EXISTS.txt', 'README']) == [False, True]

    assert g.files_exist(
        'test', ['test/NOT_EXISTS.txt', 'test/README', 'toto']) == [False, True, False]


def test_globus_4(g):
    files = ['test/README', 'test/empty', 'NOT_EXISTS']

    assert g.files_exist('test', files, [None, None, None]) == [True, True, False]
    assert g.files_exist('test', files, [0, 0, 0]) == [False, True, False]
    assert g.files_exist('test', files, [39, 39, 39]) == [True, False, False]


def test_globus_5(g):
    print("Adding text file")
    g.add_text_file('flatiron', 'test_file', 'hello world')
    print("Done!")
    assert g.file_exists('flatiron', 'test_file', size=11)
    g.rm('flatiron', 'test_file', blocking=True)
    assert not g.file_exists('flatiron', 'test_file', size=11)


def test_globus_6(g):
    assert g.files_exist(
        'test',
        ['test/_flatiron_test.txt',
        'test/_flatiron_empty.txt']) == [False, False]

    g.move_files(
        'flatiron', 'test',
        ['public/test.txt',
         'public/empty'],
        ['test/_flatiron_test.txt',
         'test/_flatiron_empty.txt'],
        blocking=True)

    assert g.files_exist(
        'test',
        ['test/_flatiron_test.txt',
         'test/_flatiron_empty.txt']) == [True, True]

    g.rm('test', 'test/_flatiron_test.txt', blocking=True)
    g.rm('test', 'test/_flatiron_empty.txt', blocking=True)

    assert g.files_exist(
        'test',
        ['test/_flatiron_test.txt',
         'test/_flatiron_empty.txt']) == [False, False]

"""
utils functions to work on the sdsc instance
"""
from pathlib import Path

ROOT_PATH = Path('/mnt/ibl')
SERVER_URL = "https://ibl.flatironinstitute.org/"


def path_from_dataset(dset):
    fr = next((fr for fr in dset['file_records'] if fr['data_url']))
    return path_from_filerecord(fr)


def path_from_filerecord(fr):
    return Path(fr['data_url'].replace(SERVER_URL, str(ROOT_PATH) + '/'))

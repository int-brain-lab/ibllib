#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Thursday, March 28th 2019, 7:53:44 pm
"""
Purge data from RIG
- Find all files by rglob
- Find all sessions of the found files
- Check Alyx if corresponding datasetTypes have been registered as existing
sessions and files on Flatiron
- Delete local raw file if found on Flatiron
"""
from alf.folders import session_name
from pathlib import Path
from oneibl.one import ONE
import argparse


def purge_local_data(local_folder, file_name, lab=None, dry=False):
    # Figure out datasetType from file_name or file path
    file_name = Path(file_name).name
    alf_parts = file_name.split('.')
    dstype = '.'.join(alf_parts[:2])
    print(f'Looking for file <{file_name}> in folder <{local_folder}>')
    # Get all paths for file_name in local folder
    local_folder = Path(local_folder)
    files = list(local_folder.rglob(f'*{file_name}'))
    print(f'Found {len(files)} files')
    print(f'Checking on Flatiron for datsetType: {dstype}...')
    # Get all sessions and details from Alyx that have the dstype
    one = ONE()
    if lab is None:
        eid, det = one.search(dataset_types=[dstype], details=True)
    else:
        eid, det = one.search(dataset_types=[dstype], lab=lab, details=True)
    urls = []
    for d in det:
        urls.extend([x['data_url'] for x in d['data_dataset_session_related']
                     if x['dataset_type'] == dstype])
    # Remove None answers when session is registered but dstype not htere yet
    urls = [u for u in urls if u is not None]
    print(f'Found files on Flatiron: {len(urls)}')
    to_remove = []
    for f in files:
        sess_name = session_name(f)
        for u in urls:
            if sess_name in u:
                to_remove.append(f)
    print(f'Local files to remove: {len(to_remove)}')
    for f in to_remove:
        print(f)
        if dry:
            continue
        else:
            f.unlink()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Delete files from rig')
    parser.add_argument('folder', help='Local iblrig_data folder')
    parser.add_argument(
        'file', help='File name to search and destroy for every session')
    parser.add_argument('-lab', required=False, default=None,
                        help='Lab name, search on Alyx faster. default: None')
    parser.add_argument('--dry', required=False, default=False,
                        action='store_true', help='Dry run? default: False')
    args = parser.parse_args()
    purge_local_data(args.folder, args.file, lab=args.lab, dry=args.dry)
    print('Done\n')

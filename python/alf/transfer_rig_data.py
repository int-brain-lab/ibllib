# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Wednesday, January 16th 2019, 2:03:59 pm
# @Last Modified by: Niccolò Bonacchi
# @Last Modified time: 16-01-2019 02:04:01.011
import shutil
from shutil import ignore_patterns as ig
import sys
from pathlib import Path

from ibllib.io.raw_data_loaders import read_flag_file


def main(local_folder: str, remote_folder: str) -> None:
    local_folder = Path(local_folder)
    remote_folder = Path(remote_folder)

    src_session_paths = [x.parent for x in local_folder.rglob("transfer_me.flag")]

    # Create all dst paths
    dst_session_paths = []
    for s in src_session_paths:
        mouse = s.parts[-3]
        date = s.parts[-2]
        sess = s.parts[-1]
        d = remote_folder / mouse / date / sess
        dst_session_paths.append(d)

    for src, dst in zip(src_session_paths, dst_session_paths):
        flag_file = src / "transfer_me.flag"
        flag = read_flag_file(flag_file)
        if isinstance(flag, list):
            raise NotImplementedError
        else:
            shutil.copytree(src, dst, ignore=ig(str(flag_file.name)))
        # finally if folder was created delete the src flag_file
        if dst.exists():
            flag = dst / 'extract_me.flag'
            open(flag, 'a').close()
            flag_file.unlink()
            print(f"Copied to {remote_folder}: Session {flag_file.parent}")


if __name__ == "__main__":
    # main(local_folder, remote_folder)
    if len(sys.argv) < 3:
        print("ERROR: Not enough inputs")
    elif len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        print("ERROR: Too many inputs")

#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Wednesday, January 16th 2019, 2:03:59 pm
import argparse
import logging
import shutil
from pathlib import Path
from shutil import ignore_patterns as ig

import ibllib.io.extractors.base
import ibllib.io.flags as flags

log = logging.getLogger('ibllib')
log.setLevel(logging.INFO)


def main(local_folder: str, remote_folder: str, force: bool = False) -> None:
    local_folder = Path(local_folder)
    remote_folder = Path(remote_folder)

    src_session_paths = [x.parent for x in local_folder.rglob("transfer_me.flag")]

    if not src_session_paths:
        log.info("Nothing to transfer, exiting...")
        return

    # Create all dst paths
    dst_session_paths = []
    for s in src_session_paths:
        mouse = s.parts[-3]
        date = s.parts[-2]
        sess = s.parts[-1]
        d = remote_folder / mouse / date / sess
        dst_session_paths.append(d)

    for src, dst in zip(src_session_paths, dst_session_paths):
        src_flag_file = src / "transfer_me.flag"
        flag = flags.read_flag_file(src_flag_file)
        if isinstance(flag, list):
            raise NotImplementedError
        else:
            if force:
                shutil.rmtree(dst, ignore_errors=True)
            log.info(f"Copying {src}...")
            shutil.copytree(src, dst, ignore=ig(str(src_flag_file.name)))
        # finally if folder was created delete the src flag_file and create compress_me.flag
        if dst.exists():
            task_type = ibllib.io.extractors.base.get_session_extractor_type(Path(src))
            if task_type not in ['ephys', 'ephys_sync', 'ephys_mock']:
                flags.write_flag_file(dst.joinpath('raw_session.flag'))
            log.info(f"Copied to {remote_folder}: Session {src_flag_file.parent}")
            src_flag_file.unlink()

        # Cleanup
        src_audio_file = src / 'raw_behavior_data' / '_iblrig_micData.raw.wav'
        src_video_file = src / 'raw_video_data' / '_iblrig_leftCamera.raw.avi'
        dst_audio_file = dst / 'raw_behavior_data' / '_iblrig_micData.raw.wav'
        dst_video_file = dst / 'raw_video_data' / '_iblrig_leftCamera.raw.avi'

        if src_audio_file.exists() and \
                src_audio_file.stat().st_size == dst_audio_file.stat().st_size:
            src_audio_file.unlink()

        if src_video_file.exists() and \
                src_video_file.stat().st_size == dst_video_file.stat().st_size:
            src_video_file.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transfer files to IBL local server')
    parser.add_argument(
        'local_folder', help='Local iblrig_data/Subjects folder')
    parser.add_argument(
        'remote_folder', help='Remote iblrig_data/Subjects folder')
    args = parser.parse_args()
    main(args.local_folder, args.remote_folder)

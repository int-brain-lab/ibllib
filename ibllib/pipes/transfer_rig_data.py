#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date: Wednesday, January 16th 2019, 2:03:59 pm
import logging
import shutil
import argparse
from pathlib import Path
from shutil import ignore_patterns as ig

import ibllib.io.flags as flags
from ibllib.pipes import extract_session

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
            task_type = extract_session.get_task_extractor_type(Path(src))
            _create_flags_for_task(dst, task_type)
            log.info(
                f"Copied to {remote_folder}: Session {src_flag_file.parent}")
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


def _create_flags_for_task(dst, task_type):
    # create_flags_for_session()
    if task_type in ['habituation']:
        flags.write_flag_file(dst.joinpath('extract_me.flag'))
    if task_type in ['training', 'biased']:
        flags.write_flag_file(dst.joinpath('extract_me.flag'))
        flags.create_compress_video_flags(dst, flag_name='compress_video.flag')
        flags.create_audio_flags(dst, 'audio_training.flag')
    elif task_type in ['ephys', 'ephys_sync']:
        # Ephys flags copied by transfer_ephys_session from ephyspc
        flags.create_compress_video_flags(dst, flag_name='compress_video_ephys.flag')
        flags.create_audio_flags(dst, 'audio_ephys.flag')
    elif task_type in ['ephys_mock']:
        flags.write_flag_file(dst.joinpath('extract_ephys.flag'))
        flags.create_compress_video_flags(dst, flag_name='compress_video_ephys.flag')
        flags.create_audio_flags(dst, 'audio_ephys.flag')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Transfer files to IBL local server')
    parser.add_argument(
        'local_folder', help='Local iblrig_data/Subjects folder')
    parser.add_argument(
        'remote_folder', help='Remote iblrig_data/Subjects folder')
    args = parser.parse_args()
    main(args.local_folder, args.remote_folder)

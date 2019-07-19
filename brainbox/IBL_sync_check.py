import ibllib.dsp as dsp
import ibllib.io.spikeglx
import ibllib.io.extractors.ephys_fpga
import cv2
import csv
import numpy as np
from datetime import datetime
import ibllib.plots
import ibllib.io.extractors.ephys_fpga as ephys_fpga
import matplotlib.pyplot as plt

sync_test_folder = '/home/mic/Downloads/FlatIron/20190710_sync_test_CCU/20190710_sync_test'

###########
'''
ephys
'''
###########


def get_ephys_data(sync_test_folder):

    startTime = datetime.now()
    output_path = sync_test_folder
    # full path to the raw ephys
    raw_ephys_apfile = (sync_test_folder + '/ephys/20190709_sync_right_g0_t0.imec.ap.bin')

    # load reader object, and extract sync traces
    sr = ibllib.io.spikeglx.Reader(raw_ephys_apfile)
    sync = ibllib.io.extractors.ephys_fpga._sync_to_alf(sr, output_path, save=False)
    # wg.print_progress()

    print('sample rate', sr.fs)
    print(datetime.now() - startTime)  # took 24 seconds to run
    return sr, sync


def first_occ_index(array, n_at_least):

    '''
    Getting index of first occurence in boolean array
    with at least n consecutive False entries
    '''
    curr_found_false = 0
    curr_index = 0
    for index, elem in enumerate(array):
        if not elem:
            if curr_found_false == 0:
                curr_index = index
            curr_found_false += 1
            if curr_found_false == n_at_least:
                return curr_index
        else:
            curr_found_false = 0


def event_extraction_and_comparison(sr, sync_test_folder):

    # it took 8 min to run that for 6 min of data, all 300 ish channels
    # weird channels for Guido's set: [36,75,112,151,188,227,264,303,317,340,379,384]

    # sr,sync=get_ephys_data(sync_test_folder)

    startTime = datetime.now()
    '''
    this function first finds the times of square signal fronts in ephys and
    compares them to corresponding ones in the sync signal.
    Iteratively for small data chunks
    '''

    BATCH_SIZE_SAMPLES = 50000

    wg = dsp.WindowGenerator(sr.ns, BATCH_SIZE_SAMPLES, overlap=1)

    # if the data is needed as well, loop over the file
    # raw data contains raw ephys traces, while raw_sync contains the 16 sync traces

    rawdata, _ = sr.read_samples(0, BATCH_SIZE_SAMPLES)
    _, chans = rawdata.shape

    temporal_errors = []
    d_errs = {}
    for j in range(chans):
        d_errs[j] = []

    k = 0
    missmatches_in_signals = []

    for first, last in list(wg.firstlast)[20:-20]:  # skip beginning and end of recording
        print('segment %s of %s' % (k, len(list(wg.firstlast)[20:-20])))
        k += 1

        rawdata, rawsync = sr.read_samples(first, last)

        # get fronts for sync signal
        diffs = np.diff(rawsync.T[0])
        sync_up_fronts = np.where(diffs == 1)[0] + first
        sync_down_fronts = np.where(diffs == -1)[0] + first

        # get fronts for each ephys channel
        obs, chans = rawdata.shape
        for i in range(chans):
            # i=0
            Mean = np.mean(rawdata.T[i])
            Std = np.std(rawdata.T[i])

            ups = np.invert(rawdata.T[i] > Mean + 6 * Std)
            downs = np.invert(rawdata.T[i] < Mean - 6 * Std)

            up_fronts = []
            down_fronts = []
            # Activity front at least 10 samples long (empirical)

            u = first_occ_index(ups, 10)

            try:
                up_fronts.append(u + first)
            except Exception:
                print('no up fronts detected in segment %s, channel %s' % (k, i))
                continue  # jump to next segment without any front times comparison

            while u < len(ups):
                w = u + 15000
                try:
                    u = first_occ_index(ups[w:], 10) + w + first
                    up_fronts.append(u)
                except Exception:
                    break

            u = first_occ_index(downs, 10)  # 10 is empirical

            try:
                down_fronts.append(u + first)
            except Exception:
                print('no down fronts detected in segment %s, channel %s' % (k, i))
                continue  # jump to next segment without any front times comparison

            while u < len(downs):
                w = u + 15000
                try:
                    u = first_occ_index(downs[w:], 10) + w + first
                    down_fronts.append(u)
                except Exception:
                    break

            if len(up_fronts) != len(sync_up_fronts):
                print('differnt number of fronts detected; segment %s, \
                channel %s, difference %s' % (k, i, len(up_fronts) - len(sync_up_fronts)))
                missmatches_in_signals.append((k, i, len(up_fronts) - len(sync_up_fronts)))
                continue

            d_errs[i].append(np.mean(abs(np.array(up_fronts) - np.array(sync_up_fronts))))
            temporal_errors.append(np.mean(abs(np.array(up_fronts) - np.array(sync_up_fronts))))
            temporal_errors.append(
                np.mean(abs(np.array(down_fronts) - np.array(sync_down_fronts))))

    try:
        max_err = str(np.round(np.max(temporal_errors) / float(sr.fs), 10))
        av_error = str(np.round(np.mean(temporal_errors) / float(sr.fs), 10))
        print(
            'overall temporal error of all wavefronts and channels: \
             %s sec; maximal error: %s sec' % (av_error, max_err))
        print('time to run this function: ', datetime.now() - startTime)
    except Exception:
        print('no signals found to compare')

    channels_without_signals = []
    for ch in d_errs:
        if len(d_errs[ch]) == 0:
            channels_without_signals.append(ch)
        else:
            if np.max(d_errs[ch]) > 0.01 * sr.fs:
                print('large temporal error for channel %s at segment %s'
                      % (ch, np.argmax(d_errs[ch])))

    if all(np.abs(np.asarray(missmatches_in_signals).T[-1])):
        print('all signals detected')
    else:
        print('signal mismatch!!')

    return d_errs


def concat_rawdata(sr):

    # for plotting concatenate some data chunks; just to check

    # sr, sync=get_ephys_data(sync_test_folder)
    BATCH_SIZE_SAMPLES = 50000
    n_blocks = 20  # can't plot full data, choose some blocks that you chunk together

    wg = dsp.WindowGenerator(sr.ns, BATCH_SIZE_SAMPLES, overlap=1)
    rawdata, rawsync = sr.read_samples(0, BATCH_SIZE_SAMPLES)
    obs, chans = rawdata.shape
    D_data = np.zeros([n_blocks * BATCH_SIZE_SAMPLES, chans])
    obs, chans = rawsync.shape
    D_sync = np.zeros([n_blocks * BATCH_SIZE_SAMPLES, chans])

    for i in range(n_blocks):
        first, last = list(wg.firstlast)[i]
        rawdata, rawsync = sr.read_samples(first, last)
        D_data[BATCH_SIZE_SAMPLES * i:BATCH_SIZE_SAMPLES * (i + 1)] = rawdata
        D_sync[BATCH_SIZE_SAMPLES * i:BATCH_SIZE_SAMPLES * (i + 1)] = rawsync

    return D_data, D_sync


###########
'''
video
'''
###########


def convert_pgts(time):

    """Convert PointGray cameras timestamps to seconds.
    Use convert then uncycle"""
    # offset = time & 0xFFF
    cycle1 = (time >> 12) & 0x1FFF
    cycle2 = (time >> 25) & 0x7F
    seconds = cycle2 + cycle1 / 8000.
    return seconds


def uncycle_pgts(time):

    """Unwrap the converted seconds of a PointGray camera timestamp series."""
    cycles = np.insert(np.diff(time) < 0, 0, False)
    cycleindex = np.cumsum(cycles)
    return time + cycleindex * 128


def get_video_stamps_and_brightness(sync_test_folder):

    # for each frame in the video, set 1 or zero corresponding to LED status
    startTime = datetime.now()

    d = {}
    # '_iblrig_leftCamera.raw.avi', took it out, as it was faulty in Guido's data
    vids = ['_iblrig_bodyCamera.raw.avi', '_iblrig_rightCamera.raw.avi']

    # 1 min for 30 Hz video, 5 min for 150 Hz video, 2 min for 60 Hz -> maybe 9 min in total!

    for vid in vids:
        video_path = sync_test_folder + '/video/' + vid

        print(video_path)
        cap = cv2.VideoCapture(video_path)
        # fps = cap.get(cv2.CAP_PROP_FPS)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        brightness = np.zeros(frameCount)

        # for each frame, save brightness in array
        for i in range(frameCount):
            cap.set(1, i)
            _, frame = cap.read()
            brightness[i] = np.sum(frame)

        with open(video_path[:-4] + '_timestamps.ssv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=' ')
            ssv_times = np.array([line for line in csv_reader])

        ssv_times_sec = [convert_pgts(int(time)) for time in ssv_times[:, 0]]
        uncycle_pgts(ssv_times_sec)

        d[vid] = [brightness, uncycle_pgts(ssv_times_sec)]

    cap.release()
    print(datetime.now() - startTime)

    return d


def plot_camera_sync(d, sync):

    # d=get_video_stamps_and_brightness(sync_test_folder)
    # sr, sync, rawdata, rawsync=get_ephys_data(sync_test_folder)

    # using the probe 3a channel map:
    '''
    0: Arduino synchronization signal
    2: 150 Hz camera
    3: 30 Hz camera
    4: 60 Hz camera
    7: Bpod
    11: Frame2TTL
    12 & 13: Rotary Encoder
    15: Audio
    '''
    y = {
        '_iblrig_bodyCamera.raw.avi': 3,
        '_iblrig_rightCamera.raw.avi': 4,
        '_iblrig_leftCamera.raw.avi': 2}

    s3 = ephys_fpga._get_sync_fronts(sync, 0)  # get arduino sync signal

    for vid in d:
        # threshold brightness time-series of the camera to have it in {-1,1}
        r3 = [1 if x > np.mean(d[vid][0]) else -1 for x in d[vid][0]]
        # get 30 Hz fpga cam time stamps
        cam_times = ephys_fpga._get_sync_fronts(sync, y[vid])['times']

        drops = len(cam_times) - len(r3) * 2  # assuming at the end the frames are dropped
        print('%s frames dropped for %s' % (drops, vid))

        # plotting if you like
        plt.figure(vid)
        ibllib.plots.squares(s3['times'], s3['polarities'], label='fpga square signal', marker='o')
        plt.plot(cam_times[:-drops][0::2], r3, alpha=0.5,
                 label='thresholded video brightness', linewidth=2, marker='x')

        # ibllib.plots.vertical_lines(s3['times'])
        plt.legend()
        plt.title(vid)
        plt.show()
        ###########################

        # get fronts of video brightness square signal
        # manually found that len(cam_times_fpga)=2*len(r3)+8
        diffr3 = np.diff(r3)  # get signal jumps via differentiation
        fronts_brightness = []
        for i in range(len(diffr3)):
            if diffr3[i] != 0:
                fronts_brightness.append(cam_times[:-drops][0::2][i])

        if len(fronts_brightness) != len(s3['times']):
            print('Not all square signals were seen by the camera!')

        D = abs(fronts_brightness - s3['times'])
        print('Mean and std of difference between wave fronts, %s:' % vid, np.mean(D), np.std(D))


##########
'''
BPod
'''
##########


def compare_bpod_jason_with_fpga(sync_test_folder, sync):

    '''
    sr, sync=get_ephys_data(sync_test_folder)
    '''

    # get the bpod signal from the jasonable file
    import json
    with open(sync_test_folder + '/bpod/_iblrig_taskData.raw.jsonable') as fid:
        out = json.load(fid)

    ins = out['Events timestamps']['Port1In']
    outs = out['Events timestamps']['Port1Out']

    # get the fpga signal from the sync object
    s3 = ephys_fpga._get_sync_fronts(sync, 0)  # 3b channel map
    plt.plot(s3['times'], s3['polarities'])

    plt.plot(ins, np.ones(len(ins)), linestyle='', marker='o')
    plt.plot(outs, np.ones(len(outs)), linestyle='', marker='x')

    # patched_file = sync_test_folder+'/bpod/_iblrig_taskData.raw.jsonable'
    # with open(patched_file, 'w+') as fid:
    #  fid.write(json.dumps(out))
    #  ibllib.io.jsonable.read(patched_file)

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

    BATCH_SIZE_SAMPLES = 50000

    # full path to the raw ephys
    raw_ephys_apfile = (
        '/home/mic/Downloads/FlatIron/20190710_sync_test_CCU/20190710_sync_test/ephys/20190709_sync_right_g0_t0.imec.ap.bin')

    # load reader object, and extract sync traces
    sr = ibllib.io.spikeglx.Reader(raw_ephys_apfile)
    sync = ibllib.io.extractors.ephys_fpga._sync_to_alf(sr, output_path, save=False)

    # if the data is needed as well, loop over the file
    # raw data contains raw ephys traces, while raw_sync contains the 16 sync traces
    wg = dsp.WindowGenerator(sr.ns, BATCH_SIZE_SAMPLES, overlap=1)
    for first, last in wg.firstlast:
        rawdata, rawsync = sr.read_samples(first, last)
        wg.print_progress()

    print('sample rate', sr.fs)
    print(datetime.now() - startTime)  # took 24 seconds to run
    return sr, sync, rawdata, rawsync


###########
'''
video
'''
###########


def convert_pgts(time):
    """Convert PointGray cameras timestamps to seconds.
    Use convert then uncycle"""
    #offset = time & 0xFFF
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
    # '_iblrig_leftCamera.raw.avi',
    vids = ['_iblrig_bodyCamera.raw.avi', '_iblrig_rightCamera.raw.avi']

    # 1 min for 30 Hz video, 5 min for 150 Hz video, 2 min for 60 Hz -> maybe 9 min in total!

    for vid in vids:
        video_path = sync_test_folder + '/video/' + vid

        print(video_path)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frameCount / fps

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


##########
'''
BPod
'''
##########


def get_port1in(sync_test_folder):

    import json
    with open(sync_test_folder + '/bpod/_iblrig_taskData.raw.jsonable') as fid:
        out = json.load(fid)

    return out['Events timestamps']['Port1In']

# patched_file = sync_test_folder+'/bpod/_iblrig_taskData.raw.jsonable'
# with open(patched_file, 'w+') as fid:
#  fid.write(json.dumps(out))
#  ibllib.io.jsonable.read(patched_file)


#########
'''
plot and evaluate synchronicity
'''
#########


def plot_camera_sync(d, sync):

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
    y = {'_iblrig_bodyCamera.raw.avi': 3, '_iblrig_rightCamera.raw.avi': 4, '_iblrig_leftCamera.raw.avi': 2}

    s3 = ephys_fpga._get_sync_fronts(sync, 0)  # get arduino sync signal

    for vid in d:
        # threshold brightness time-series of the camera to have it in {-1,1}
        r3 = [1 if x > np.mean(d[vid][0]) else -1 for x in d[vid][0]]

        cam_times = ephys_fpga._get_sync_fronts(
            sync, y[vid])['times']  # get 30 Hz fpga cam time stamps

        drops = len(cam_times) - len(r3) * 2  # assuming at the end the frames are dropped
        print('%s frames dropped for %s' % (drops, vid))

        plt.figure(vid)
        ibllib.plots.squares(s3['times'], s3['polarities'], label='fpga square signal', marker='o')
        plt.plot(cam_times[:-drops][0::2], r3, alpha=0.5,
                 label='thresholded video brightness', linewidth=2, marker='x')

        # ibllib.plots.vertical_lines(s3['times'])
        plt.legend()
        plt.title(vid)
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

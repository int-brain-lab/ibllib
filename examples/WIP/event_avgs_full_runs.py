import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import multiprocessing
from event_avgs import *

# Constants and global vars
SCALED_LEN = 250 # Number of values to interpolate spike rates into
EID_INFO_PATH = "./data/filtered_eids/brainwide"  # Path to eid info dictionary
EVENT_NAMES = ["stimOn_times", "first_wheel_move", "feedback_times"]
NUM_LOG_BINS = 256 # compress outputs to uint8 using a log mapping

def event_avg_by_data_type(data_type, pids, trial_timing_dfs, avg_event_idxs):
    outpath = "./data/event_avgs/" + data_type + "/"

    if data_type == "raw_spikes":
      # Event avgs for raw spk/s
      # (with causal gaussian smoothing)
      event_average_all_session_firing_rates(outpath, pids, trial_timing_dfs, EVENT_NAMES,
        avg_event_idxs, one, norm_method=None, normalize=False, scaled_len=SCALED_LEN)

    elif data_type == "baselined":
      # Event avgs for raw spk/s
      # (with causal gaussian smoothing)
      event_average_all_session_firing_rates(outpath, pids, trial_timing_dfs, EVENT_NAMES,
        avg_event_idxs, one, norm_method=None, normalize=False, scaled_len=SCALED_LEN)

    elif data_type == "baselined_normalized":
      # Event avgs for baselined normalized spk/s
      event_average_all_session_firing_rates(outpath, pids, trial_timing_dfs, EVENT_NAMES,
        avg_event_idxs, one, norm_method="baseline", normalize=True, scaled_len=SCALED_LEN)

    elif data_type == "fano_factor":
      # Event avgs for fano factor
      event_average_all_session_firing_rates(outpath, pids, trial_timing_dfs, EVENT_NAMES,
        avg_event_idxs, one, norm_method="fano_factor", normalize=False, scaled_len=SCALED_LEN)

    compress_firing_data(pids, data_path=outpath, outpath=outpath, num_log_bins=NUM_LOG_BINS)


def event_avg_by_event(event_name, pids, trial_timing_dfs, timelocked_scaled_len,
                       timelocked_event_idxs):
    outpath = "./data/event_avgs/timelocked/" + str(event_name) + "/"

    # Here we cheat a bit by modifying trial_start and trial_end to be the same as the event window
    # (clamped to the actual trial start and end), then pass the altered timings into the same
    # main averaging function
    timelocked_timing_dfs = []
    for trdf_type_list in tqdm(trial_timing_dfs):
      trdf_list_copy = []
      for trdf in trdf_type_list:
        new_trial_starts = []
        new_trial_ends = []
        for idx, row in trdf.iterrows():
          event_time = row[event_name]
          new_trial_starts.append(max(row["trial_start"], event_time - event_window))
          new_trial_ends.append(min(row["trial_end"], event_time + event_window))

        trdf_copy = trdf.copy()
        trdf_copy["trial_start"] = new_trial_starts
        trdf_copy["trial_end"] = new_trial_ends
        trdf_list_copy.append(trdf_copy)
      timelocked_timing_dfs.append(trdf_list_copy)

    # Since we're only looking at the symmetric +- 0.5 sec around each event, we set the
    # event index to be right in the middle at SCALED_LEN // 2
    event_average_all_session_firing_rates(outpath, pids, timelocked_timing_dfs, [event_name],
                                          timelocked_event_idxs, one, norm_method=None,
                                          normalize=False, interp_method=None,
                                          scaled_len=timelocked_scaled_len)
    compress_firing_data(pids, data_path=outpath, outpath=outpath, num_log_bins=NUM_LOG_BINS)

if __name__ == "__main__":
  ####################################################################################################
  #                                           Prepare data                                           #
  ####################################################################################################

  # You can generate these lists of eids/pids whatever way is conveinant.
  # Here we use functions from the "filter_eids" module to manually try and filter all applicable
  # eids in the brainwide map

  '''
  # Download and filter all data for sessions we'll be averaging.
  # Load all eids
  eids = get_bwm_sessions()

  # Save all info files (csvs and dictionary)
  # NOTE: This function downloads > 100Gb of data!
  # If you want to quickly check if a new session contains a file,
  # one.list_datasets(eid) will be more useful. Here downloading makes since because we'll
  # be able to definitively check if we can actually load each dataset, and we'll be using the
  # data later when we're averaging anyway.o
  #filter_eids(eids, EID_INFO_PATH)

  # Load output dictionary
  eid_info_map = np.load(EID_INFO_PATH + "_eid_info.npy", allow_pickle=True).item()

  # Get good eid / pid pairs that have at least timing and wheel info,
  # and spikes / clusters / channels.
  with open(EID_INFO_PATH + "_good_eids.csv", newline="") as good_file:
    eid_pid_pairs = list(csv.reader(good_file))

  eids = np.array(eid_pid_pairs)[:, 0]

  # Get a 2d array of [all_trials, left_corr, left_incorr, right_corr, right_incorr] trial timing
  # dataframes for each session
  trial_timing_dfs, eid_idxs = all_session_event_timings_by_type(eids, include_wheel=True)
  trial_timing_dfs = np.array(trial_timing_dfs, dtype=object)[:, 1:] # exclude df for all trials
  np.save("./data/event_avgs/trial_timing_dfs", trial_timing_dfs)

  # Remove all eid/pids that fail to load their timing dataframe
  eid_pid_pairs = np.array(eid_pid_pairs, dtype=object)[eid_idxs]
  np.save("./data/eid_pid_pairs", eid_pid_pairs)
  '''

  # Load from saved files instead of computing
  trial_timing_dfs = np.load("./data/event_avgs/trial_timing_dfs.npy", allow_pickle=True)

  eid_pid_pairs = np.load("./data/eid_pid_pairs.npy", allow_pickle=True)
  eids = np.array(eid_pid_pairs)[:, 0]
  pids = np.array(eid_pid_pairs)[:, 1]
  print("Num insertions collected: " + str(len(pids)))

  normed_avg_event_timings_by_type = avg_event_timings_by_type(trial_timing_dfs, EVENT_NAMES)
  avg_event_idxs_by_type = np.array(normed_avg_event_timings_by_type * SCALED_LEN, dtype=int)
  print(avg_event_idxs_by_type)

  # Precomputed avg value indices for stimon, first wheel move, and feedback when SCALED_LEN = 250
  #avg_event_idxs_by_type = [[92, 130, 157], # left correct
  #                          [83, 131, 166], # left incorrect
  #                          [91, 131, 158], # right correct
  #                          [84, 131, 165]] # right incorrect

  ####################################################################################################
  #                              Event averaging for multiple data types                             #
  ####################################################################################################

  # Generate event avgs for all datatypes in paralell:
  start = time.perf_counter()
  processes = []
  # Start processes
  for data_type in ["fano_factor"]: #["raw_spikes", "baselined", "baselined_normalized", "fano_factor"]:
    p = multiprocessing.Process(target=event_avg_by_data_type,
                                args=[data_type, pids, trial_timing_dfs, avg_event_idxs_by_type])
    p.start()
    processes.append(p)

  # Stop processes once they finish
  for p in processes:
    p.join()

  end = time.perf_counter()
  print(f'Finished event avgs for all data types in {round(end-start, 2)} second(s)')

  ####################################################################################################
  #                          Event averaging timelocked to multiple events                           #
  ####################################################################################################

  # Avgs timelocked to +- 0.5 seconds for each event in stim on, feedback, first wheel move
  event_window = 0.5
  timelocked_scaled_len = int(event_window * 2 / 0.01) # (divide by 0.01s recording speed)
  timelocked_event_idxs = [[(timelocked_scaled_len // 2)] for i in range(len(trial_timing_dfs))]

  # Generate event avgs for all datatypes in paralell:
  start = time.perf_counter()
  processes = []
  # Start processes
  for event in EVENT_NAMES:
    p = multiprocessing.Process(target=event_avg_by_event, args=[event, pids,
              trial_timing_dfs, timelocked_scaled_len, timelocked_event_idxs])
    p.start()
    processes.append(p)

  # Stop processes once they finish
  for p in processes:
    p.join()

  end = time.perf_counter()
  print(f'Finished event avgs timelocked to all events in {round(end-start, 2)} second(s)')
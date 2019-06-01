import numpy as np

import ibllib.dsp as dsp

WHEEL_RADIUS_CM = 3.1


def _bpod_events_extraction(bpod_t, bpod_fronts):
    """
    From detected fronts on the bpod sync traces, outputs the synchronisation events
    related to trial start and valve opening
    :param bpod_t: numpy vector containing times of fronts
    :param bpod_fronts: numpy vector containing polarity of fronts (1 rise, -1 fall)
    :return: numpy arrays of times t_trial_start and t_valve_open
    """
    # make sure that there are no 2 consecutive fall or consecutive rise events
    assert(np.all(np.abs(np.diff(bpod_fronts)) == 2))
    # make sure that the first event is a rise
    assert(bpod_fronts[0] == 1)
    # take only even time differences: ie. from rising to falling fronts
    dt = np.diff(bpod_t)[::2]
    # detect start trials event assuming length is 0.1 ms except the first trial
    i_trial_start = np.r_[1, np.where(dt <= 1.66e-4)[0] * 2]
    # the first trial we detect the first falling edge to which we subtract 0.1ms
    t_trial_start = bpod_t[i_trial_start]
    t_trial_start[0] -= 1e-4
    # valve open events are all events that are not trial starts (first trials excluded)
    i_valve_open = np.where(np.invert(dt <= 1.66e-4))[0] * 2
    i_valve_open = np.delete(i_valve_open, np.where(i_valve_open < 2))
    t_valve_open = bpod_t[i_valve_open]
    # # some debug plots when needed
    # import matplotlib.pyplot as plt
    # import ibllib.plots as plots
    # plt.figure()
    # plots.squares(bpod_t, bpod_fronts)
    # plots.vertical_lines(t_valve_open, ymin=-0.2, ymax=1.2, linewidth=0.5, color='g')
    # plots.vertical_lines(t_trial_start, ymin=-0.2, ymax=1.2, linewidth=0.5, color='r')
    return t_trial_start, t_valve_open


def _rotary_encoder_positions_from_gray_code(channelA, channelB):
    """
    Extracts the rotary encoder absolute position (cm) as function of time from digital recording
    of the 2 channels.

    Rotary Encoder implements X1 encoding: http://www.ni.com/tutorial/7109/en/
    rising A  & B high = +1
    rising A  & B low = -1
    falling A & B high = -1
    falling A & B low = +1

    :param channelA: Vector of rotary encoder digital recording channel A
    :type channelA: numpy array
    :param channelB: Vector of rotary encoder digital recording channel B
    :type channelB: numpy array
    :return: indices vector and position vector
    """
    # detect rising and falling fronts
    t, fronts = dsp.fronts(channelA)
    # apply X1 logic to get positions in ticks
    p = (channelB[t] * 2 - 1) * fronts
    # convert position in cm
    p = np.cumsum(p) / 1024 * np.pi * WHEEL_RADIUS_CM
    return t, p


def _audio_events_extraction(audio_t, audio_fronts):
    """
    From detected fronts on the audio sync traces, outputs the synchronisation events
    related to tone in

    :param audio_t: numpy vector containing times of fronts
    :param audio_fronts: numpy vector containing polarity of fronts (1 rise, -1 fall)
    :return: numpy arrays t_ready_tone_in, t_error_tone_in
    """
    # make sure that there are no 2 consecutive fall or consecutive rise events
    assert(np.all(np.abs(np.diff(audio_fronts)) == 2))
    # make sure that the first event is a rise
    assert(audio_fronts[0] == 1)
    # take only even time differences: ie. from rising to falling fronts
    dt = np.diff(audio_t)[::2]
    # detect ready tone by length below 110 ms
    i_ready_tone_in = np.r_[1, np.where(dt <= 0.11)[0] * 2]
    t_ready_tone_in = audio_t[i_ready_tone_in]
    # error tones are events lasting from 400ms to 600ms
    i_error_tone_in = np.where(np.logical_and(0.4 < dt, dt < 0.6))[0] * 2
    t_error_tone_in = audio_t[i_error_tone_in]
    return t_ready_tone_in, t_error_tone_in

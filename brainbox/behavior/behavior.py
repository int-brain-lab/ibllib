from ibllib.io.extractors.ephys_fpga import \
    extract_first_movement_times, extract_wheel_moves, MIN_QT
from oneibl.one import ONE

one = ONE()


def load_wheel_reaction_times(eid):
    """
    Return the calculated reaction times for session.  Reaction times are defined as the time
    between the go cue (onset tone) and the onset of the first substantial wheel movement.   A
    movement is considered sufficiently large if its peak amplitude is at least 1/3rd of the
    distance to threshold (~0.1 radians).

    Negative times mean the onset of the movement occurred before the go cue.  Nans may occur if
    there was no detected movement withing the period, or when the goCue_times or feedback_times
    are nan.

    Parameters
    ----------
    eid : str
        Session UUID

    Returns
    ----------
    array-like
        reaction times
    """
    trials = one.load_object(eid, 'trials')
    # If already extracted, load and return
    if trials and 'firstMovement_times' in trials:
        return trials['firstMovement_times'] - trials['goCue_times']
    # Otherwise load wheelMoves object and calculate
    moves = one.load_object(eid, 'wheelMoves')
    # Re-extract wheel moves if necessary
    if not moves or 'peakAmplitude' not in moves:
        wheel = one.load_object(eid, 'wheel')
        wheel = {'re_ts': wheel['timestamps'], 're_pos': wheel['position']}
        moves = extract_wheel_moves(wheel)
    assert trials and moves, 'unable to load trials and wheelMoves data'
    firstMove_times, = extract_first_movement_times(moves, trials, MIN_QT)
    return firstMove_times - trials['goCue_times']

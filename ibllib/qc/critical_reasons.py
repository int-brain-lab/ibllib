'''
Prompt experimenter for reason for marking session/insertion as CRITICAL
Choices are listed in the global variables. Multiple reasons can be selected.
Places info in Alyx session note in a format that is machine retrievable (text->json)
'''
# Author: Gaelle
import json
from oneibl.one import ONE

# Global var

# Reasons for marking a session as critical
REASONS_SESS_CRIT = (
    'within experiment system crash',
    'synching impossible',
    'dud or mock session',
    'essential dataset missing',
    'other'
)

# Reasons for marking an insertion as critical
# Note: Split the reasons labelled in the GUI versus those to be seen
# when marking the insertion programmatically
REASONS_INS_CRIT_GUI = (
    'Noise and artifact',
    'Drift',
    'Poor neural yield',
    'Brain Damage',
    'Other'
)

REASONS_INS_CRIT = tuple(['Histological images missing',
                          'Track not visible on imaging data']
                         + list(REASONS_INS_CRIT_GUI))


# Util functions
def _create_note_str(ins_or_sess):
    '''
    :param ins_or_sess: str containing either 'insertion' or 'session'
    :return:
    '''
    str_static = f'=== EXPERIMENTER REASON(S) FOR MARKING THE ' \
                 f'{ins_or_sess.upper()} AS CRITICAL ==='
    return str_static


def _reason_addnumberstr(reason_list):
    """
    Adding number at the beginning of the str for ease of selection by user
    :param reason_list : list of str ; default: None
    """
    return [f'{i}) {r}' for i, r in enumerate(reason_list)]


REASONS_SESS_WITH_NUMBERS = _reason_addnumberstr(reason_list=REASONS_SESS_CRIT)
REASONS_INS_WITH_NUMBERS = _reason_addnumberstr(reason_list=REASONS_INS_CRIT)


def _reason_question_prompt(reason_list, reasons_with_numbers, ins_or_sess):
    """
    Function asking the user to enter the criteria for marking a session/insertion as CRITICAL.
    :param reason_list: list of reasons (str)
    :param reasons_with_numbers: list of reasons (str), with added number in front
    :param ins_or_sess: str containing either 'insertion' or 'session'
    """

    prompt = f'Select from this list the reason(s) why you are marking the ' \
             f'{ins_or_sess} as CRITICAL:' \
             f' \n {reasons_with_numbers} \n' \
             f'and enter the corresponding numbers separated by commas, e.g. 1,3 -> enter: '
    ans = input(prompt).strip().lower()
    # turn str into numbers
    string_list = ans.split(',')
    try:  # try-except inc ase users enters something else than a number
        integer_map = map(int, string_list)
        integer_list = list(integer_map)
    except ValueError:
        print(f'{ans} is invalid, please try again...')
        return _reason_question_prompt()

    if all(elem in range(0, len(reasons_with_numbers)) for elem in integer_list):
        reasons_out = [reason_list[integer_n] for integer_n in integer_list]
        print(f'You selected reason(s): {reasons_out}')
        return reasons_out
    else:
        print(f'{ans} is invalid, please try again...')
        return _reason_question_prompt()


def _enquire_why_other():
    prompt = 'Explain why you selected "other" (free text): '
    ans = input(prompt).strip().lower()
    return ans


def _create_note_json(reasons_selected, reason_for_other, note_title):
    note_session = {
        "title": note_title,
        "reasons_selected": reasons_selected,
        "reason_for_other": reason_for_other
    }
    return json.dumps(note_session)


def _delete_note_yesno(notes):
    """
    Function asking user whether notes are to be deleted.
    :param notes: Alyx notes, from ONE query
    :return: y / n (string)
    """
    prompt = f'You are about to delete {len(notes)} existing notes; ' \
             f'do you want to proceed? y/n: '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return _delete_note_yesno()
    else:
        return ans


def _upload_note_alyx(eid, note_text, content_type, str_notes_static, one=None, overwrite=False):
    """
    Function to upload a note to Alyx.
    It will check if notes with STR_NOTES_STATIC already exists for this session,
    and ask if OK to overwrite.
    :param eid: session or isnertion eid
    :param note_text: text to enter within the note object
    :param one: default: None -> ONE()
    :param str_notes_static: string within the notes that will be searched for
    :param content_type: 'session' or 'insertion'
    :param overwrite: if set to False, will check whether other notes exists and ask
    if deleting is OK.
    If set to True, will delete any previous note without asking.
    :return:
    """
    if one is None:
        one = ONE()
    my_note = {'user': one._par.ALYX_LOGIN,
               'content_type': content_type,
               'object_id': eid,
               'text': f'{note_text}'}
    # check if such a note already exists, ask if OK to overwrite
    notes = one.alyx.rest('notes', 'list',
                          django=f'text__icontains,{str_notes_static},'
                                 f'object_id,{eid}')
    if len(notes) == 0:
        one.alyx.rest('notes', 'create', data=my_note)
        print('The selected reasons were saved on Alyx.')
    else:
        if overwrite:
            ans = 'y'
        else:
            ans = _delete_note_yesno(notes=notes)
        if ans == 'y':
            for note in notes:
                one.alyx.rest('notes', 'delete', id=note['id'])
            one.alyx.rest('notes', 'create', data=my_note)
            print('The selected reasons were saved on Alyx ; old notes were deleted')
        else:
            print('The selected reasons were NOT saved on Alyx ; old notes remain.')


def main_gui(eid, reasons_selected, one=None):
    """
    Main function to call to input a reason for marking an insertion as
    CRITICAL from the alignment GUI. It will:
    - create note text, after deleting any similar notes existing already

    :param: eid: insertion id
    :param: reasons_selected: list of str, str are picked within REASONS_INS_CRIT_GUI
    """
    # hit the database to check if eid is insertion eid
    ins_list = one.alyx.rest('insertions', 'list', id=eid)
    if len(ins_list) != 1:
        raise ValueError(f'N={len(ins_list)} insertion found, expected N=1. Check eid provided.')

    # assert that reasons are all within REASONS_INS_CRIT_GUI
    for item_str in reasons_selected:
        assert item_str in REASONS_INS_CRIT_GUI

    # create note title and text
    note_title = _create_note_str('insertion')

    note_text = _create_note_json(reasons_selected=reasons_selected,
                                  reason_for_other=[],
                                  note_title=note_title)

    # upload note to Alyx
    _upload_note_alyx(eid, note_text, content_type='probeinsertion',
                      str_notes_static=note_title, one=one, overwrite=True)


def main(eid, one=None):
    """
    Main function to call to input a reason for marking a session/insertion
    as CRITICAL programmatically. It will:
    - ask reasons for selection of critical status
    - check if 'other' reason has been selected, inquire why (free text)
    - create note text, checking whether similar notes exist already
    - upload note to Alyx if none exist previously or if overwrite is chosen
    Q&A are prompted via the Python terminal.

    Example:
    # Retrieve Alyx note to test
    one = ONE(base_url='https://dev.alyx.internationalbrainlab.org')
    eid = '2ffd3ed5-477e-4153-9af7-7fdad3c6946b'
    main(eid=eid, one=one)

    # Get notes with pattern
    notes = one.alyx.rest('notes', 'list',
                          django=f'text__icontains,{STR_NOTES_STATIC},'
                                 f'object_id,{eid}')
    test_json_read = json.loads(notes[0]['text'])

    :param eid: session/insertion eid
    :param one: default: None -> ONE()
    :return:
    """
    if one is None:
        one = ONE()
    # ask reasons for selection of critical status

    # hit the database to know if eid is insertion or session eid

    sess_list = one.alyx.rest('sessions', 'list', id=eid)
    ins_list = one.alyx.rest('insertions', 'list', id=eid)

    if len(sess_list) > 0 and len(ins_list) == 0:  # session
        reason_list = REASONS_SESS_CRIT
        reasons_with_numbers = REASONS_SESS_WITH_NUMBERS
        ins_or_sess = 'session'
        content_type = 'session'
    elif len(ins_list) > 0 and len(sess_list) == 0:  # insertion
        reason_list = REASONS_INS_CRIT
        reasons_with_numbers = REASONS_INS_WITH_NUMBERS
        content_type = 'probeinsertion'
        ins_or_sess = 'insertion'
    else:
        raise ValueError(f'Inadequate number of session (n={len(sess_list)}) '
                         f'or insertion (n={len(ins_list)}) found for eid {eid}.'
                         f'The query output should be of length 1.')

    reasons_selected = _reason_question_prompt(reason_list=reason_list,
                                               reasons_with_numbers=reasons_with_numbers,
                                               ins_or_sess=ins_or_sess)

    # check if 'other' reason has been selected, inquire why
    if 'other' in reasons_selected:
        reason_for_other = _enquire_why_other()
    else:
        reason_for_other = []

    # create note title and text
    note_title = _create_note_str(ins_or_sess)

    note_text = _create_note_json(reasons_selected=reasons_selected,
                                  reason_for_other=reason_for_other,
                                  note_title=note_title)

    # upload note to Alyx
    _upload_note_alyx(eid, note_text, content_type=content_type,
                      str_notes_static=note_title, one=one)

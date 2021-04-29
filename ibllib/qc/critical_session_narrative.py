"""
Prompt experimenter for reason for marking session as CRITICAL
Choices are:
- within experiment system crash
- syncing impossible
- dud or mock session
- essential dataset missing
- other

Multiple reasons can be selected
Place info in Alyx session note in a format that is machine retrievable (text->json)
"""
# Author: Gaelle
import json
from oneibl.one import ONE

# Global var
STR_NOTES_STATIC = '=== EXPERIMENTER REASON(S) FOR MARKING THE SESSION AS CRITICAL ==='

REASONS_SESS_CRIT = (
    'within experiment system crash',
    'synching impossible',
    'dud or mock session',
    'essential dataset missing',
    'other'
)


# Util functions
def _reason_addnumberstr(reason_list=REASONS_SESS_CRIT):
    """
    Adding number at the beginning of the str for ease of selection by user
    :param reason_list : list of str ; default: REASONS_SESS_CRIT
    """
    return [f'{i}) {r}' for i, r in enumerate(reason_list)]


REASONS_WITH_NUMBERS = _reason_addnumberstr()


def _reason_question_prompt():
    """
    Function asking the user to enter the criteria for marking a sess as CRITICAL.
    """

    prompt = f'Select from this list the reason(s) why you are marking the session as CRITICAL:' \
             f' \n {REASONS_WITH_NUMBERS} \n' \
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

    if all(elem in range(0, len(REASONS_WITH_NUMBERS)) for elem in integer_list):
        reasons_out = [REASONS_SESS_CRIT[integer_n] for integer_n in integer_list]
        print(f'You selected reason(s): {reasons_out}')
        return reasons_out
    else:
        print(f'{ans} is invalid, please try again...')
        return _reason_question_prompt()


def _enquire_why_other():
    prompt = 'Explain why you selected "other" (free text): '
    ans = input(prompt).strip().lower()
    return ans


def _create_note_json(reasons_selected, reason_for_other, note_title=STR_NOTES_STATIC):
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


def _upload_note_alyx(eid, note_text, one=None, str_notes_static=STR_NOTES_STATIC):
    """
    Function to upload a note to Alyx.
    It will check if notes with STR_NOTES_STATIC already exists for this session,
    and ask if OK to overwrite.
    :param eid: session eid
    :param note_text: text to enter within the note object
    :param one: default: None -> ONE()
    :param str_notes_static: string within the notes that will be searched for
    :return:
    """
    if one is None:
        one = ONE()
    my_note = {'user': one._par.ALYX_LOGIN,
               'content_type': 'session',
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
        ans = _delete_note_yesno(notes=notes)
        if ans == 'y':
            for note in notes:
                one.alyx.rest('notes', 'delete', id=note['id'])
            one.alyx.rest('notes', 'create', data=my_note)
            print('The selected reasons were saved on Alyx; old notes were deleted')
        else:
            print('The selected reasons were NOT saved in Alyx ; old notes remain.')


def main(eid, one=None):
    """
    Main function to call to input a reason for marking a session as CRITICAL. It will:
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

    :param eid: session eid
    :param one: default: None -> ONE()
    :return:
    """
    if one is None:
        one = ONE()
    # ask reasons for selection of critical status
    reasons_selected = _reason_question_prompt()

    # check if 'other' reason has been selected, inquire why
    if 'other' in reasons_selected:
        reason_for_other = _enquire_why_other()
    else:
        reason_for_other = []

    # create note text
    note_text = _create_note_json(reasons_selected=reasons_selected,
                                  reason_for_other=reason_for_other)

    # upload note to Alyx
    _upload_note_alyx(eid, note_text, one=one)

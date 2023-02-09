"""
Prompt experimenter for reason for marking session/insertion as CRITICAL
Choices are listed in the global variables. Multiple reasons can be selected.
Places info in Alyx session note in a format that is machine retrievable (text->json)
"""
# Author: Gaelle
import abc
import logging
import json
from requests.exceptions import HTTPError
from datetime import datetime
from one.api import ONE

_logger = logging.getLogger("ibllib")


def main_gui(uuid, reasons_selected, one=None):
    """
    Main function to call to input a reason for marking an insertion as
    CRITICAL from the alignment GUI. It will:
    - create note text, after deleting any similar notes existing already

    :param: uuid: insertion id
    :param: reasons_selected: list of str, str are picked within REASONS_INS_CRIT_GUI
    """
    # hit the database to check if uuid is insertion uuid
    ins_list = one.alyx.rest('insertions', 'list', id=uuid, no_cache=True)
    if len(ins_list) != 1:
        raise ValueError(f'N={len(ins_list)} insertion found, expected N=1. Check uuid provided.')

    note = CriticalInsertionNote(uuid, one)

    # assert that reasons are all within REASONS_INS_CRIT_GUI
    for item_str in reasons_selected:
        assert item_str in note.descriptions_gui

    note.selected_reasons = reasons_selected
    note.other_reason = []
    note._upload_note(overwrite=True)


def main(uuid, one=None):
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
    uuid = '2ffd3ed5-477e-4153-9af7-7fdad3c6946b'
    main(uuid=uuid, one=one)

    # Get notes with pattern
    notes = one.alyx.rest('notes', 'list',
                          django=f'text__icontains,{STR_NOTES_STATIC},'
                                 f'object_id,{uuid}')
    test_json_read = json.loads(notes[0]['text'])

    :param uuid: session/insertion uuid
    :param one: default: None -> ONE()
    :return:
    """
    if one is None:
        one = ONE()
    # ask reasons for selection of critical status

    # hit the database to know if uuid is insertion or session uuid
    sess_list = one.alyx.get('/sessions?&django=pk,' + uuid, clobber=True)
    ins_list = one.alyx.get('/insertions?&django=pk,' + uuid, clobber=True)

    if len(sess_list) > 0 and len(ins_list) == 0:  # session
        note = CriticalSessionNote(uuid, one)
    elif len(ins_list) > 0 and len(sess_list) == 0:  # insertion
        note = CriticalInsertionNote(uuid, one)
    else:
        raise ValueError(f'Inadequate number of session (n={len(sess_list)}) '
                         f'or insertion (n={len(ins_list)}) found for uuid {uuid}.'
                         f'The query output should be of length 1.')

    note.upload_note()


class Note(abc.ABC):
    descriptions = []

    @property
    def default_descriptions(self):
        return self.descriptions + ['Other']

    @property
    def extra_prompt(self):
        return ''

    @property
    def note_title(self):
        return ''

    @property
    def n_description(self):
        return len(self.default_descriptions)

    def __init__(self, uuid, one, content_type=None):
        """
        Base class for attaching notes to an alyx endpoint. Do not use this class directly but use parent classes that inherit
        this base class

        :param uuid: uuid of session/ insertion or other model to attach note to
        :param one: ONE instance
        :param content_type: alyx endpoint of uuid
        """
        self.uuid = uuid
        self.one = one
        self.selected_reasons = []
        self.other_reason = []
        if content_type is not None:
            self.content_type = content_type
        else:
            self.content_type = self.get_content_type()

    def get_content_type(self):
        """
        Infer the content_type from the uuid. Only checks to see if uuid is a session or insertion. If not recognised will raise
        an error and the content_type must be specified on note initialisation e.g Note(uuid, one, content_type='subject')
        :return:
        """

        # see if it as session or an insertion
        if self.one.uuid2path(self.uuid):
            content_type = 'session'
        else:
            try:
                _ = self.one.pid2uuid(self.uuid)
                content_type = 'probeinsertion'
            except HTTPError:
                raise ValueError('Content type cannot be recognised from {uuid}. Specify on '
                                 'initialistion e.g Note(uuid, one, content_type="subject"')

        return content_type

    def describe(self):
        """
        Print list of default reasons that can be chosen from
        :return:
        """
        for i, d in enumerate(self.descriptions):
            print(f'{i}. {d} \n')

    def numbered_descriptions(self):
        """
        Return list of numbered default reasons
        :return:
        """
        return [f'{i}) {d}' for i, d in enumerate(self.default_descriptions)]

    def upload_note(self, nums=None, other_reason=None, **kwargs):
        """
        Upload note to alyx. If no values for nums and other_reason are specified, user will receive a prompt in command line
        asking them to choose from default list of reasons to add to note as well as option for free text. To upload without
        receiving prompt a value for either nums or other_reason must be given

        :param nums: string of numbers matching those in default descrptions, e.g, '1,3'. Options can be see using note.describe()
        :param other_reason: other comment or reasons to add to note (string)
        :param kwargs:
        :return:
        """

        if nums is None and other_reason is None:
            self.selected_reasons, self.other_reason = self.reasons_prompt()
        else:
            self.selected_reasons = self._map_num_to_description(nums)
            self.other_reason = other_reason or []

        self._upload_note(**kwargs)

    def _upload_note(self, **kwargs):
        existing_note, notes = self._check_existing_note()
        if existing_note:
            self.update_existing_note(notes, **kwargs)
        else:
            text = self.format_note(**kwargs)
            self._create_note(text)
            _logger.info('The selected reasons were saved on Alyx.')

    def _create_note(self, text):

        data = {'user': self.one.alyx.user,
                'content_type': self.content_type,
                'object_id': self.uuid,
                'text': f'{text}'}
        self.one.alyx.rest('notes', 'create', data=data)

    def _update_note(self, note_id, text):

        data = {'user': self.one.alyx.user,
                'content_type': self.content_type,
                'object_id': self.uuid,
                'text': f'{text}'}
        self.one.alyx.rest('notes', 'partial_update', id=note_id, data=data)

    def _delete_note(self, note_id):
        self.one.alyx.rest('notes', 'delete', id=note_id)

    def _delete_notes(self, notes):
        for note in notes:
            self._delete_note(note['id'])

    def _check_existing_note(self):
        notes = self.one.alyx.rest('notes', 'list', django=f'text__icontains,{self.note_title},object_id,{self.uuid}',
                                   no_cache=True)
        if len(notes) == 0:
            return False, None
        else:
            return True, notes

    def _map_num_to_description(self, nums):

        if nums is None:
            return []

        string_list = nums.split(',')
        int_list = list(map(int, string_list))

        if max(int_list) >= self.n_description or min(int_list) < 0:
            raise ValueError(f'Chosen values out of range, must be between 0 and {self.n_description - 1}')

        return [self.default_descriptions[n] for n in int_list]

    def reasons_prompt(self):
        """
        Prompt for user to enter reasons
        :return:
        """

        prompt = f'{self.extra_prompt} ' \
                 f'\n {self.numbered_descriptions()} \n ' \
                 f'and enter the corresponding numbers separated by commas, e.g. 1,3 -> enter: '

        ans = input(prompt).strip().lower()

        try:
            selected_reasons = self._map_num_to_description(ans)
            print(f'You selected reason(s): {selected_reasons}')
            if 'Other' in selected_reasons:
                other_reasons = self.other_reason_prompt()
                return selected_reasons, other_reasons
            else:
                return selected_reasons, []

        except ValueError:
            print(f'{ans} is invalid, please try again...')
            return self.reasons_prompt()

    def other_reason_prompt(self):
        """
        Prompt for user to enter other reasons
        :return:
        """

        prompt = 'Explain why you selected "other" (free text): '
        ans = input(prompt).strip().lower()
        return ans

    @abc.abstractmethod
    def format_note(self, **kwargs):
        """
        Method to format text field of note according to type of note wanting to be uploaded
        :param kwargs:
        :return:
        """

    @abc.abstractmethod
    def update_existing_note(self, note, **kwargs):
        """
        Method to specify behavior in the case of a note with the same title already existing
        :param note:
        :param kwargs:
        :return:
        """


class CriticalNote(Note):
    """
    Class for uploading a critical note to a session or insertion. Do not use directly but use CriticalSessionNote or
    CriticalInsertionNote instead
    """

    def format_note(self, **kwargs):
        note_text = {
            "title": self.note_title,
            "reasons_selected": self.selected_reasons,
            "reason_for_other": self.other_reason
        }
        return json.dumps(note_text)

    def update_existing_note(self, notes, **kwargs):

        overwrite = kwargs.get('overwrite', None)
        if overwrite is None:
            overwrite = self.delete_note_prompt(notes)

        if overwrite:
            self._delete_notes(notes)
            text = self.format_note()
            self._create_note(text)
            _logger.info('The selected reasons were saved on Alyx; old notes were deleted')
        else:
            _logger.info('The selected reasons were NOT saved on Alyx; old notes remain.')

    def delete_note_prompt(self, notes):

        prompt = f'You are about to delete {len(notes)} existing notes; ' \
                 f'do you want to proceed? y/n: '

        ans = input(prompt).strip().lower()

        if ans not in ['y', 'n']:
            print(f'{ans} is invalid, please try again...')
            return self.delete_note_prompt(notes)
        else:
            return True if ans == 'y' else False


class CriticalInsertionNote(CriticalNote):
    """
    Class for uploading a critical note to an insertion.

    Example
    -------
    note = CriticalInsertionNote(pid, one)
    # print list of default reasons
    note.describe()
    # to receive a command line prompt to fill in note
    note.upload_note()
    # to upload note automatically without prompt
    note.upload_note(nums='1,4', other_reason='lots of bad channels')
    """

    descriptions_gui = [
        'Noise and artifact',
        'Drift',
        'Poor neural yield',
        'Brain Damage'
        'Other'
    ]

    descriptions = [
        'Histological images missing',
        'Track not visible on imaging data'
    ]

    @property
    def default_descriptions(self):
        return self.descriptions + self.descriptions_gui

    @property
    def extra_prompt(self):
        return 'Select from this list the reason(s) why you are marking the insertion as CRITICAL:'

    @property
    def note_title(self):
        return '=== EXPERIMENTER REASON(S) FOR MARKING THE INSERTION AS CRITICAL ==='

    def __init__(self, uuid, one):
        super(CriticalInsertionNote, self).__init__(uuid, one, content_type='probeinsertion')


class CriticalSessionNote(CriticalNote):
    """
    Class for uploading a critical note to a session.

    Example
    -------
    note = CriticalInsertionNote(uuid, one)
    # print list of default reasons
    note.describe()
    # to receive a command line prompt to fill in note
    note.upload_note()
    # to upload note automatically without prompt
    note.upload_note(nums='1,4', other_reason='session with no ephys recording')
    """

    descriptions = [
        'within experiment system crash',
        'synching impossible',
        'dud or mock session',
        'essential dataset missing',
    ]

    @property
    def extra_prompt(self):
        return 'Select from this list the reason(s) why you are marking the session as CRITICAL:'

    @property
    def note_title(self):
        return '=== EXPERIMENTER REASON(S) FOR MARKING THE SESSION AS CRITICAL ==='

    def __init__(self, uuid, one):
        super(CriticalSessionNote, self).__init__(uuid, one, content_type='session')


class SignOffNote(Note):
    """
    Class for signing off a session and optionally adding a related explanation note.
    Do not use directly but use classes that inherit from this class e.g TaskSignOffNote, RawEphysSignOffNote
    """

    @property
    def extra_prompt(self):
        return 'Select from this list the reason(s) that describe issues with this session:'

    @property
    def note_title(self):
        return f'=== SIGN-OFF NOTE FOR {self.sign_off_key} ==='

    def __init__(self, uuid, one, sign_off_key):
        self.sign_off_key = sign_off_key
        super(SignOffNote, self).__init__(uuid, one, content_type='session')
        self.datetime_key = self.get_datetime_key()
        self.session = one.alyx.rest('sessions', 'read', id=self.uuid, no_cache=True)

    def upload_note(self, nums=None, other_reason=None, **kwargs):
        super(SignOffNote, self).upload_note(nums=nums, other_reason=other_reason, **kwargs)
        self.sign_off()

    def sign_off(self):

        json = self.session['json']
        json['sign_off_checklist'][self.sign_off_key] = {'date': self.datetime_key.split('_')[0],
                                                         'user': self.datetime_key.split('_')[1]}

        self.one.alyx.json_field_update("sessions", self.uuid, 'json', data=json)

    def format_note(self, **kwargs):

        note_text = {
            "title": self.note_title,
            f'{self.datetime_key}': {"reasons_selected": self.selected_reasons,
                                     "reason_for_other": self.other_reason}
        }

        return json.dumps(note_text)

    def format_existing_note(self, orignal_note):

        extra_note = {f'{self.datetime_key}': {"reasons_selected": self.selected_reasons,
                                               "reason_for_other": self.other_reason}
                      }

        orignal_note.update(extra_note)

        return json.dumps(orignal_note)

    def update_existing_note(self, notes):
        if len(notes) != 1:
            raise ValueError(f'{len(notes)} with same title found, only expect at most 1. Clean up before proceeding')
        else:
            original_note = json.loads(notes[0]['text'])
            text = self.format_existing_note(original_note)
            self._update_note(notes[0]['id'], text)

    def get_datetime_key(self):
        user = self.one.alyx.user
        date = datetime.now().date().isoformat()
        return date + '_' + user


class TaskSignOffNote(SignOffNote):

    """
    Class for signing off a task part of a session and optionally adding a related explanation note.

    Example
    -------
    note = TaskSignOffNote(uuid, one, '_ephysChoiceWorld_00')
    # to sign off session without any note
    note.sign_off()
    # print list of default reasons
    note.describe()
    # to upload note and sign off with prompt
    note.upload_note()
    # to upload note automatically without prompt
    note.upload_note(nums='1,4', other_reason='session with no ephys recording')
    """

    descriptions = [
        'raw trial data does not exist',
        'wheel data corrupt',
        'task data could not be synced',
    ]


class PassiveSignOffNote(SignOffNote):

    """
    Class for signing off a passive part of a session and optionally adding a related explanation note.

    Example
    -------
    note = PassiveSignOffNote(uuid, one, '_passiveChoiceWorld_00')
    # to sign off session without any note
    note.sign_off()
    # print list of default reasons
    note.describe()
    # to upload note and sign off with prompt
    note.upload_note()
    # to upload note automatically without prompt
    note.upload_note(nums='1,4', other_reason='session with no ephys recording')
    """

    descriptions = [
        'Raw passive data doesn’t exist (no. of spacers = 0)',
        'Incorrect number or spacers (i.e passive cutoff midway)',
        'RFmap file doesn’t exist',
        'Gabor patches couldn’t be extracted',
        'Trial playback couldn’t be extracted',
    ]


class VideoSignOffNote(SignOffNote):

    """
    Class for signing off a video part of a session and optionally adding a related explanation note.

    Example
    -------
    note = VideoSignOffNote(uuid, one, '_camera_left')
    # to sign off session without any note
    note.sign_off()
    # print list of default reasons
    note.describe()
    # to upload note and sign off with prompt
    note.upload_note()
    # to upload note automatically without prompt
    note.upload_note(nums='1,4', other_reason='session with no ephys recording')
    """

    descriptions = [
        'The video timestamps are not the same length as the video file (either empty or slightly longer/shorter)',
        'The rotary encoder trace doesn’t not appear synced with the video',
        'The QC fails because the GPIO file is missing or empty',
        'The frame rate in the video header is wrong (the video plays too slow or fast)',
        'The resolution is not what is defined in the experiment description file',
        'The DLC QC fails because something is obscuring the pupil',
    ]


class RawEphysSignOffNote(SignOffNote):

    """
    Class for signing off a raw ephys part of a session and optionally adding a related explanation note.

    Example
    -------
    note = RawEphysSignOffNote(uuid, one, '_neuropixel_raw_probe00')
    # to sign off session without any note
    note.sign_off()
    # print list of default reasons
    note.describe()
    # to upload note and sign off with prompt
    note.upload_note()
    # to upload note automatically without prompt
    note.upload_note(nums='1,4', other_reason='session with no ephys recording')
    """

    descriptions = [
        'Data has striping',
        'Horizontal band',
        'Discontunuity',
    ]


class SpikeSortingSignOffNote(SignOffNote):

    """
    Class for signing off a spikesorting part of a session and optionally adding a related explanation note.

    Example
    -------
    note = SpikeSortingSignOffNote(uuid, one, '_neuropixel_spike_sorting_probe00')
    # to sign off session without any note
    note.sign_off()
    # print list of default reasons
    note.describe()
    # to upload note and sign off with prompt
    note.upload_note()
    # to upload note automatically without prompt
    note.upload_note(nums='1,4', other_reason='session with no ephys recording')
    """

    descriptions = [
        'Spikesorting could not be run',
        'Poor quality spikesorting',
    ]


class AlignmentSignOffNote(SignOffNote):

    """
    Class for signing off a alignment part of a session and optionally adding a related explanation note.

    Example
    -------
    note = AlignmentSignOffNote(uuid, one, '_neuropixel_alignment_probe00')
    # to sign off session without any note
    note.sign_off()
    # print list of default reasons
    note.describe()
    # to upload note and sign off with prompt
    note.upload_note()
    # to upload note automatically without prompt
    note.upload_note(nums='1,4', other_reason='session with no ephys recording')
    """

    descriptions = []

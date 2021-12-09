import logging
import requests
import traceback
import json
import abc

from one.api import ONE
from ibllib.pipes import tasks
from ibllib.misc import version
_logger = logging.getLogger('ibllib')


class ReportSnapshot(tasks.Task):

    def __init__(self, session_path, object_id, content_type='session', **kwargs):
        self.object_id = object_id
        self.content_type = content_type
        self.images = []
        super(ReportSnapshot, self).__init__(session_path, **kwargs)

    def _run(self, overwrite=False):
        # Can be used to generate the image if desired
        pass

    def register_images(self, widths=None, function=None):
        report_tag = '## report ##'
        snapshot = Snapshot(one=self.one, object_id=self.object_id, content_type=self.content_type)
        jsons = []
        texts = []
        for f in self.outputs:
            jsons.append(dict(tag=report_tag, version=version.ibllib(),
                              function=(function or str(self.__class__).split("'")[1]), name=f.stem))
            texts.append(f"{f.stem}")
        return snapshot.register_images(self.outputs, jsons=jsons, texts=texts, widths=widths)


class ReportSnapshotProbe(ReportSnapshot):
    signature = {
        'input_files': [],  # see setUp method for declaration of inputs
        'output_files': []  # see setUp method for declaration of inputs
    }

    def __init__(self, pid, one=None, brain_regions=None, **kwargs):
        """
        :param pid: probe insertion UUID from Alyx
        :param one: one instance
        :param brain_regions: (optional) ibllib.atlas.BrainRegion object
        :param kwargs:
        """
        assert one
        self.one = one
        self.brain_regions = brain_regions
        self.content_type = 'probeinsertion'
        self.pid = pid
        self.eid, self.pname = self.one.pid2eid(self.pid)
        self.session_path = self.one.eid2path(self.eid)
        self.output_directory = self.session_path.joinpath('snapshot', self.pname)
        self.get_probe_signature()
        super(ReportSnapshotProbe, self).__init__(self.session_path, object_id=pid, content_type=self.content_type, **kwargs)

    @property
    def pid_label(self):
        """returns a probe insertion stub to label titles, for example: 'SWC_054_2020-10-05_001_probe01'"""
        return '_'.join(list(self.session_path.parts[-3:]) + [self.pname])

    @abc.abstractmethod
    def get_probe_signature(self):
        # method that gets input and output signatures from the probe name. The format is a dictionary as follows:
        # return {'input_files': input_signature, 'output_files': output_signature}
        pass


class Snapshot:
    """
    A class to register images in form of Notes, linked to an object on Alyx.

    :param object_id: The id of the object the image should be linked to
    :param content_type: Which type of object to link to, e.g. 'session', 'probeinsertion', 'subject',
    default is 'session'
    :param one: An ONE instance, if None is given it will be instantiated.
    """

    def __init__(self, object_id, content_type='session', one=None):
        self.one = one or ONE()
        self.object_id = object_id
        self.content_type = content_type
        self.images = []

    def plot(self):
        """
        Placeholder method to be overriden by child object
        :return:
        """
        pass

    def generate_image(self, plt_func, plt_kwargs):
        """
        Takes a plotting function and adds the output to the Snapshot.images list for registration

        :param plt_func: A plotting function that returns the path to an image.
        :param plt_kwargs: Dictionary with keyword arguments for the plotting function
        """
        img_path = plt_func(**plt_kwargs)
        if isinstance(img_path, list):
            self.images.extend(img_path)
        else:
            self.images.append(img_path)
        return img_path

    def register_image(self, image_file, text='', json_field=None, width=None):
        """
        Registers an image as a Note, attached to the object specified by Snapshot.object_id

        :param image_file: Path to the image to to registered
        :param text: str, text to describe the image, defaults ot empty string
        :param json_field: dict, to be added to the json field of the Note
        :param width: width to scale the image to, defaults to None (scale to UPLOADED_IMAGE_WIDTH in alyx.settings.py),
        other options are 'orig' (don't change size) or any integer (scale to width=int, aspect ratios won't be changed)

        :returns: dict, note as registered in database
        """
        # the protocol is not compatible with byte streaming and json, so serialize the json object here
        note = {
            'user': self.one.alyx.user, 'content_type': self.content_type, 'object_id': self.object_id,
            'text': text, 'width': width, 'json': json.dumps(json_field)}
        _logger.info(f'Registering image to {self.content_type} with id {self.object_id}')
        # to make sure an eventual note gets deleted with the image call the delete REST endpoint first
        current_note = self.one.alyx.rest('notes', 'list',
                                          django=f"object_id,{self.object_id},text,{text},json__name,{text}",
                                          no_cache=True)
        if len(current_note) == 1:
            self.one.alyx.rest('notes', 'delete', id=current_note[0]['id'])
        # Open image for upload
        fig_open = open(image_file, 'rb')
        # Catch error that results from object_id - content_type mismatch
        try:
            note_db = self.one.alyx.rest('notes', 'create', data=note, files={'image': fig_open})
            fig_open.close()
            return note_db
        except requests.HTTPError as e:
            if "matching query does not exist.'" in str(e):
                fig_open.close()
                _logger.error(f'The object_id {self.object_id} does not match an object of type {self.content_type}')
                _logger.debug(traceback.format_exc())
            else:
                fig_open.close()
                raise

    def register_images(self, image_list=None, texts=None, widths=None, jsons=None):
        """
        Registers a list of images as Notes, attached to the object specified by Snapshot.object_id.
        The images can be passed as image_list. If None are passed, will try to register the images in Snapshot.images.

        :param image_list: List of paths to the images to to registered. If None, will try to register any images in
                           Snapshot.images
        :param texts: List of text to describe the images. If len(texts)==1, the same text will be used for all images
        :param widths: List of width to scale the figure to (see Snapshot.register_image). If len(widths)==1,
                       the same width will be used for all images
        :param jsons: List of dictionaries to populate the json field of the note in Alyx. If len(jsons)==1,
                       the same dict will be used for all images
        :returns: list of dicts, notes as registered in database
        """
        if not image_list or len(image_list) == 0:
            if len(self.images) == 0:
                _logger.warning(
                    "No figures were passed to register_figures, and self.figures is empty. No figures to register")
                return
            else:
                image_list = self.images
        widths = widths or [None]
        texts = texts or ['']
        jsons = jsons or [None]

        if len(texts) == 1:
            texts = len(image_list) * texts
        if len(widths) == 1:
            widths = len(image_list) * widths
        if len(jsons) == 1:
            jsons = len(image_list) * jsons
        note_dbs = []
        for figure, text, width, json_field in zip(image_list, texts, widths, jsons):
            note_dbs.append(self.register_image(figure, text=text, width=width, json_field=json_field))
        return note_dbs

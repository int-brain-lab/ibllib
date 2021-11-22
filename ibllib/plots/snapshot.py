import logging
import requests
import traceback

from one.api import ONE

_logger = logging.getLogger('ibllib')


class Snapshot:
    """
    A class to register images in form of Notes, linked to an object on alyx.

    :param object_id: The id of the object the image should be linked to
    :param content_type: Which type of object to link to, e.g. 'session', 'probeinsertions', default is 'session'
    """

    def __init__(self, object_id, content_type='session', one=None):
        self.one = one or ONE()
        self.object_id = object_id
        self.content_type = content_type
        self.figures = []

    def generate_figure(self, plt_func, plt_kwargs):
        fig_path = plt_func(**plt_kwargs)
        self.figures.append(fig_path)
        return fig_path

    def register_figure(self, figure, text='', width=None):
        fig_open = open(figure, 'rb')
        note = {
            'user': self.one.alyx.user, 'content_type': self.content_type, 'object_id': self.object_id,
            'text': text, 'width': width}
        _logger.info(f'Registering image to {self.content_type} with id {self.object_id}')
        # Catch error that results from object_id - content_type mismatch
        try:
            self.one.alyx.rest('notes', 'create', data=note, files={'image': fig_open})
        except requests.HTTPError as e:
            if "matching query does not exist.'" in str(e):
                _logger.error(f'The object_id {self.object_id} does not match an object of type {self.content_type}')
                _logger.debug(traceback.format_exc())
            else:
                raise

    def register_figures(self, figures=None, texts=[''], widths=[None]):
        if not figures or len(figures) == 0:
            if len(self.figures) == 0:
                _logger.warning(
                    "No figures were passed to register_figures, and self.figures is empty. No figures to register")
                return
            else:
                figures = self.figures
        if len(texts) == 1:
            texts = len(figures) * texts
        if len(widths) == 1:
            widths = len(figures) * widths
        for figure, text, width in zip(figures, texts, widths):
            self.register_figure(figure, text=text, width=width)

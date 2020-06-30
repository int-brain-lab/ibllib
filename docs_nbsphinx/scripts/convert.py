import os
import json
import logging
import numpy as np

logger = logging.getLogger('ibllib')

IPYTHON_VERSION = 4


class NotebookConverter(object):

    def __init__(self, nb_path, colab_template=None, overwrite=True):

        self.nb_path = os.path.abspath(nb_path)
        fn = os.path.basename(self.nb_path)
        self.nb_dir = os.path.dirname(self.nb_path)
        self.nb_name, _ = os.path.splitext(fn)

        if colab_template is not None:
            self.colab_template = os.path.abspath(colab_template)
        else:
            self.colab_template = None

        self.overwrite = overwrite

        # the google colab notebook
        self.colab_nb_path = os.path.join(self.nb_dir, f'colab_{fn}')


    def append(self):
        """
        Append cells required to run in google colab to top of ipynb file
        Returns
        -------
        colab_file_path : str``
            The path to the colab notebook
        """
        if os.path.exists(self.colab_nb_path) and not self.overwrite:
            logger.debug("RST version of notebook already exists at {0}. Use "
                         "overwrite=True or --overwrite (at cmd line) to re-run"
                         .format(self.colab_nb_path))
            return self.colab_nb_path

        # Read in the colab template
        with open(self.colab_template, 'r') as file:
            colab_template = file.read()
            colab_dict = json.loads(colab_template)
            colab_cells = colab_dict['cells']

        # Read in the notebook
        with open(self.nb_path, 'r') as file:
            nb = file.read()
            nb_dict = json.loads(nb)
            nb_cells = nb_dict['cells']

        colab_nb = nb_dict.copy()
        # Assumes the first cell of nb is the title
        colab_nb['cells'] = list(np.concatenate([[nb_cells[0]], colab_cells, nb_cells[1:]]))

        nb_sphinx_dict = {"nbsphinx": {"execute": "never"}}
        colab_nb['metadata'].update(nb_sphinx_dict)

        with open(self.colab_nb_path, 'w') as json_file:
            json.dump(colab_nb, json_file, indent=1)

        return self.colab_nb_path


def process_notebooks(nbfile_or_path, verbosity=None,
                      **kwargs):
    """
    Execute and optionally convert the specified notebook file or directory of
    notebook files.
    This is a wrapper around the ``NotebookConverter`` class that does file
    handling.
    Parameters
    ----------
    nbfile_or_path : str
        Either a single notebook filename or a path containing notebook files.
    exec_only : bool, optional
        Just execute the notebooks, don't run them.
    verbosity : int, optional
        A ``logging`` verbosity level, e.g., logging.DEBUG or etc. to specify
        the log level.
    **kwargs
        Any other keyword arguments are passed to the ``NotebookConverter``
        init.
    """
    if verbosity is not None:
        logger.setLevel(verbosity)

    if os.path.isdir(nbfile_or_path):
        # It's a path, so we need to walk through recursively and find any
        # notebook files
        for root, dirs, files in os.walk(nbfile_or_path):
            for name in files:
                _, ext = os.path.splitext(name)
                full_path = os.path.join(root, name)

                if 'ipynb_checkpoints' in full_path: # skip checkpoint saves
                    continue

                if name.startswith('exec'): # notebook already executed
                    continue

                if name.startswith('colab'): # ignore colab files
                    continue

                if ext == '.ipynb':
                    nbc = NotebookConverter(full_path, **kwargs)
                    nbc.append()

    else:
        # It's a single file, so convert it
        nbc = NotebookConverter(nbfile_or_path, **kwargs)
        nbc.append()


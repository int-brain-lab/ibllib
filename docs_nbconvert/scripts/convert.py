import os
import json
import time
import logging
import numpy as np

from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from nbconvert.exporters import RSTExporter
from nbconvert.exporters import HTMLExporter
from nbconvert.writers import FilesWriter
import nbformat

logger = logging.getLogger('ibllib')

IPYTHON_VERSION = 4


class NotebookConverter(object):

    def __init__(self, nb_path, output_path=None, template_file=None, colab_template=None,
                 overwrite=True, kernel_name=None):

        self.nb_path = os.path.abspath(nb_path)
        fn = os.path.basename(self.nb_path)
        self.nb_dir = os.path.dirname(self.nb_path)
        self.nb_name, _ = os.path.splitext(fn)

        if output_path is not None:
            self.output_path = output_path
            os.makedirs(self.output_path, exist_ok=True)
        else:
            self.output_path = self.nb_dir

        if template_file is not None:
            self.template_file = os.path.abspath(template_file)
        else:
            self.template_file = None

        if colab_template is not None:
            self.colab_template = os.path.abspath(colab_template)
        else:
            self.colab_template = None

        self.overwrite = overwrite

        # the executed notebook
        self.executed_nb_path = os.path.join(self.output_path, f'executed_{fn}')

        # the google colab notebook
        self.colab_nb_path = os.path.join(self.nb_dir, f'colab_{fn}')

        # the RST file
        self.rst_path = os.path.join(self.output_path, f'{self.nb_name}.rst')

        logger.info('Processing notebook {0} (in {1})'.format(fn,
                                                              self.nb_dir))

        self.execute_kwargs = dict(timeout=900, kernel_name='python3')

    def execute(self, write=True):
        """
        Execute the specified notebook file, and optionally write out the
        executed notebook to a new file.
        Parameters
        ----------
        write : bool, optional
            Write the executed notebook to a new file, or not.
        Returns
        -------
        executed_nb_path : str, ``None``
            The path to the executed notebook path, or ``None`` if
            ``write=False``.
        """

        if os.path.exists(self.executed_nb_path) and not self.overwrite:
            logger.debug("Executed notebook already exists at {0}. Use "
                         "overwrite=True or --overwrite (at cmd line) to re-run"
                         .format(self.executed_nb_path))
            return self.executed_nb_path

        # Execute the notebook
        logger.debug('Executing notebook using kwargs '
                     '"{}"...'.format(self.execute_kwargs))
        t0 = time.time()
        executor = ExecutePreprocessor(**self.execute_kwargs)

        with open(self.nb_path) as f:
            nb = nbformat.read(f, as_version=IPYTHON_VERSION)

        try:
            executor.preprocess(nb, {'metadata': {'path': self.nb_dir}})
        except CellExecutionError:
            # TODO: should we fail fast and raise, or record all errors?
            raise

        logger.info("Finished running notebook ({:.2f} seconds)".format(time.time() - t0))

        if write:
            logger.debug('Writing executed notebook to file {0}...'
                         .format(self.executed_nb_path))
            with open(self.executed_nb_path, 'w') as f:
                nbformat.write(nb, f)

            return self.executed_nb_path

    def convert(self, remove_executed=False):
        """
        Convert the executed notebook to a restructured text (RST) file.
        Parameters
        ----------
        remove_executed : bool, optional
            Controls whether to remove the executed notebook or not.
        Returns
        -------
        output_file_path : str``
            The path to the converted notebook
        """

        if not os.path.exists(self.executed_nb_path):
            raise IOError("Executed notebook file doesn't exist! Expected: {0}"
                          .format(self.executed_nb_path))

        if os.path.exists(self.rst_path) and not self.overwrite:
            logger.debug("RST version of notebook already exists at {0}. Use "
                         "overwrite=True or --overwrite (at cmd line) to re-run"
                         .format(self.rst_path))
            return self.rst_path

        # Initialize the resources dict - see:
        # https://github.com/jupyter/nbconvert/blob/master/nbconvert/nbconvertapp.py#L327
        resources = {}
        #resources['config_dir'] = '' # we don't need to specify config
        resources['unique_key'] = self.nb_name

        # path to store extra files, like plots generated
        resources['output_files_dir'] = 'nboutput/'

        # Exports the notebook to RST
        logger.debug('Exporting notebook to RST...')
        exporter = RSTExporter()
        #exporter = HTMLExporter()

        if self.template_file:
            exporter.template_file = self.template_file

        output, resources = exporter.from_filename(self.executed_nb_path,
                                                   resources=resources)

        # Write the output RST file
        writer = FilesWriter()
        output_file_path = writer.write(output, resources, notebook_name=self.nb_name)

        if remove_executed: # optionally, clean up the executed notebook file
            os.remove(self.executed_nb_path)

        return output_file_path

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
        with open(self.colab_nb_path, 'w') as json_file:
            json.dump(colab_nb, json_file, indent=1)

        return self.colab_nb_path


def process_notebooks(nbfile_or_path, exec_only=False, verbosity=None,
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
                    nbc.execute()

                    if not exec_only:
                        nbc.convert()
                        nbc.append()

    else:
        # It's a single file, so convert it
        nbc = NotebookConverter(nbfile_or_path, **kwargs)
        nbc.execute()

        if not exec_only:
            nbc.convert()
            nbc.append()


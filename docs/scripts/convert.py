from os import path, walk, remove, makedirs, sep
import os
from pathlib import Path
import re
import time
import logging

from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError
from nbconvert.exporters import RSTExporter
from nbconvert.writers import FilesWriter
import nbformat

logger = logging.getLogger('ibllib')

IPYTHON_VERSION = 4

# output_path = 'C:/Users/Mayo/iblenv/bb_documentation/template.tpl'

class NotebookConverter(object):

    def __init__(self, nb_path, output_path=None, template_file=None,
                 overwrite=False, kernel_name=None):

        self.nb_path = path.abspath(nb_path)
        fn = path.basename(self.nb_path)
        self.nb_dir = path.dirname(self.nb_path)
        self.nb_name, _ = path.splitext(fn)

        if output_path is not None:
            self.output_path = output_path
            makedirs(self.output_path, exist_ok=True)
        else:
            self.output_path = self.nb_dir

        if template_file is not None:
            self.template_file = path.abspath(template_file)
        else:
            self.template_file = None

        self.overwrite = overwrite

        # the executed notebook
        self.executed_nb_path = path.join(self.output_path, f'executed_{fn}')

        logger.info('Processing notebook {0} (in {1})'.format(fn,
                                                              self.nb_dir))

        # the RST file
        self.rst_path = path.join(self.output_path, f'{self.nb_name}.rst')

        self.execute_kwargs = dict(timeout=900)
        #if kernel_name:
        #self.execute_kwargs['kernel_name'] = 'python3'

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

        if path.exists(self.executed_nb_path) and not self.overwrite:
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

        logger.info("Finished running notebook ({:.2f} seconds)"
                    .format(time.time() - t0))

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
        delete_executed : bool, optional
            Controls whether to remove the executed notebook or not.
        """

        if not path.exists(self.executed_nb_path):
            raise IOError("Executed notebook file doesn't exist! Expected: {0}"
                          .format(self.executed_nb_path))

        if path.exists(self.rst_path) and not self.overwrite:
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

        if self.template_file:
            exporter.template_file = self.template_file

        #with open(self.executed_nb_path) as f:
        #    nb = nbformat.read(f, as_version=4)

        #output, resources = exporter.from_notebook_node(nb)
        output, resources = exporter.from_filename(self.executed_nb_path,
                                                   resources=resources)

        # Write the output RST file
        writer = FilesWriter()
        output_file_path = writer.write(output, resources, notebook_name=self.nb_name)

        if remove_executed: # optionally, clean up the executed notebook file
            remove(self.executed_nb_path)

        return output_file_path

def process_notebooks(nbfile_or_path, exec_only=False, verbosity=None,
                      **kwargs):
    """
    Execute and optionally convert the specified notebook file or directory of
    notebook files.
    This is a wrapper around the ``NBTutorialsConverter`` class that does file
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
        Any other keyword arguments are passed to the ``NBTutorialsConverter``
        init.
    """
    if verbosity is not None:
        logger.setLevel(verbosity)

    if path.isdir(nbfile_or_path):
        # It's a path, so we need to walk through recursively and find any
        # notebook files
        for root, dirs, files in walk(nbfile_or_path):
            for name in files:
                _,ext = path.splitext(name)
                full_path = path.join(root, name)

                if 'ipynb_checkpoints' in full_path: # skip checkpoint saves
                    continue

                if name.startswith('exec'): # notebook already executed
                    continue

                if ext == '.ipynb':
                    nbc = NotebookConverter(full_path, **kwargs)
                    nbc.execute()

                    if not exec_only:
                        nbc.convert()

    else:
        # It's a single file, so convert it
        nbc = NotebookConverter(nbfile_or_path, **kwargs)
        nbc.execute()

        if not exec_only:
            nbc.convert()


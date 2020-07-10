import os
import json
import time
import numpy as np

from nbconvert.preprocessors import (ExecutePreprocessor, CellExecutionError,
                                     ClearOutputPreprocessor)
from nbconvert.exporters import RSTExporter
from nbconvert.writers import FilesWriter
import nbformat

IPYTHON_VERSION = 4


class NotebookConverter(object):

    def __init__(self, nb_path, output_path=None, rst_template=None, colab_template=None,
                 overwrite=True, kernel_name=None):
        """
        Parameters
        ----------
        nb_path : str
            Path to ipython notebook
        output_path: str, default=None
            Path to where executed notebook, rst file and colab notebook will be saved. Default is
            to save in same directory of notebook
        rst_template: str, default=None
            Path to rst template file used for styling during RST conversion. If not specified,
            uses default template in nbconvert
        colab_template: str, default=None
            Path to colab code to append to notebook to make it colab compatible. If colab=True but
            colab_template not specified, the code skips colab conversion bit
        overwrite: bool, default=True
            Whether to save executed notebook as same filename as unexecuted notebook or create new
            file with naming convention 'exec_....'. Default is to write to same file
        kernel_name: str
            Kernel to use to run notebooks. If not specified defaults to 'python3'
        """

        self.nb_path = os.path.abspath(nb_path)
        self.nb = os.path.basename(self.nb_path)
        self.nb_dir = os.path.dirname(self.nb_path)
        self.nb_name, _ = os.path.splitext(self.nb)

        # If no output path is specified save everything into directory containing notebook
        if output_path is not None:
            self.output_path = os.path.abspath(output_path)
            os.makedirs(self.output_path, exist_ok=True)
        else:
            self.output_path = self.nb_dir

        # If a rst template is passed
        if rst_template is not None:
            self.rst_template = os.path.abspath(rst_template)
        else:
            self.rst_template = None

        if colab_template is not None:
            self.colab_template = os.path.abspath(colab_template)
        else:
            self.colab_template = None

        self.colab_nb_path = os.path.join(self.output_path, f'colab_{self.nb}')

        # If overwrite is True, write the executed notebook to the same name as the notebook
        if overwrite:
            self.executed_nb_path = os.path.join(self.output_path, self.nb)
        else:
            self.executed_nb_path = os.path.join(self.output_path, f'executed_{self.nb}')

        if kernel_name is not None:
            self.execute_kwargs = dict(timeout=900, kernel_name=kernel_name, allow_errors=False)
        else:
            self.execute_kwargs = dict(timeout=900, kernel_name='python3', allow_errors=False)

    def execute(self, write=True):
        """
        Executes the specified notebook file, and optionally write out the executed notebook to a
        new file.
        Parameters
        ----------
        write : bool, optional
            Write the executed notebook to a new file, or not.
        Returns
        -------
        executed_nb_path : str, ``None``
            The path to the executed notebook path, or ``None`` if ``write=False``.
        """

        # Execute the notebook
        print(f"Executing notebook {self.nb} in {self.nb_dir}")
        t0 = time.time()

        clear_executor = ClearOutputPreprocessor()
        executor = ExecutePreprocessor(**self.execute_kwargs)

        with open(self.nb_path) as f:
            nb = nbformat.read(f, as_version=IPYTHON_VERSION)

        # First clean up the notebook and remove any cells that have been run
        clear_executor.preprocess(nb, {})

        try:
            executor.preprocess(nb, {'metadata': {'path': self.nb_dir}})
        except CellExecutionError as err:
            print(f"Error executing notebook {self.nb}")
            print(err)

        print(f"Finished running notebook ({time.time() - t0})")

        if write:
            print(f"Writing executed notebook to {self.executed_nb_path}")
            with open(self.executed_nb_path, 'w') as f:
                nbformat.write(nb, f)

        return self.executed_nb_path

    def convert(self):
        """
        Converts the executed notebook to a restructured text (RST) file.
        Returns
        -------
        output_file_path : str``
            The path to the converted notebook
        """

        # Only convert if executed notebook exists
        if not os.path.exists(self.executed_nb_path):
            raise IOError("Executed notebook file doesn't exist! Expected: {0}"
                          .format(self.executed_nb_path))

        # Initialize the resources dict
        resources = dict()
        resources['unique_key'] = self.nb_name

        # path to store extra files, like plots generated
        resources['output_files_dir'] = 'nboutput/'

        # Exports the notebook to RST
        print("Exporting executed notebook to RST format")
        exporter = RSTExporter()

        # If a RST template file has been specified use this template
        if self.rst_template:
            exporter.template_file = self.rst_template

        output, resources = exporter.from_filename(self.executed_nb_path, resources=resources)

        # Write the output RST file
        writer = FilesWriter()
        output_file_path = writer.write(output, resources, notebook_name=self.nb_name)

        return output_file_path

    def append(self):
        """
        Append cells required to run in google colab to top of ipynb file. If you want to apply
        this on the unexecuted notebook, must run append method before execute!!
        Returns
        -------
        colab_file_path : str
            The path to the colab notebook
        """

        if self.colab_template is None:
            print("No colab template specified, skipping this step")
            return

        else:
            # Read in the colab template
            with open(self.colab_template, 'r') as f:
                colab_template = f.read()
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

            # Make sure colab notebooks are never executed with nbsphinx
            nb_sphinx_dict = {"nbsphinx": {"execute": "never"}}
            colab_nb['metadata'].update(nb_sphinx_dict)

            with open(self.colab_nb_path, 'w') as json_file:
                json.dump(colab_nb, json_file, indent=1)

            return self.colab_nb_path

    def unexecute(self):
        """
        Unexecutes the notebook i.e. removes all output cells
        """
        print(f"Cleaning up notebook {self.nb} in {self.nb_dir}")

        with open(self.executed_nb_path) as f:
            nb = nbformat.read(f, as_version=IPYTHON_VERSION)

        clear_executor = ClearOutputPreprocessor()
        clear_executor.preprocess(nb, {})

        with open(self.executed_nb_path, 'w') as f:
            nbformat.write(nb, f)


def process_notebooks(nbfile_or_path, execute=True, cleanup=False, rst=False, colab=False,
                      **kwargs):
    """
    Execute and optionally convert the specified notebook file or directory of
    notebook files.
    Wrapper for `NotebookConverter` class that does all the file handling.
    Parameters
    ----------
    nbfile_or_path : str
        Either a single notebook filename or a path containing notebook files.
    execute : bool
        Whether or not to execute the notebooks
    cleanup : bool, default = False
        Whether to unexecute notebook and clean up files. To clean up must set this to True and
        execute argument to False
    rst : bool, default=False
        Whether to convert executed notebook to rst format
    colab : bool, default=False
        Whether to make colab compatible notebook
    **kwargs
        Other keyword arguments that are passed to the 'NotebookExecuter'
    """

    if os.path.isdir(nbfile_or_path):
        # It's a path, so we need to walk through recursively and find any
        # notebook files
        for root, dirs, files in os.walk(nbfile_or_path):
            for name in files:
                _, ext = os.path.splitext(name)
                full_path = os.path.join(root, name)

                # skip checkpoints
                if 'ipynb_checkpoints' in full_path:
                    continue

                # if name starts with 'exec' and cleanup=True delete the notebook
                if name.startswith('exec'):
                    if cleanup:
                        os.remove(full_path)
                        continue
                    else:
                        continue

                # if name starts with 'colab' and cleanup=True delete colab notebook
                if name.startswith('colab'):
                    if cleanup:
                        os.remove(full_path)
                        continue
                    else:
                        continue

                # if name rst file and cleanup=True delete file
                if ext == '.rst':
                    if cleanup:
                        os.remove(full_path)
                        continue
                    else:
                        continue

                # if file has 'ipynb' extension create the NotebookConverter object
                if ext == '.ipynb':
                    nbc = NotebookConverter(full_path, **kwargs)
                    # Execute the notebook and optionally make colab and rst files
                    if execute:
                        if colab:
                            nbc.append()
                        nbc.execute()
                        if rst:
                            nbc.convert()
                    # If cleanup is true and execute is false unexecute the notebook
                    elif not execute & cleanup:
                        nbc.unexecute()

    else:
        # If a single file is passed in
        nbc = NotebookConverter(nbfile_or_path, **kwargs)
        if execute:
            if colab:
                nbc.append()
            nbc.execute()
            if rst:
                nbc.convert()
        # If cleanup is true and execute is false unexecute the notebook
        elif not execute & cleanup:
            nbc.unexecute()

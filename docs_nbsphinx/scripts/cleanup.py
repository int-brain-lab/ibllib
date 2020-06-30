from nbconvert.preprocessors import ClearOutputPreprocessor
import nbformat
import os
from pathlib import Path
IPYTHON_VERSION = 4

#def clean_build_notebooks(nbfile_or_path):
# make this something you pass in
nbfile_or_path = Path('C:/Users/Mayo/iblenv/iblmayo/ibllib/docs_nbsphinx/_build/html/notebooks')
print('clearning up stufffff')
for root, dirs, files in os.walk(nbfile_or_path):
    for name in files:
        _, ext = os.path.splitext(name)
        full_path = os.path.join(root, name)
        if (name.startswith('colab')) & (ext == '.html'):  # ignore colab files
            os.remove(full_path)
        if (name.startswith('colab')) & (ext == '.ipynb'):  # ignore colab files
            continue
        if ext == '.ipynb':
            with open(full_path) as f:
                nb = nbformat.read(f, as_version=IPYTHON_VERSION)
            clear_executor = ClearOutputPreprocessor()
            clear_executor.preprocess(nb, {})
            with open(full_path, 'w') as f:
                nbformat.write(nb, f)



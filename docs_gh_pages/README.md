# Overview of documentation

The documentation is built locally and hosted on a github-pages website at this address:
https://int-brain-lab.github.io/ibllib/docs/

The website is generated using
 1. The markdown files in the `./docs-gh-pages` folder
 2. The interactive python notebooks (.ipynb) in the `./docs-gh-pages/notebooks`
 3. The interactive python notebooks in the  `./examples` and `./brainbox/examples` folders
 4. The docstrings in the source code of the `./ibllib`, `./alf`, `./one` and `./brainbox` folders


# Contributing to documentation

### Including notebooks located outside of docs folder
To include .ipynb notebooks that are not in the `./docs-gh-pages` folder in the documentation you must make a 
[`.nblink`](https://github.com/vidartf/nbsphinx-link]) file that points to the location
of the external notebook. Example `.nblink` files can be found in `./docs-gh-pages/notebooks_external`.
It is recommended to place all `.nblink` files in this folder and they can be added to the 
[examples page](https://int-brain-lab.github.io/ibllib/docs/06_recipes.html) of the website
by appending them to the `06_recipes.rst` file.


## Making documentation
Once you have made your changes to the documentation, the documentation can be built using the following command. This
executes all .ipynb notebooks included in the documentation and uses nb-sphinx and sphinx to then generate the built 
html version of the files. 

```python
cd ./docs-gh-pages
python make_script.py -e -d -c
```
- `-e` executes all the notebooks specified in the build path
- `-d` builds the documentation using sphinx
- `-c` unexecutes all notebooks and removes any unwanted files

Once this script has completed a preview of the documentation can be viewed by opening 
`./docs-gh-pages/_build/html/index.html` in a web browser.

Check that all notebooks have run without errors and that your changes have been implemented correctly!

## Pushing changes to gh-pages
Once you are happy with the built documentation, the changes can be deployed to the website by running the following
command

```python
python make_script.py -gh -m "your commit message"
```

## Flake8nb
Linting checks can be applied to ipynb. notebooks using flake8_nb, an example implementation is
```python
flake8_nb .\notebooks\one_intro\one_intro.ipynb
```


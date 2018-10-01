# Installation guide
## Python Module
### Linux
Clone the github, setup a virtualenv, install requirements and install the IBL packages.
```
git clone https://github.com/int-brain-lab/ibllib.git
cd ibllib/
virtualenv venv --python=python3.6
source ./venv/bin/activate
pip install -r ./python/requirements.txt
cd python
python ./setup.py install
```

Eventually run the tests, it needs Alyx and FlatIron credentials as defined above.
```
source run_tests
```
To run with Spyder, just link the Python Interpreter with the virtualenv you've created.
To run with Jupyter, type in the command with your virtual environment activated as above:
```
jupyter notebook
```

### Windows
Clone the github if you have Git for Windows installed.
```
git clone https://github.com/int-brain-lab/ibllib.git
```
Otherwise download the the zip file from the Github website and unzip.

```
conda create -n ibllibenv
conda activate ibllibenv
cd ibllib/python
pip install -r requirements.txt
python setup.py install
```

Eventually run the tests, it needs Alyx and FlatIron credentials as defined above.
```
call run_tests.bat
```

### Getting started
Short introduction for the Ipython notebook [here](./_static/one_demo.html)
A longer tutorial is available in the tutorial section.

## Matlab Package
In a command line, clone the git (Linux) or download the zip file from GitHub (Windows).
```
git clone https://github.com/int-brain-lab/ibllib.git
```

Launch Matlab.
Set Path, add with subfolders the '.\ibllib\matlab' directory.


```matlab
one.setup

```

Eventually run the suite of Unit tests to check the installation:

```matlab
RunTestsIBL('All')

```

A great place to start is with the tutorials of the next section.

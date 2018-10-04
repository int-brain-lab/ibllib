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

Introduction to ONE

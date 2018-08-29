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

Eventually run the tests, it needs credentials as defined above.
```
source run_tests
```

To run with Spyder, just link the Python Interpreter with the virtualenv you've created.

### Windows

## Matlab Package
In a command line, clone the git (Linux) or download the zip file from GitHub (Windows).
```
git clone https://github.com/int-brain-lab/ibllib.git
```

```matlab
AlyxClient.setup()
```

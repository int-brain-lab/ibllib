# IBLLIB Python Installation Guide

## Python-specific Dependencies

Python-specific dependency : **Python 3.7 or higher**.

## Initialisation

The following steps will indicate how to :
1. setup a virtual environment using Anaconda
2. define ONE connection settings
3. Optional: test for the installation

### Environment and ibllib setup
The IBL has a unified set of dependencies for various applications, ibllib being one of those.
The steps to setup the unified environment are here.
https://github.com/int-brain-lab/iblenv

### Instantiate One class: Define connection settings

The first step is to instantiate the **One class**: behind the scenes, the constructor connects to the IBL cloud database and gets credentials.

The connections settings are defined in a JSON parameter file (named *.one_params*). The file is created when you first run `ONE.setup()`
-   In Linux, the file is in `~/.one_params`.
-   In Windows, the file is in the Roaming App directory `C:\Users\CurrentUser\AppData\Roaming\.one_params`.
-   In Mac OS, the file is in the user directory `/Users/CurrentUser/.one_params`.

In case of doubt, type the command `io.getappdir` in a Matlab prompt, `from pathlib import Path; print(Path.home())` in Python.
It will return the directory of the JSON *.one_params* file.

**_Note_: The JSON _.one_params_ file is uniquely stored, and shared across Matlab and Python.**


There are two ways to define the connection settings.

#### 1. The `setup()` static method

In a Python terminal, type:

```python
from oneibl.one import ONE
ONE.setup() # For first time use, need to define connection credentials
```

**_Note_**:
-   you can access a python terminal from a shell terminal by typing the command `python` in the ibllibenv virtual environment) For an `ipython` console, `pip install ipython` in the environment.
-   To run with _Spyder_, link the Python Interpreter to the virtual environment you created.
-   To run with _Jupyter_, type in the command `jupyter notebook` with your virtual environment activated as above (`pip install jupyter` will install jupyter if it's not already available).


You will be asked to enter the following information:

```python
ALYX_LOGIN 				# Input your IBL user name
ALYX_PWD				# Input your IBL password
ALYX_URL:				# Should be automatically set as: https://alyx.internationalbrainlab.org - press ENTER
CACHE_DIR:				# Local repository, can ammend or press ENTER
FTP_DATA_SERVER: 		# Should be automatically set as: ftp://ibl.flatironinstitute.org - press ENTER
FTP_DATA_SERVER_LOGIN:	# Should be automatically set as: iblftp - press ENTER
FTP_DATA_SERVER_PWD		# Request Password for FTP from Olivier
HTTP_DATA_SERVER: 		# Should be automatically set as: http://ibl.flatironinstitute.org  - press ENTER
HTTP_DATA_SERVER_LOGIN: # Should be automatically set as: iblmember  - press ENTER
HTTP_DATA_SERVER_PWD	# Request Password for HTTP from IBL admins
```

**Note: using `one.setup` changes the JSON *.one_params* file.** Again, the file is shared across Python and Matlab platforms.

#### 2. Edit the JSON *.one_params* file manually
**_Note_**: In Mac OS/Linux, use the command `nano` in a terminal.

Once the connections settings are defined, there is no need to setup the class One again if willing to connect with the credentials saved in the JSON *.one_params* file.

The tutorial in the next section will show you how to change credentials withough changing the JSON file (useful for seldom connection with different credentials).

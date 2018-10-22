# IBLLIB Python user guide

## Python-specific Dependencies

Python-specific dependency : **Python 3.6 or higher**.

### Install Anaconda/Miniconda onto your machine
Download and install  the  Anaconda  python  distribution from here (chosing the right OS): https://www.anaconda.com/download/#download
_Note_ : Download the latest version.


## Initialisation

Before you begin, make sure you have installed ibllib properly on your system as per the previous instructions.
Make sure your computer is connected to an IBL accredited network.

The following steps will indicate how to :
1. setup a virtual environment using Anaconda
2. install requirements and packages
3. define ONE connection settings
4. test for the installation
All of that can be done using shell terminal command lines.

### Setup a virtual environment using Anaconda

In a shell terminal, type the following commands:

**Linux:**
```
cd ibllib/
virtualenv venv --python=python3.6
source ./venv/bin/activate
```

**Windows:**
```
conda create -n ibllibenv
conda activate ibllibenv
```

**Mac OS:**
```
cd Documents/GitHub/ibllib
conda create -n ibllibenv python=3.6 anaconda
source activate ibllibenv
```


### Install requirements and packages
**Linux and Mac OS:**
```
pip install -r ./python/requirements.txt
cd python
pip install -e .
```

**Windows:**
```
cd ibllib/python
pip install -r requirements.txt
pip install -e .
```


### Instantiate One class: Define connection settings

The first step is to instantiate the **One class**: behind the scenes, the constructor connects to the IBL cloud database and gets credentials.

The connections settings are defined in a JSON parameter file (named *.one_params*).
-   In Linux, the file is in `~/.one_params`.
-   In Windows, the file is in the Roaming App directory `C:\Users\olivier\AppData\Roaming\.one_params`.
-   In Mac OS, the file is in the user directory `/Users/olivier/.one_params`.

In case of doubt, type the command `io.getappdir` in a Matlab prompt. It will return the directory of the JSON *.one_params* file.

**_Note_: The JSON _.one_params_ file is uniquely stored, and shared across Matlab and Python.**


There are two ways to define the connection settings.

#### 1. The `setup()` static method

In a Python terminal, type:

```python
from oneibl.one import ONE
one = ONE() # need to instantiate the class to have the API.
one.setup() # For first time use, need to define connection credentials
```

**_Note_**:
-   you can access a python terminal from a shell terminal by typing the command `python` in the ibllibenv virtual environment)
-   To run with _Spyder_, link the Python Interpreter to the virtual environment you created.
-   To run with _Jupyter_, type in the command `jupyter notebook` with your virtual environment activated as above


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
HTTP_DATA_SERVER_PWD	# Request Password for HTTP from Olivier
```

**Note: using `One.setup` changes the JSON *.one_params* file.** Also note that the file is shared across Python and Matlab platforms.

#### 2. Edit the JSON *.one_params* file manually
**_Note_**: In Mac OS/Linux, use the command `nano` in a terminal.


Once the connections settings are defined, there is no need to setup the class One again if willing to connect with the credentials saved in the JSON *.one_params* file.

The tutorial in the next section will show you how to change credentials withough changing the JSON file (useful for seldom connection with different credentials).


### Run tests
Exit the python terminal.
In a shell terminal, in the ibllibenv, `cd python` (as above) and write the following:
-   **Linux and Mac OS**: `source run_tests`
-   **Windows**: `call run_tests.bat`



If you see any Failure message, please report on GitHub or contact Olivier.
If not, you are ready for the tutorial - go to next section !


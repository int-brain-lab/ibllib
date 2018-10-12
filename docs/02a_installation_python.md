# IBLLIB Installation guide

The IBLLIB provides the users with functions to connect to the IBL databases, search, load and read data onto local machines (amongst other). These functions are developed in both Matlab and Python languages.

Contained within the IBLLIB, **Open Neurophysiology Environment (ONE)** is a set of simple loader functions.


## Main Dependencies

Main dependencies: **IBLLIB** repository and **Git**. 

_Note:_ Alternatively to installing Git and following the steps below, you can download the IBLLIB repository directly from the github webpage https://github.com/int-brain-lab/ibllib.git .

### Install Git onto your machine
**Linux** : Download and install Git from here: TODO OLIVIER .
**Mac OS** : Download and install Git from here: https://git-scm.com/download/mac .
**Windows** : Download and install Git from here: https://github.com/git-for-windows/git/releases/ .


### Clone IBLLIB repository onto your machine
The following steps indicate how to clone the IBLLIB repository on your machine once Git is installed.
Open a shell terminal, and type the following command to clone the IBLLIB repository:

(this command is valid for any type of OS)
```
git clone https://github.com/int-brain-lab/ibllib.git
```

**Linux** : This command creates a folder in your working directory.
**Mac OS**: This command creates a folder GitHub in Documents. The GitHub folder ontains the -ibllib- folder.
**Windows** : This command creates a folder in your working directory.

# IBLLIB Python user guide

## Python-specific Dependencies

Python-specific dependency : **Anaconda python distribution**.

### Install Anaconda onto your machine
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
python ./setup.py install
```

**Windows:**
```
cd ibllib/python
pip install -r requirements.txt
python setup.py install
```


### Instantiate One class: Define connection settings

The first step is to instantiate the **One class**: behind the scenes, the constructor connects to the IBL cloud database and gets credentials. 

The connections settings are defined in a JSON parameter file (named *.one_params*).
In Linux, the file is in `~/.one_params`.
In Windows, the file is in the Roaming App directory `C:\Users\olivier\AppData\Roaming\.one_params`.
In Mac OS, the file is in the user directory `/Users/olivier/.one_params`.
In case of doubt, type the command `io.getappdir` in a Matlab prompt. It will return the directory of the JSON *.one_params* file. 
Note: **The JSON *.one_params* file is uniquely stored, and shared across Matlab and Python.**


There are two manners to define the connection settings.

1. The `setup()` static method allows to update parameters via a Python user prompt.

In a Python terminal, type: 
_Note_: you can access a python terminal from a shell terminal by typing the command `python` in the ibllibenv virtual environment)
_Note_:
To run with Spyder, link the Python Interpreter to the virtual environment you created.
To run with Jupyter, type in the command with your virtual environment activated as above:
```
jupyter notebook
```

```python
from oneibl.one import ONE
one = ONE() # need to instantiate the class to have the API.
one.setup() # For first time use, need to define connection credentials

```

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

**Note that using `one.setup()` changes the JSON *.one_params* file.** Also note that the file is shared across Python and Matlab platforms.

2. Update the JSON *.one_params* file manually, for example via a text editor. 
_Note_: In Mac OS, use the command nano in a terminal.


Once the connections settings are defined, there is no need to setup the class One again if willing to connect with the credentials saved in the JSON *.one_params* file.

The tutorial in the next section will show you how to change credentials withough changing the JSON file (useful for seldom connection with different credentials).


### Run tests
Exit the python terminal.
In a shell terminal, in the ibllibenv, `cd python` (as above) and write the following:


**Linux and Mac OS**: `source run_tests`
**Windows**: `call run_tests.bat`



If you see any Failure message, please report on GitHub or contact Olivier.
If not, you are ready for the tutorial - go to next section !


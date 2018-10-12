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

**Linux** : TODO
**Mac OS**: This command creates a folder GitHub in Documents. The GitHub folder ontains the -ibllib- folder.
**Windows** : TODO

# IBLLIB Python user guide

## Python-specific Dependencies

Python-specific dependency : **Anaconda python distribution**.

### Install Anaconda onto your machine
Download and install  the  Anaconda  python  distribution from here (chosing the right OS): https://www.anaconda.com/download/#download  
_Note_ : Download the latest version.


## Initilisation

Before you begin, make sure you have installed ibllib properly on your system as per the previous instructions.
Make sure your computer is connected to an IBL accredited network.

The following steps will indicate how to :
1. setup a virtual environment using Anaconda
2. install requirements and packages
3. test for the installation
All of that is done using shell terminal command lines.

### Linux

In a shell terminal, type the following commands:
1. To setup a virtual environment:
```
cd ibllib/
virtualenv venv --python=python3.6
source ./venv/bin/activate
```
2. To install requirements and packages:
```
pip install -r ./python/requirements.txt
cd python
python ./setup.py install
```

3. Eventually run the tests (it needs Alyx and FlatIron credentials, see connection settings below).
```
source run_tests
```
To run with Spyder, just link the Python Interpreter with the virtual environment you created.
To run with Jupyter, type in the command with your virtual environment activated as above:
```
jupyter notebook
```

### Windows
In a shell terminal, type the following commands:
1. To setup a virtual environment:
```
conda create -n ibllibenv
conda activate ibllibenv
```
2. To install requirements and packages:
```
cd ibllib/python
pip install -r requirements.txt
python setup.py install
```
3. Eventually run the tests (it needs Alyx and FlatIron credentials, see connection settings below).
```
call run_tests.bat
```

### Mac OS
In a shell terminal, type the following commands:
1. To setup a virtual environment:
```
cd Documents/GitHub/ibllib
conda create -n ibllibenv python=3.6 anaconda
source activate ibllibenv
```
Type _y_ to proceed.
_Note_: If you have issues with creating an environment, check the associate [GitHub Help webpage](https://uoa-eresearch.github.io/eresearch-cookbook/recipe/2014/11/20/conda/).

2. To install requirements and packages:
```
pip install -r ./python/requirements.txt
cd python
python ./setup.py install
```
3. Eventually run the tests (it needs Alyx and FlatIron credentials, see connection settings below).
```
source run_tests
```

### Define connection settings
```
Param ALYX_LOGIN,  current value is [test_user]:
Param ALYX_URL,  current value is [https://test.alyx.internationalbrainlab.org]:
Param CACHE_DIR,  current value is [/Users/gaelle/Downloads/FlatIron]:
Param FTP_DATA_SERVER,  current value is [ftp://ibl.flatironinstitute.com]:
Param FTP_DATA_SERVER_LOGIN,  current value is [iblftp]:
Param HTTP_DATA_SERVER,  current value is [http://ibl.flatironinstitute.com]:
Param HTTP_DATA_SERVER_LOGIN,  current value is [iblmember]:
Enter the Alyx password for test_user(leave empty to keep current):
Enter the FlatIron HTTP password for iblmember(leave empty to keep current): 
Enter the FlatIron FTP password for iblftp(leave empty to keep current): 
ONE Parameter file location: /Users/gaelle/.one_params

```
# ONE Matlab Installation Guide


## Matlab-specific Dependencies

Matlab-specific dependency : **Matlab R2016b or higher**.

### Install Matlab onto your machine
Download and install the latest Matlab version from here: [https://www.mathworks.com/downloads/](https://www.mathworks.com/downloads/)



## Initialisation

Clone or download the repository here: [https://github.com/int-brain-lab/ibllib-matlab](https://github.com/int-brain-lab/ibllib-matlab)

Launch Matlab.
Set the Matlab path, add with subfolders the full `.\ibllib-matlab` directory.

### Instantiate One class: Define connection settings

The first step is to instantiate the **One class**: behind the scenes, the constructor connects to the IBL cloud database and gets credentials. 

The connections settings are defined in a JSON parameter file (named *.one_params*).
-   In Linux, the file is in `~/.one_params`.
-   In Windows, the file is in the Roaming App directory `C:\Users\olivier\AppData\Roaming\.one_params`.
-   In Mac OS, the file is in the user directory `/Users/olivier/.one_params`.

In case of doubt, type the command `io.getappdir` in a Matlab prompt. It will return the directory of the JSON `.one_params` file. 


There are two manners to define the connection settings.

### 1. The `setup()` static method in Matlab
In a Matlab prompt, write:

```matlab
One.setup
```

You will be asked to enter the following information:

```matlab
ALYX_LOGIN 				% Input your IBL user name
ALYX_PWD				% Input your IBL password
ALYX_URL:				% Should be automatically set as: https://alyx.internationalbrainlab.org - press ENTER
CACHE_DIR:				% Local repository, can ammend or press ENTER
FTP_DATA_SERVER: 		% Should be automatically set as: ftp://ibl.flatironinstitute.org - press ENTER
FTP_DATA_SERVER_LOGIN:	% Should be automatically set as: iblftp - press ENTER
FTP_DATA_SERVER_PWD		% Request Password for FTP from Olivier
HTTP_DATA_SERVER: 		% Should be automatically set as: http://ibl.flatironinstitute.org  - press ENTER 
HTTP_DATA_SERVER_LOGIN: % Should be automatically set as: iblmember  - press ENTER
HTTP_DATA_SERVER_PWD	% Request Password for HTTP from Olivier
```

The path to the *.one_params* file is displayed in the Matlab prompt as `ans`.

**Note: using `One.setup` changes the JSON *.one_params* file.** Also note that the file is shared across Python and Matlab platforms.



### 2. Edit the JSON *.one_params* 
**_Note_**: In Mac OS or Linux, use the command `nano` in a terminal.


Once the connections settings are defined, there is no need to setup the class One again if willing to connect with the credentials saved in the JSON *.one_params* file.

The tutorial in the next section will show you how to change credentials without changing the JSON file (useful for a temporary connection with different credentials).



### Run tests

Once the One class is instantiated with your IBL credentials, run the suite of Unit tests to check the installation:

```matlab
RunTestsIBL('All')

```

If you see any Failure message, please report on GitHub or contact data admins (Olivier / Niccolo).
If not, you are ready for the tutorial - go to next section !

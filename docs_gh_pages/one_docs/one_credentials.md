# Setting up ONE credentials

In order to use the ONE interface to access IBL data, it is necessary to provide some credentials that allow ONE to 
connect to the Alyx database and the FlatIron server. These credentials are stored locally on your computer in a JSON 
parameter file called *.one_params*. The following instructions will walk you through how to create and 
configure this file.

```{important}
To set up credentials you will need access to your Alyx username and password in addition to the IBL FlatIron password. 
If you do not have access to these, please get in contact with a member of the IBL software team.
```


## Using the `one.setup()` method

ONE contains a `setup` method that automatically creates the *.one_params* file in the correct directory and prompts you
to enter your credentials. 

In a python terminal ([with your ibl environment activated](../02_installation)), type:

```python
from oneibl.one import ONE
ONE.setup() 
```

You will be prompted to enter information in the following order. 
 
  
```python
ALYX_LOGIN:             # Input your Alyx username
ALYX_URL:               # Change to https://alyx.internationalbrainlab.org
CACHE_DIR:              # Optionally change or keep default
FTP_DATA_SERVER:        # Keep default - should be automatically set as: ftp://ibl.flatironinstitute.org
FTP_DATA_SERVER_LOGIN:	# Keep default - should be automatically set as: iblftp
HTTP_DATA_SERVER:       # Keep default - should be automatically set as: http://ibl.flatironinstitute.org
HTTP_DATA_SERVER_LOGIN: # Keep default - should be automatically set as: iblmember
Alyx password:          # Input your Alyx password
FlatIron HTTP password:	# Input FlatIron password
FlatIron FTP password:	# Input FlatIron password
```
The entries that you will need to change from default are: `ALYX_LOGIN`, `ALYX_URL`, `Alyx password`, 
`FlatIron HTTP password` and `FlatIron FTP password`. You can also optionally change the `CACHE_DIR` (the local 
directory where downloaded files will be saved). For the remaining entries keep the default values by pressing 
the Enter key.

Once you have completed the setup process the location where the *.one_params* is saved will be printed in the python
terminal. This location differs depending on your operating system

-   **Linux**  `~/.one_params`
-   **Windows** `C:\Users\CurrentUser\AppData\Roaming\.one_params`
-   **Mac** `/Users/CurrentUser/.one_params`

Double check that the file has been created in the correct location and that the content looks like this,

```python
    {
    "ALYX_LOGIN": "alyx_username",
    "ALYX_PWD": "alyx_password",
    "ALYX_URL": "https://alyx.internationalbrainlab.org",
    "CACHE_DIR": "cache directory that you chose",
    "FTP_DATA_SERVER": "ftp://ibl.flatironinstitute.org",
    "FTP_DATA_SERVER_LOGIN": "iblftp",
    "FTP_DATA_SERVER_PWD": "flatiron_password",
    "HTTP_DATA_SERVER": "http://ibl.flatironinstitute.org",
    "HTTP_DATA_SERVER_LOGIN": "iblmember",
    "HTTP_DATA_SERVER_PWD": "flatiron_password",
    "GLOBUS_CLIENT_ID": null
    }          

```


```{note}
It is also possible to manually create the *.one_params* JSON file. To do this, open a text-editor and copy 
the content of the json output above. Enter your Alyx and IBL credentials and save the file in the correct
location according to the operating system that you use. This can be found by typing, 
```python
from pathlib import Path
print(Path.home())
```


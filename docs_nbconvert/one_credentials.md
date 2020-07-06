# Setting up ONE credentials



To allow the ONE module we must define some credentials, these are stored locally on your computer and the one module
uses these when attempting to access alyx and flatiron

Warning! In order to setup credentials you must have access to your alyx user name and password and the ibl flation
password. If you do not have these, please get in contact with a member of the SW dev team

The first step
The file can be created and credentials set using the one.setup() command
from oneibl.one
one.setup()

You will be given the following options.
For all except , for the others enter the credentials

Double check that the file has been created. On windows it will be located in, wheras on Mac\Linux it will be

Once you have run the setup and saved these credentials you will not have to run these steps again

Note. It is also possible to create the one.params file manually

# Setting up Datajoint credentials

Before you can start using Datajoint with IBL data on your local computer, you will need to configure some Datajoint 
credentials. We need to specify a database connection to tell Datajoint where to look for IBl data as well as grant 
access by providing a username and password.

```{important}
To set up credentials you will need access to your Datajoint username and password. If you do not have access to these, 
please get in contact with a member of the IBL software team.
```

## Configuring the database host

Start by importing Datajoint and printing out the current configuration settings.

In a python terminal ([with your ibl environment activated](../02_installation)), type:

```python
import datajoint as dj
dj.config
```

The database connection is specified by the key `database.host`. You may find that this is already set to 
`datajoint.internationalbrainlab.org`. If it is not, we can manually change it by typing,

```python
dj.config['database.host'] = 'datajoint.internationalbrainlab.org'
dj.config
```

Now we are pointing to the correct database, let's try connecting. You can explicitly trigger a connection using 
`dj.conn()`. You will be prompted to enter a username and password.

```python
dj.conn()
```

Once you verify that the connection is working, it is useful to save the configuration so that you don’t have to keep on 
changing the `database.host` every time you work with DataJoint. Simply run the following command to save the 
configuration,

```python
dj.config.save_local()
```

This will have created a JSON configuration file `dj_local_config` in your local directory. Check that this has been 
created and that the content matches the following

```python
{
    "database.host": "datajoint.internationalbrainlab.org",
    "database.password": null,
    "database.user": null,
    "database.port": 3306,
    "database.reconnect": true,
    "connection.init_function": null,
    "connection.charset": "",
    "loglevel": "INFO",
    "safemode": true,
    "fetch_format": "array",
    "display.limit": 12,
    "display.width": 14,
    "display.show_tuple_count": true
}
```
Notice how the `database.host` is set to the correct value.

## Saving username and password
Although you now don’t have to keep on specifying `database.host` inside `dj.config` every time DataJoint tries to 
connect to the database, it will prompt you for your username and password. This may be fine when working interactively, 
however, it can be rather limiting when you want a script to run without interaction. To get around this, you can also 
save your username and password in a similar way.

```python
from getpass import getpass # use this to enter your password without displaying it in the terminal
dj.config['database.user'] = 'dj_username' # Type in your dj username
dj.config['database.password'] = getpass('Type password:') # Type in your dj password
```

Let's confirm the username and password have been correctly configured,

```python
dj.conn()
```

You should find that Datajoint automatically connects to the database! Finally, let's make sure we save these 
credentials into the local config file by typing,

```python
dj.config.save_local()
```

Now that these credentials have been configured, every time you type `import datajoint as dj` it will automatically 
connect to the correct database and log you in. Let's go to the next session, to get started with using Datajoint with 
IBL data.


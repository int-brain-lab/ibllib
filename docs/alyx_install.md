# Alyx installation instructions


## Install ubuntu serevr 16.04

Install ubuntu server 16.04, pick a user name and a server name.

Information you should remember about your server:

    serverIP address    e.g. 10.40.11.137
    servername          e.g. myalyx
    server username     e.g. myuser (user should have sudo privileges)
    userpassword        e.g. mypass

Once ubuntu is installed on your server box you shouldn't need physical access to it any more.

From a computer on the same network type:

    someuser@somebox~$ ssh myuser@myalyx

OR if your network has no DNS server

    someuser@somebox~$ myuser@10.40.11.137

Type in your password:

    myuser@myalyx's password: mypass
    myuser@myalyx~$

OK, now you can start installing all the required software:


## Postgres installation and setup

Install postgresql and python3-pip (we'll need this for later) ubuntu server should already have it.

    myuser@myalyx~$ sudo apt-get update
    myuser@myalyx~$ sudo apt-get install python3-pip python3-dev libpq-dev postgresql postgresql-contrib

Now to configure your postgresql backend:

    myuser@myalyx~$ sudo -u postgres psql

Create a database. pick a name I use the termination db for clarity e.g. labdb

    postgres=# CREATE DATABASE labdb;

Create a user and a password e.g. labdbuser, labdbuserpass

    postgres=# CREATE USER labdbuser WITH PASSWORD 'labdbuserpass';

Configure user/role

    postgres=# ALTER ROLE labdbuser SET client_encoding TO 'utf8';
    postgres=# ALTER ROLE labdbuser SET default_transaction_isolation TO 'read committed';
    postgres=# ALTER ROLE labdbuser SET timezone TO 'UTC';
    postgres=# GRANT ALL PRIVILEGES ON DATABASE labdb TO labdbuser;
    postgres=# ALTER USER labdbuser WITH SUPERUSER;
    postgres=# ALTER USER labdbuser WITH CREATEROLE;
    postgres=# ALTER USER labdbuser WITH CREATEDB;

Now you can list your progress using the command __\l__. you should have something like this:

    postgres=# \l
                                      List of databases
       Name    |  Owner   | Encoding |   Collate   |    Ctype    |   Access privileges
    -----------+----------+----------+-------------+-------------+-----------------------
     labdb     | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
     postgres  | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 |
     template0 | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | =c/postgres          +
               |          |          |             |             | postgres=CTc/postgres
     template1 | postgres | UTF8     | en_US.UTF-8 | en_US.UTF-8 | =c/postgres          +
               |          |          |             |             | postgres=CTc/postgres
    (4 rows)

To list users/roles type __\dg__ or __\du__

    postgres=# \dg
                                       List of roles
     Role name |                         Attributes                         | Member of
    -----------+------------------------------------------------------------+-----------
     labdbuser | Superuser, Create role, Create DB                          | {}
     postgres  | Superuser, Create role, Create DB, Replication, Bypass RLS | {}
    ---

Now you can quit postgres prompt and install alyx:

    postgres=# \q
    myuser@myalyx~$

## Alyx

If not already there, change to your home folder:

    myuser@myalyx~$ cd ~

Clone the alyx repo by typing:

    myuser@myalyx~$ sudo git clone https://github.com/cortex-lab/alyx.git

Now let's cd into it and make sure we got the master branch:

    myuser@myalyx~$ cd alyx/
    myuser@myalyx~$ git branch
    ---
    * master
    ---

You should have installed python3-pip earlier so you can now install all the requirements for alyx:

    myuser@myalyx~/alyx$ pip3 install -r requirements.txt

Now in order for alyx to run it needs some information about the backend and users of the database we just created.

To do that, first copy the file secret_settings_template.py and rename it to settings_secret.py:

    myuser@myalyx~/alyx$ sudo cp alyx/alyx/settings_secret_template.py alyx/alyx/settings_secret.py

Now change the contents of the file to reflect the configuration created in step 1. like this:

    myuser@myalyx~/alyx$ sudo nano alyx/alyx/settings_secret.py
    ---
    DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'labdb',
        'USER': 'labdbuser',
        'PASSWORD': 'labdbuserpass]',
        'HOST': '127.0.0.1',
        'PORT': '5432',
        }
    }
    ---

Now to setup the default configuration of alyx type:

    myuser@myalyx~/alyx$ cd alyx
    myuser@myalyx~/alyx/alyx$ sudo ./rm_all_db_migrations.sh
    myuser@myalyx~/alyx/alyx$ sudo python3 manage.py makemigrations
    myuser@myalyx~/alyx/alyx$ sudo python3 manage.py migrate

Finally to run the server to the local network type:

    myuser@myalyx~/alyx/alyx$ sudo python3 manage.py runserver 0.0.0.0:8000

In order to see alyx's welcome page you probably need to add your server to the list of allowed servers in settings.py. It should be in `~/alyx/alyx/alyx/settings.py`

Also it's a good idea to change the list of superusers and colony manager.

You'll need to create a superuser for Alyx by running:

    myuser@myalyx~/alyx/alyx$ sudo python3 manage.py createsuperuser

Create the default root user by pressing Enter and set a contact e-mail and password.
With this user you should be able to set up any other user and select permissions as needed from whithin Alyx itself.

You can use screen to run the server and detatch form it so you can logoff the ssh session without stopping the server

Use -S to name the screen session

    myuser@myalyx~/alyx/alyx$ screen -S alyx
    myuser@myalyx~/alyx/alyx$ sudo python3 manage.py sunserver 0.0.0.0:8000

To detach from session type:

    myuser@myalyx~/alyx/alyx$ Ctrl + A d

To reattach run:

    myuser@myalyx~/alyx/alyx$ screen -r alyx  # to reattach


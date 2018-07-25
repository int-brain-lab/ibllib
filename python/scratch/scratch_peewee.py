# -*- coding: utf-8 -*-
# @Author: Niccolò Bonacchi
# @Date:   2018-06-29 11:13:39
# @Last Modified by:   Niccolò Bonacchi
# @Last Modified time: 2018-06-29 14:14:38
from peewee import *
from playhouse.reflection import Introspector

db = 'mainendb'
user = 'nico'
password = '123'
host = '10.40.11.236'
port = 5432

db = PostgresqlDatabase(db, user=user, password=password, host=host,
                        port=port)

# introspector = Introspector.from_database(db)
# models = introspector.generate_models()

import json

acros = json.loads(open("acronyms.json").read())
names = json.loads(open("full_names.json").read())

ac2name = dict(zip(acros, names))
ac2name['void'] = 'void'
name2ac = dict(zip(names, acros))
name2ac['void'] = 'void'

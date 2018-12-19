import csv


def read_as_list(csv_file):
    out = []
    with open(csv_file) as fid:
        cr = csv.reader(fid, delimiter=',')
        c = 0
        for row in cr:
            if c == 0:
                cnames = row
            dico = {}
            for i in range(len(cnames)):
                if cnames[i]:
                    dico[cnames[i]] = row[i]
            out.append(dico)
            c += 1
    return out

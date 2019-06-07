import csv


def csv_as_list(csv_file):
    out = []
    with open(csv_file) as fid:
        cr = csv.reader(fid, delimiter=',')
        c = -1
        for row in cr:
            c += 1
            if c == 0:
                cnames = row
                continue
            dico = {}
            for i in range(len(cnames)):
                if cnames[i]:
                    dico[cnames[i]] = row[i]
            out.append(dico)
    return out

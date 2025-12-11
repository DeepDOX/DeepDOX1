import json, os, sys
import numpy as np
from scipy import stats
from statistics import mean

CASF_DATA = dict()
CASF_GROUP = dict()

with open(os.path.join(os.path.dirname(__file__), "CASF2016pKi.json")) as fhr:
    for line in json.load(fhr):
        pdb = line["pdb"]
        pki = line["value"]
        group = line["group"]
        assert pdb not in CASF_DATA
        CASF_DATA[pdb] = pki
        if group in CASF_GROUP:
            CASF_GROUP[group].append(pdb)
        else:
            CASF_GROUP[group] = [pdb, ]
    del line
    del pdb
    del pki
    del group
    assert len(CASF_DATA) == 285
    assert len(CASF_GROUP) == 57


def pearson_correlation_coefficient(predicts: dict[str, float], ignore_missing=True) -> float | None:
    x = list()
    y = list()
    missing = list()
    for key, real in CASF_DATA.items():
        if key in predicts:
            x.append(predicts[key])
            y.append(real)
        elif ignore_missing:
            missing.append(key)
        else:
            missing.append(key)
            x.append(0.)
            y.append(real)
    if len(missing) > 0:
        print("[Warning] The following PDB(s) in CASF are missing:\n[Warning]  " + " ".join(missing), file=sys.stderr)
    extra = list()
    for key in predicts.keys():
        if key not in CASF_DATA:
            extra.append(key)
    if len(extra) > 0:
        print("[Warning] The following PDB(s) are NOT in CASF:\n[Warning]  " + " ".join(extra), file=sys.stderr)
    if len(x) < 3:
        print("[Warning] There are {} PDB(s), which is too little.".format(len(x)), file=sys.stderr)
        return None
    else:
        r = stats.pearsonr(np.array(x), np.array(y))
        return float(r.statistic)


def spearman_correlation_coefficient(predicts: dict[str, float], ignore_missing=False) -> float | None:
    spearmans = list()
    missing = list()
    for i in range(1, 58):
        pdbs = CASF_GROUP[i]
        x = list()
        y = list()
        for key in pdbs:
            if key in predicts:
                x.append(predicts[key])
                y.append(CASF_DATA[key])
            elif ignore_missing:
                missing.append(key)
            else:
                missing.append(key)
                x.append(0.)
                y.append(CASF_DATA[key])
        gplen = len(x)
        if gplen < 3:
            print("[Warning] There are {} PDB(s) in GROUP {}, which is too little.".format(gplen, i), file=sys.stderr)
        else:
            if gplen < 5:
                print("[Warning] The GROUP {} has {} PDB(s), which is less than usual.".format(i, gplen),
                      file=sys.stderr)
            r = stats.spearmanr(np.array(x), np.array(y))
            spearmans.append(float(r.statistic))
    if len(missing) > 0:
        print("[Warning] The following PDB(s) in CASF are missing:\n[Warning]  " + " ".join(missing), file=sys.stderr)
    extra = list()
    for key in predicts.keys():
        if key not in CASF_DATA:
            extra.append(key)
    if len(extra) > 0:
        print("[Warning] The following PDB(s) are NOT in CASF:\n[Warning]  " + " ".join(extra), file=sys.stderr)
    if len(spearmans) == 0:
        print("[Warning] There are no enough groups to compute Spearman correlation coefficient", file=sys.stderr)
        return None
    else:
        return mean(spearmans)

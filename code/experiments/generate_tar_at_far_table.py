"""
Calculate TAR for specific FAR values. Spit out table organized by subgroup.
"""
import numpy as np
import pandas as pd
from facebias.utils import add_package_path
from sklearn.metrics import confusion_matrix, roc_curve

# import sys
# sys.path.append('../../')
# from utils.io import sys_home

add_package_path()

from facebias.iotools import load_bfw_datatable

global_threshold = 0.6

dir_data = "../../data/"
dir_features = f"{dir_data}features/senet50/"
f_datatable = f"{dir_data}bfw-datatable.pkl"

data = load_bfw_datatable(f_datatable, default_score_col="senet50")
data["yp0"] = (data["score"] > global_threshold).astype(int)

fpath = f"{dir_data}/interm/thresholds.pkl"
threholds = pd.read_pickle(fpath)
threholds.optimal_threshold = threholds.loc[
    threholds.optimal_threshold == None, "optimal_threshold"
] = global_threshold
data["yp3"] = (data["score"] > threholds["optimal_threshold"]).astype(int)

target_far_values = np.array([0.3, 0.1, 0.01, 0.001, 0.0001])

tags = [
    "\\textbf{\gls{af}}",
    "\\textbf{\gls{am}}",
    "\\textbf{\gls{bf}}",
    "\\textbf{\gls{bm}}",
    "\\textbf{\gls{if}}",
    "\\textbf{\gls{im}}",
    "\\textbf{\gls{wf}}",
    "\\textbf{\gls{wm}}",
]
strout = "{0} & {1:.3f} & {2:.3f} & {3:.3f} & {4:.5f} &  {5:.3f} & {6}\\\\"
toprint = []
toprint1 = []
means = []
means1 = []
for i, att1 in enumerate(np.unique(data.att1)):
    pairs = data.loc[data.att1 == att1]
    # pairs.y0 == pairs.label
    fpr, tpr, thresholds = roc_curve(pairs.label.astype(int), pairs.score)
    confusion = confusion_matrix(pairs.label.astype(int).values, pairs.yp0.values)

    tn, fp, fn, tp = np.hstack(confusion)
    acc = (tp + tn) / (tn + tp + fn + fp)
    tars = []
    tars1 = []

    for th in target_far_values:
        yp0 = pairs.loc[pairs.yp0 == 0, "score"] > th
        yp3 = pairs.loc[pairs.yp3 == 0, "score"] > th
        # bob.measure.eer()
        acc0 = 1.0 - float(sum(yp0) / len(pairs.loc[pairs.yp0 == 0]))
        acc4 = 1.0 - float(sum(yp3) / len(pairs.loc[pairs.yp3 == 0]))

        tars.append(acc0)
        tars1.append(acc4)

    means.append(tars)
    means1.append(tars1)

    toprint.append(
        strout.format(tags[i], tars[0], tars[1], tars[2], tars[3], tars[4], "--")
    )  # tars[3]))
    toprint1.append(
        strout.format(tags[i], tars1[0], tars1[1], tars1[2], tars[3], tars[4], "--")
    )  # , tars1[3]))

m1 = np.array(means).mean(axis=0)
m2 = np.array(means1).mean(axis=0)
toprint.append(
    strout.format("Avg.", m1[0], m1[1], m1[2], m1[3], m1[4], "--")
)  # tars[3]))
toprint1.append(
    strout.format("Avg.", m2[0], m2[1], m2[2], m2[3], m2[4], "--")
)  # tars[3]))
for p in toprint:
    print(p)
# print(toprint + '\\\\')
# [print('{0:.3f}'.format(m*100)) for m in means]
# print('{0:.3f}'.format(np.mean(means)*100))
print()
print()
for p in toprint1:
    print(p)

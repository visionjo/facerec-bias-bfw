"""
Calculate TAR for specific FAR values. Spit out table organized by subgroup.
"""
import numpy as np
import pandas as pd
from bob import measure
from facebias.utils import add_package_path

# import sys
# sys.path.append('../../')
# from utils.io import sys_home

add_package_path()

from facebias.iotools import (
    load_bfw_datatable,
)

global_threshold = 0.6

df = pd.read_pickle('/Users/jrobby/WORK/src/facebias/data/interm/meta.pkl')
lut_thresholds = {}
for meta in df:
    lut_thresholds["".join(meta[0])] = meta[1]
dir_data = "../../data/"
dir_features = f"{dir_data}features/senet50/"
f_datatable = f"{dir_data}bfw-datatable.pkl"

data = load_bfw_datatable(f_datatable)
data["score"] = data["senet50"]
data["yp0"] = (data["score"] > global_threshold).astype(int)

fpath = f'{dir_data}/interm/thresholds.pkl'
threholds = pd.read_pickle(fpath)
threholds.optimal_threshold = threholds.loc[
    threholds.optimal_threshold == None, 'optimal_threshold'] = global_threshold

data['yp3'] = (data["score"] > threholds['optimal_threshold']).astype(int)

target_far_values = np.array([0.3, 0.1, 0.01, 0.001, 0.0001])

means = []
means1 = []
for i, att1 in enumerate(np.unique(data.a1)):
    pairs = data.loc[data.a1 == att1]
    negatives = pairs.loc[pairs.label == 0, 'score'].values
    positives = pairs.loc[pairs.label == 1, 'score'].values
    positives.sort()
    negatives.sort()
    for th in target_far_values:
        thr = measure.far_threshold(negatives, positives, global_threshold,
                                    is_sorted=True)
        
        far = measure.farfrr(negatives, positives, thr)
        print(f"{far} @ {th}")

        far = measure.farfrr(negatives, positives,
                             measure.far_threshold(negatives, positives,
                                                   lut_thresholds[att1]))
        measure.false_alarm_rate(negatives, positives)
        thr = measure.far_threshold(negatives, positives, lut_thresholds[att1])
        far = measure.farfrr(negatives, positives, thr)
        print(f"{far} @ {th} {thr}")
        print("")
        print("")
    # pairs.y0 == pairs.label
    # fpr, tpr, thresholds = roc_curve(pairs.label.astype(int), pairs.score)
    # confusion = confusion_matrix(pairs.label.astype(int).values,
    #                              pairs.yp0.values)

    # tn, fp, fn, tp = np.hstack(confusion)
    # acc = (tp + tn) / (tn + tp + fn + fp)
    # tars = []
    # tars1 = []
    #
    # for th in target_far_values:
    #     yp0 = pairs.loc[pairs.yp0 == 0, "score"] > th
    #     yp3 = pairs.loc[pairs.yp3 == 0, "score"] > th
    #     # bob.measure.eer()
    #     acc0 = 1.0 - float(sum(yp0) / len(pairs.loc[pairs.yp0 == 0]))
    #     acc4 = 1.0 - float(sum(yp3) / len(pairs.loc[pairs.yp3 == 0]))
    #
    #     tars.append(acc0)
    #     tars1.append(acc4)
    #
    # means.append(tars)
    # means1.append(tars1)

# m1 = np.array(means).mean(axis=0)
# m2 = np.array(means1).mean(axis=0)

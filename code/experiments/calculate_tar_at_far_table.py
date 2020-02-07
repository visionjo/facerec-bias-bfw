"""
Calculate TAR for specific FAR values. Spit out table organized by subgroup.
"""
import numpy as np
import pandas as pd
from facebias.metrics import calculate_tar_and_far_values
from facebias.utils import add_package_path, find_best_threshold
from sklearn.metrics import confusion_matrix, roc_curve

# import sys
# sys.path.append('../../')
# from utils.io import sys_home

add_package_path()

from facebias.iotools import load_bfw_datatable

dir_data = "../../data/"
dir_features = f"{dir_data}features/senet50/"
f_datatable = f"{dir_data}bfw-datatable.pkl"
f_threshold = f"{dir_data}/interm/thresholds.pkl"

thresholds_arr = np.linspace(0.4, 0.7, 500)
global_threshold = []
data = load_bfw_datatable(f_datatable, default_score_col="senet50")

data["yp0"] = 0

folds = data.fold.unique()

for fold in folds:
    ids = data.fold != fold
    threshold, score = find_best_threshold(
        thresholds_arr, data.loc[ids, ["label", "score"]]
    )
    print(threshold, score)
    data.loc[ids, "yp0"] = (data["score"] >= threshold).astype(int)
    global_threshold.append(threshold)

# Read in data
df = pd.read_pickle(f"{dir_data}interm/meta.pkl")
lut_thresholds = {}
for meta in df:
    lut_thresholds["".join(meta[0])] = meta[1]

threholds = pd.read_pickle(f_threshold)
threholds.optimal_threshold = threholds.loc[
    threholds.optimal_threshold is None, "optimal_threshold"
] = np.mean(global_threshold)

# threshold scores and store decision as int (1 or 0)

data["yp3"] = (data["score"] > threholds["optimal_threshold"]).astype(int)


def evaluate_tar_at_far_values(scores, fars=np.array([0.3, 0.1, 0.01, 0.001, 0.0001])):
    for th in fars:
        yp0_fp = sum((scores > th).astype(int))
        # bob.measure.eer()
        acc0 = 1.0 - float(yp0_fp / sum(data.loc[data.yp0 == 0]))
        # acc4 = 1.0 - float(yp3_fp / len(data.loc[data.yp3 == 0]))

        tars.append(acc0)
        # tars1.append(acc4)
    return tars, fars


target_far_values = np.array([0.3, 0.1, 0.01, 0.001, 0.0001])
means = []
means1 = []
attributes = np.unique(data.a1)
tar_table = pd.DataFrame(
    np.zeros((len(attributes), len(attributes))), columns=attributes, index=attributes
)


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


for th in target_far_values:
    for fold in folds:
        df_fold = data.loc[data.fold == fold]
        for i, attribute in enumerate(attributes):
            df_attribute = df_fold.loc[df_fold.a1 == attribute]
            y_true = df_attribute.label.values.astype(int)
            yp0 = (df_attribute["score"] > global_threshold[0]).astype(int)

            far, tar, thresh = calculate_tar_and_far_values(
                y_true, scores=df_attribute["score"]
            )

            ids = np.nanargmin(np.absolute(far - th))
            # negatives = pairs.loc[pairs.label == 0, "score"].values
            # positives = pairs.loc[pairs.label == 1, "score"].values
            # positives.sort()
            # negatives.sort()
            tars = []
            tars1 = []
            n_positive = sum(df_attribute.label == 1)
            n_negative = sum(df_attribute.label == 0)

            confusion = confusion_matrix(y_true, yp0)

            fpr, tpr, threshold_arr = roc_curve(y_true, df_attribute["score"])

            far_values = fpr / n_negative

            ids_far = np.argmin(np.abs(far_values - find_nearest(far_values, th)))

            tn, fp, fn, tp = np.hstack(confusion)
            far = fp / n_negative

            tar = tp / n_positive
            acc = (tp + tn) / (tn + tp + fn + fp)

            # for th in target_far_values:
            yp0_fp = sum(
                (df_attribute.loc[df_attribute.yp0 == 0, "score"] > th).astype(int)
            )
        # yp3_fp = sum((pairs.loc[pairs.yp3 == 0, "score"] > th).astype(int))
        # bob.measure.eer()
        acc0 = 1.0 - 1.0 * yp0_fp / df_attribute.loc[df_attribute.yp0 == 0].shape[0]
        # acc4 = 1.0 - float(yp3_fp / len(pairs.loc[pairs.yp3 == 0]))

        tars.append(acc0)
        # tars1.append(acc4)

        # far = measure.farfrr(negatives, positives,
        #                      measure.far_threshold(negatives, positives,
        #                                            lut_thresholds[att1]))
        # measure.false_alarm_rate((negatives, positives),
        #                          lut_thresholds[att1])
        # thr = measure.far_threshold(negatives, positives,
        #                             lut_thresholds[att1])

        # tar = sum(
        #     (pairs.loc[pairs.yp0 == 0, "score"] > thr).astype(int)) / len(
        #     pairs.loc[pairs.yp0 == 0, "score"])

        # far = measure.farfrr(negatives, positives, thr)
        # print(f"{far} @ {th} {thr}")
        # print("")
        # print("")

        means.append(tars)
        means1.append(tars1)

        # m1 = np.array(means).mean(axis=0)
        # m2 = np.array(means1).mean(axis=0)

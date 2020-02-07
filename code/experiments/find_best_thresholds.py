"""
Find the optimal thresholds per subgroup
"""
import numpy as np
import pandas as pd

from facebias.iotools import load_bfw_datatable
from facebias.utils import find_best_threshold

bfw_version = "0.1.5"
dir_data = "../../data/bfw/"

dir_features = f"{dir_data}features/sphereface/"
dir_meta = f"{dir_data}meta/"
f_datatable = f"{dir_meta}bfw-v{bfw_version}-datatable.pkl"
f_threshold = f"{dir_meta}thresholds.pkl"

thresholds_arr = np.linspace(0.18, 0.4, 5000)
global_threshold = []
data = load_bfw_datatable(f_datatable, default_score_col="sphereface")

data["yp0"] = 0

folds = data.fold.unique()
for fold in folds:
    ## Determine global thresholds and scores per fold
    test_ids = data.fold == fold
    threshold, score = find_best_threshold(
        thresholds_arr, data.loc[test_ids == False, ["label", "score"]]
    )
    print(f"Fold {fold}: with t_g={threshold}, acc={score}")
    data.loc[test_ids, "yp0"] = (data.loc[test_ids, "score"] >= threshold).astype(int)
    global_threshold.append(threshold)

data["iscorrect"] = (data["yp0"] == data["label"]).astype(int)

print(
    pd.DataFrame(
        data.groupby("att1").sum()["iscorrect"]
        / data.groupby("att1").count()["iscorrect"]
    )
)

accuracy = (
        data.groupby("att1").sum()["iscorrect"] / data.groupby("att1").count()["iscorrect"]
).mean()
std = (
        data.groupby("att1").sum()["iscorrect"] / data.groupby("att1").count()["iscorrect"]
).std()

print(f"Accuracy:{accuracy}\nSTD:{std}")
exit()

subgroup_labels, gender_labels, ethnicity_labels = (
    data.a1.unique().astype(str),
    data.g1.unique().astype(str),
    data.e1.unique().astype(str),
)

subgroup_labels.sort()
gender_labels.sort()
ethnicity_labels.sort()

best_thresholds = {}
best_scores = {}
n_pairs = len(data)

print("## All Subgroups ##")
thr = 0.60
data["optimal_threshold"] = 0.0
data["yp1"] = False

for att in subgroup_labels:
    ## Determine global thresholds and scores per fold
    best_thresholds_li = []
    best_scores_li = []
    df_subgroup = data.loc[data.a1 == att]
    att_ids = (data.a1 == att) * (att == data.a2)
    for fold in folds:
        test_ids = df_subgroup.fold == fold
        threshold, score = find_best_threshold(
            thresholds_arr, df_subgroup.loc[test_ids == False, ["label", "score"]]
        )
        print(f"Fold {fold}: with t_g={threshold}, acc={score}")
        df_subgroup.loc[test_ids, "optimal_threshold"] = threshold

        df_subgroup.loc[test_ids, "yp1"] = (df_subgroup["score"] >= threshold).astype(
            int
        )

        best_thresholds_li.append(threshold)
        best_scores_li.append(score)
    data.loc[df_subgroup.index, "yp1"] = df_subgroup["yp1"]
    data.loc[df_subgroup.index, "optimal_threshold"] = df_subgroup["optimal_threshold"]
    best_thresholds[att] = best_thresholds_li
    best_scores[att] = best_scores_li
data["optimal_threshold"] = 0.6
for att in subgroup_labels:
    data.loc[(data.a1 == att) * (data.a2 == att), "optimal_threshold"] = np.mean(
        best_thresholds[att]
    )

data["yp1"] = data["optimal_threshold"] < data["score"]

data["iscorrect1"] = (data["yp1"] == data["label"]).astype(int)
print(
    pd.DataFrame(
        data.groupby("att1").sum()["iscorrect1"]
        / data.groupby("att1").count()["iscorrect1"]
    )
)
accuracy1 = (
        data.groupby("att1").sum()["iscorrect1"]
        / data.groupby("att1").count()["iscorrect1"]
).mean()
std1 = (
        data.groupby("att1").sum()["iscorrect1"]
        / data.groupby("att1").count()["iscorrect1"]
).std()

print(f"Accuracy:{accuracy1}\nSTD:{std1}")
for att, scores in best_scores.items():
    print(
        f"{att}: {np.mean(best_thresholds[att])} +/- {np.std(best_thresholds[att])} at {np.mean(best_thresholds[att])}"
        f" +/-  {np.std(best_thresholds[att])}"
    )

pd.to_pickle(data, "../../data/bfw/meta/datatable-w-threshold.pkl")

meta = np.array(
    [
        (k, np.mean(th), np.mean(s))
        for th, (k, s) in zip(best_thresholds.values(), best_scores.items())
    ]
).reshape(-1, 3)
print(meta)

data.to_pickle(meta, "../../data/bfw/meta/optimal-thresholds.pkl")

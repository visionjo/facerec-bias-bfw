import numpy as np

import facebias.utils as futils
from facebias.iotools import load_bfw_datatable

version = "0.1.5"
dir_data = "../../data/bfw/meta/"
f_datatable = f"{dir_data}bfw-v{version}-datatable.pkl"

thresholds = np.linspace(0.15, 0.6, 1000)
data = load_bfw_datatable(f_datatable, default_score_col="sphereface")

# reorder for find_best_threshold(s
data = data.loc[data.a1 == data.a2]
# data = data[["p1", "p2", "score", "label", "fold"]]
global_threshold, _ = futils.find_best_threshold(
    thresholds, data[["p1", "p2", "score", "label", "fold"]]
)

running_scores = []
data["y1"] = (data["score"] > global_threshold).values.astype(int)
data.fold = data.fold.astype(int)
folds = data.fold.unique()

for k in folds:
    # for each fold
    df_test = data.loc[data.fold == k]
    y_pred = df_test["y1"]
    y_true = df_test["label"].values.astype(int)
    running_scores.append(sum((y_pred == y_true).astype(int)) / len(y_pred))
    print(running_scores[-1])
print(np.mean(running_scores))
data["iscorrect"] = data.y1 == data.label

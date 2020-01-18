import numpy as np

from facebias.utils import add_package_path

add_package_path()
import facebias.utils as fbutils
from facebias.io import load_bfw_datatable

dir_data = "../../data/bfw-data/"
f_datatable = f"{dir_data}bfw-datatable.pkl"

thresholds = np.linspace(0.2, 0.9, 1000)
data = load_bfw_datatable(
    f_datatable, cols=["p1", "p2", "score", "label", "fold", "senet50"]
)
data["score"] = data["senet50"]
del data["senet50"]

# reorder for find_best_threshold(s
data = data[["p1", "p2", "score", "label", "fold"]]

global_threshold, _ = fbutils.find_best_threshold(thresholds, data)

running_scores = []
data["y1"] = (data["score"] > global_threshold).values.astype(int)
for k in range(1, 1 + data.fold.max()):
    # for each fold
    df_test = data.loc[data.fold == k]
    y_pred = df_test["y1"]
    y_true = df_test["label"].values.astype(int)
    running_scores.append(sum((y_pred == y_true).astype(int)) / len(y_pred))
    print(running_scores[-1])
print(np.mean(running_scores))
data["iscorrect"] = data.y1 == data.label

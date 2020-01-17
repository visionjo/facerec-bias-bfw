import pandas as pd
import glob
import numpy as np

from facebias.utils import add_package_path

add_package_path()


from facebias.ioutils import (
    load_bfw_datatable,
    save_bfw_datatable,
    load_features_from_image_list,
)


dir_data = "../../data/bfw-data/"
dir_features = f"{dir_data}features/senet50/"
f_datatable = f"{dir_data}bfw-datatable.pkl"

global_threshold = 0.28

data = load_bfw_datatable(f_datatable)
data["score"] = data["senet50"]

# create ali_images list of all faces (i.e., unique set)
li_images = list(np.unique(data.p1.to_list() + data.p2.to_list()))

# read features as a dictionary, with keys set as the filepath of the image with values set as the face encodings
features = load_features_from_image_list(li_images, dir_features, ext_feat="npy")

running_scores = []
data["y1"] = (data["score"] > global_threshold).values.astype(int)
for k in range(1, 1 + data.fold.max()):
    df_test = data.loc[data.fold == k]
    y_pred = df_test["y1"]
    y_true = df_test["label"].values.astype(int)
    running_scores.append(sum(y_pred == y_true) / len(y_pred))
    print(running_scores[-1])
print(np.mean(running_scores))
data["iscorrect"] = data.y1 == data.label

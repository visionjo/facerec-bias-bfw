import numpy as np
import pandas as pd
from facebias.utils import add_package_path

add_package_path()

dir_data = "../../data/bfw-data/"
dir_features = f"{dir_data}features/senet50/"
f_datatable = f"{dir_data}bfw-v0.1.5-datatable.pkl"

data = pd.read_pickle(f_datatable)
data["score"] = data["senet50"]

thresholds = np.linspace(0.2, 0.9, 5000)

subgroup_labels, gender_labels, ethnicity_labels = (
    data.a1.unique().astype(str),
    data.g1.unique().astype(str),
    data.e1.unique().astype(str),
)
subgroup_labels.sort()
gender_labels.sort()
ethnicity_labels.sort()

best_thresholds = {}
baseline_scores = {}
best_scores = {}
npairs = len(data)

print("## All Subgroups ##")
thr = 0.60
data["optimal_threshold"] = None
for atts in subgroup_labels:
    ids = (data.a1 == atts) & (data.a2 == atts)
    df = data.loc[ids]

    # bs, bt = f oimn
    avg_scores = [((df.score > th) == df.label).mean() for th in thresholds]

    max_ids = np.argmax(avg_scores)
    best_scores[tuple(atts)] = avg_scores[max_ids]
    best_thresholds[tuple(atts)] = thresholds[max_ids]
    data.loc[ids, "optimal_threshold"] = thresholds[max_ids]

    print(
        "{0}-{1}: {2:.4f} with threshold of {3:.4f}".format(
            atts[0], atts[1], avg_scores[max_ids], thresholds[max_ids]
        )
    )
    y_predict = df.score > thr
    avg = (y_predict == df.label).mean()
    print("{0:.4f} with threshold of {1:.4f}".format(avg, thr))
    print()
    baseline_scores[tuple(atts)] = avg

meta = np.array(
    [
        (k, th, s, sc)
        for th, (k, s), sc in zip(
            best_thresholds.values(), best_scores.items(), baseline_scores.values()
        )
    ]
).reshape(-1, 4)
print(meta)
pd.to_pickle(meta, "../../data/interm/meta.pkl")
data.to_pickle("../../data/interm/thresholds.pkl")

for (k, th, s, sc) in meta:
    print("{0}: {2:.4f} (th={1:.5f})\t {3:.5f} (baseline)".format(k, th, s, sc))

# for m in meta:
#     print("{0}:{1}\t{2:.4f}\t{3:.4f}".format(m))


# print(best_thresholds)

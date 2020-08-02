import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from pandas_ml import ConfusionMatrix

##
# Analyze age predictions for the following three groups:
#
# young: a<25
# middle: 24>a<50
# old: a>=50
#
# The analysis considers different sets of demographics as the cohorts of
# interest. For instance, different rates (i.e., confusion metrics) for the
# different age groups across different genders and or ethnicities.

dir_data = Path("").home() / "Dropbox/age-classes"
fn_predictions = ["young-meta.csv", "middle-meta.csv", "old-meta.csv"]

dic_data = {f: pd.read_csv(dir_data / f, delimiter=":") for f in fn_predictions}

df = dic_data[list(dic_data.keys())[0]]

df.columns = ["name", "folder", "category"]
df["actual"] = 0
df = df.loc[df.category != 4]

df["subgroup"] = [
    x[-3].split("_")[0][0].upper() + x[-3].split("_")[1][0].upper()
    for x in df.folder.str.split("/")
]
subgroups = df.subgroup.unique()

tags = df.category.unique()
fig, axs = plt.subplots(int(len(subgroups) / 2), len(tags), figsize=(12, 9))

confusion_matrix = {}
for subgroup, ax in zip(subgroups, axs.ravel()):
    dftemp = df.loc[df.subgroup == subgroup]
    confmat = pd.crosstab(
        dftemp["actual"], dftemp["category"], rownames=["actual"], colnames=["category"]
    )
    confusion_matrix[subgroup] = confmat
    for tag in tags:
        if tag not in confmat.columns:
            confmat[tag] = 0
    sns.heatmap(
        confmat,
        ax=ax,
        annot=True,
        square=True,
        fmt="d",
        xticklabels=["young", "old"],
        cbar=False,
    )
    ax.set_xlabel = ""
    ax.set_ylabel = ""
    # val = confmat[tags[0]].values[0]
plt.show()
print(confusion_matrix)


# sns.heatmap(confusion_matrix, annot=True)
# plt.show()

df_complete = pd.DataFrame(data=None, columns=["file", "subgroup", "y_predict", "y"])
for k, v in dic_data.items():
    df_tmp = pd.DataFrame()
    tag = k.split("-")[0]
    df_tmp["y_predict"] = v[v.columns[2]]
    # df_tmp['file'] = v['folder']
    if "folder" not in v.columns:
        v["folder"] = v[v.columns[[1]]]
        v["name"] = v[v.columns[[0]]]
    df_tmp["file"] = v.apply(
        lambda x: "/".join(x["folder"].split("/")[-3:]) + x["name"], axis=1
    )
    df_tmp["y"] = 0 if tag == "young" else 1 if tag == "middle" else 2
    df_tmp["subgroup"] = df_tmp.apply(
        lambda x: x["file"].split("/")[0].split("_")[0][0].upper()
        + x["file"].split("/")[0].split("_")[0][0].upper(),
        axis=1,
    )
    df_complete = df_complete.append(df_tmp)

df = pd.DataFrame(df_complete, columns=["y", "y_predict"])
Confusion_Matrix = ConfusionMatrix(df["y"], df["y_predict"])
Confusion_Matrix.print_stats()

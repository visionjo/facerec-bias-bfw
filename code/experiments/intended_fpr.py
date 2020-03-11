# Generate DET Curves specific to the different subgroups of the Balance Faces in the Wild (BFW) dataset.

# Uses the data in `data/bfw-datatable.pkl` to evaluate DET curves of different attributes.
import pandas as pd
import numpy as np

# Load out custom tool for loading and processing the data
from facebias.iotools import load_bfw_datatable, makedir

# datatable (See Load Data)
dir_data = "../../data/bfw/"
dir_features = f"{dir_data}features/sphereface/"
f_datatable = f"{dir_data}meta/bfw-v0.1.5-datatable.pkl"
use_feature = "sphereface"

dir_results = f"../../results/{use_feature}/"
makedir(dir_results)

data = load_bfw_datatable(f_datatable, default_score_col=use_feature)
data.head()

classes_abbreviated = np.unique(list(np.unique(data.a1)) + list(np.unique(data.a2)))
classes_abbreviated.sort()

print(f"there are {len(classes_abbreviated)} types: {classes_abbreviated}")

results = {}

subgroups = data.groupby("a1")
li_subgroups = subgroups.groups

#%%

for i, subgroup in enumerate(li_subgroups):
    # for each subgroup
    fout = f"{dir_results}/det_data_{subgroup}.pkl"
    results[fout] = pd.read_pickle(fout)

#%%

subgroups = data.groupby("g1")
li_subgroups = subgroups.groups
classes_abbreviated = list(li_subgroups.keys())
print(f"there are {len(classes_abbreviated)} types: {classes_abbreviated}")

#%%

for i, subgroup in enumerate(li_subgroups):
    # for each subgroup
    fout = f"{dir_results}/det_data_{subgroup}.pkl"
    results[fout] = pd.read_pickle(fout)
#%%

subgroups = data.groupby("e1")
li_subgroups = subgroups.groups
classes_abbreviated = list(li_subgroups.keys())
print(f"there are {len(classes_abbreviated)} types: {classes_abbreviated}")

#%%

for i, subgroup in enumerate(li_subgroups):
    # for each subgroup
    fout = f"{dir_results}/det_data_{subgroup}.pkl"
    results[fout] = pd.read_pickle(fout)

pd.to_pickle(results, 'det-results.pkl')
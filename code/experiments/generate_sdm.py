import matplotlib.pyplot as plt
from facebias.analysis import overlapped_score_distribution
from facebias.iotools import load_bfw_datatable


bfw_version = "0.1.5"
dir_data = "../../data/bfw/"

dir_features = f"{dir_data}features/sphereface/"
dir_meta = f"{dir_data}meta/"
f_datatable = f"{dir_meta}bfw-v{bfw_version}-datatable.pkl"
dir_output = f"{dir_meta}results/"
data = load_bfw_datatable(f_datatable, default_score_col="sphereface")

ax = overlapped_score_distribution(data, log_scale=True, save_figure_path='SDM.pdf', title='Signal Detection Model (Pairwise)')
# plt.show()

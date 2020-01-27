import os
import numpy as np
import pandas as pd
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.style as style
import matplotlib.colors as colors
import seaborn as sns

from os.path import join
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

from facebias.iotools import load_features_from_image_list
from facebias.visualization import draw_det_curve
from facebias.metrics import calculate_det_curves
from facebias.preprocessing import get_attribute_gender_ethnicity, load_image_pair_with_attribute_and_score


def violin_plot(data, save_figure_path=None):
    fontsize = 12
    new_labels = ["Imposter", "Genuine"]
    palette = {new_labels[0]: "orange", new_labels[1]: "lightblue"}

    data["Tag"] = data.label
    data.loc[data.label == 0, "Tag"] = new_labels[0]
    data.loc[data.label == 1, "Tag"] = new_labels[1]

    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    sns.violinplot(x="a1", y="score", hue="Tag", data=data, linewidth=1.25, dodge=True,
                   split=True, palette=palette, ax=ax, scale_hue=True, inner=None)
    plt.xlabel("Subgroup", fontsize=fontsize)
    plt.ylabel("Score", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    plt.legend(loc="best", fontsize=fontsize)
    plt.title("Score Distribution for Genuine and Imposter Pairs Across Subgroup", fontsize=fontsize)

    # save figure
    if save_figure_path is not None:
        plt.savefig(save_figure_path)


def det_plot(data, group_by, plot_title, save_figure_path=None):
    subgroups = data.groupby(group_by)
    li_subgroups = subgroups.groups

    fontsize = 12
    fig, ax = plt.subplots(figsize=(8,8), constrained_layout=True)
    for subgroup in li_subgroups:
        # for each subgroup
        df_subgroup = subgroups.get_group(subgroup)
        labels, scores = df_subgroup["label"].values.astype(int), df_subgroup["score"].values
        fpr, fnr, thresholds = calculate_det_curves(labels, scores)
        ax = draw_det_curve(fpr, fnr, ax=ax, label=subgroup, fontsize=fontsize, title=plot_title)

    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.e"))
    plt.minorticks_off()
    ax.set_ylabel("FNR (%)", fontsize=fontsize)
    ax.set_xlabel("FPR", fontsize=fontsize)
    plt.legend(fontsize=fontsize)
    ax.set_xlim([1e-4, 1])
    ax.set_ylim([0, 30])

    ax.tick_params(axis="both", labelsize=fontsize)

    # save figure
    if save_figure_path is not None:
        plt.savefig(save_figure_path)


def plot_confusion_matrix(df, save_figure_path=None):
    # plot confusion matrix in heatmap format
    fig, ax = plt.subplots(figsize=(9, 9))

    # set color scheme and style
    cmap = colors.LinearSegmentedColormap.from_list("nameofcolormap", ["w", "b"], gamma=2.0)
    style.use("seaborn-paper")  # sets the size of the charts
    sns.set_style({"xtick.bottom": True}, {"ytick.left": True})

    # plot heatmap using seaborn
    ax = sns.heatmap(df, annot=True, linewidths=.1, square=True, cmap=cmap,
                     cbar_kws={"shrink": .7, "ticks": [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]},
                     linecolor="black", ax=ax, fmt=".2f", annot_kws={"size": 14}, cbar=True)

    # add the column names as labels, set fontsize and set title
    fontsize = 14
    ax.set_yticklabels(df.columns, rotation=0, fontsize=fontsize)
    ax.set_xticklabels(df.columns, fontsize=fontsize)
    ax.axhline(y=0, color='k', linewidth=2)
    ax.axhline(y=df.shape[1], color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2)
    ax.axvline(x=df.shape[0], color='k', linewidth=2)
    ax.set_title("Rank 1 (%) Error", fontsize=fontsize)

    # save figure
    if save_figure_path is not None:
        plt.savefig(save_figure_path)


def confusion_matrix(image_list_path, embedding_dir_path, save_figure_path=None):
    data = pd.read_csv(image_list_path)
    image_list = data["path"].to_list()
    feature = load_features_from_image_list(image_list, embedding_dir_path, ext_feat="npy")
    data = get_attribute_gender_ethnicity(data, "path")
    data["id"] = data["path"].apply(lambda x: "/".join(x.split("/")[:-1])).astype("category")
    score_matrix = cosine_similarity(data["path"].apply(lambda x: feature[x][0]).to_list())
    score_matrix[np.eye(len(score_matrix)) == 1] = 0

    data["tag"] = LabelEncoder().fit_transform(data["id"])
    data["nn_index"] = np.argmax(score_matrix, axis=1)  # nearest neighbor index
    data["nn"] = data.loc[data["nn_index"]]["tag"].to_list()  # nearest neighbor
    data["nn_type"] = data.loc[data["nn_index"]]["a"].to_list()
    data["wrong_nn"] = (data["tag"] != data["nn"]).astype(int)

    # construct confusion matrix
    conf = data.groupby(by=["a", "nn_type"])["wrong_nn"].sum()
    confusion_npy = conf.values.reshape(1, -1)
    confusion_npy[np.isnan(confusion_npy)] = 0
    confusion_npy = confusion_npy.reshape((8, -1))
    all_subgroup = data["a"].unique()
    confusion_df = pd.DataFrame(confusion_npy, index=all_subgroup, columns=all_subgroup)

    n_samples_per_subgroup = data["a"].count() / len(all_subgroup)
    confusion_percent_error_df = (confusion_df / n_samples_per_subgroup) * 100
    plot_confusion_matrix(confusion_percent_error_df, save_figure_path)


def create_bias_analysis_plots(image_pair_path, image_list_path, embedding_dir_path, processed_data=None,
                               save_processed_data=None, save_figure_dir="results"):
    if processed_data is not None:
        with open(processed_data, "rb") as f:
            data_pair_df = pk.load(f)
    else:
        data_pair_df = load_image_pair_with_attribute_and_score(image_pair_path, embedding_dir_path)
        if save_processed_data is not None:
            Path(os.path.dirname(save_processed_data)).mkdir(parents=True, exist_ok=True)
            with open(save_processed_data, "wb") as f:
                pk.dump(data_pair_df, f)

    # before saving figure, create nested directories if necessary
    Path(save_figure_dir).mkdir(parents=True, exist_ok=True)
    violin_plot(data_pair_df, join(save_figure_dir, "score_dist_violin.png"))
    det_plot(data_pair_df, "a1", "DET Curve Per Ethnicity and Gender", join(save_figure_dir, "det_subgroup.png"))
    det_plot(data_pair_df, "g1", "DET Curve Per Gender", join(save_figure_dir, "det_gender.png"))
    det_plot(data_pair_df, "e1", "DET Curve Per Ethnicity", join(save_figure_dir, "det_ethnicity.png"))
    confusion_matrix(image_list_path, embedding_dir_path, join(save_figure_dir, "confusion_matrix.png"))

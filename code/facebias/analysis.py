import argparse
import os
import pickle as pk
from os.path import join
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.style as style
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from facebias.iotools import load_features_from_image_list
from facebias.metrics import calculate_det_curves
from facebias.preprocessing import (
    get_attribute_gender_ethnicity,
    load_image_pair_with_attribute_and_score,
)
from facebias.utils import replace_ext
from facebias.visualization import draw_det_curve
from matplotlib.lines import Line2D
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


def violin_plot(data, save_figure_path=None):
    """
    Plot a violin plot of the distribution of the cosine similarity score of
    impostor pairs (different people) and genuine pair (same people) the plots
    are separated by ethnicity-gender attribute of the first person of each pair
    The final plot is saved to 'save_figure_path'

    Parameters
    ----------
    data:   pandas.DataFrame that contains column 'p1', 'p2', 'a1', 'a2',
            'score', and 'label' 'p1' and 'p2' are the pair of images. 'a1' and
            'a2' are the abbreviated attribute of 'p1' and 'p2' respectively.
            'score' is the cosine similarity score between 'p1' and 'p2',
            'label' is a binary indicating whether 'p1' and
            'p2' are the same person
    save_figure_path:   path to save the resulting violin plot. will not save is
                        the value is None
    """
    fontsize = 12
    new_labels = ["Imposter", "Genuine"]
    palette = {new_labels[0]: "orange", new_labels[1]: "lightblue"}

    data["Tag"] = data["label"]
    data.loc[data["label"] == 0, "Tag"] = new_labels[0]
    data.loc[data["label"] == 1, "Tag"] = new_labels[1]

    fig, ax = plt.subplots(1, 1, figsize=(13, 7))
    sns.violinplot(
        x="a1",
        y="score",
        hue="Tag",
        data=data,
        linewidth=1.25,
        dodge=True,
        split=True,
        palette=palette,
        ax=ax,
        scale_hue=True,
        inner=None,
    )
    plt.xlabel("Subgroup", fontsize=fontsize)
    plt.ylabel("Score", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    plt.legend(loc="best", fontsize=fontsize)
    plt.title(
        "Score Distribution for Genuine and Imposter Pairs Across Subgroup",
        fontsize=fontsize,
    )

    # save figure
    if save_figure_path is not None:
        plt.savefig(save_figure_path)


def overlapped_score_distribution(data, log_scale=False, save_figure_path=None):
    """
    Plot the score distribution of the cosine similarity score of impostor pairs
    (different people) and
    Genuine pair (same people) the plots are separated by ethnicity-gender
    attribute of the first person of each pair.
    Curves of different ethnicity-gender attribute are distinguished by colors.
    The final plot is saved to 'save_figure_path'

    Parameters
    ----------
    data:               pandas.DataFrame that contains column 'p1', 'p2',
                        'att1', 'att2', 'score', and 'label'
                        'p1' and 'p2' are the pair of images.
                        'att1' and 'att2' are the abbreviated attribute
                        (ethnicity-gender) of 'p1' and 'p2' respectively.
                        'score' is the cosine similarity score between 'p1' and
                        'p2', 'label' is a binary indicating whether 'p1' and
                        'p2' are the same person
    log_scale:          boolean indicating whether to use log scale on y axis
    save_figure_path:   path to save the resulting score distribution plot. will
                        not save is the value is None
    """
    # set figure size
    plt.figure(figsize=(20, 10))

    # set color scheme and font size
    att_to_color = {
        "AM": "blue",
        "AF": "orange",
        "IM": "green",
        "IF": "red",
        "BM": "Purple",
        "BF": "brown",
        "WM": "hotpink",
        "WF": "black",
    }
    fontsize = 14

    # plot distribution for each ethnicity-gender attribute
    for att in [f"{e}{g}" for e in ["A", "I", "B", "W"] for g in ["M", "F"]]:
        data_att = data.loc[data["a1"] == att]

        # plot intra score
        sns.distplot(
            data_att.loc[data_att["label"] == 1]["score"],
            hist=False,
            label=att,
            color=att_to_color[att],
        )
        # plot inter score
        sns.distplot(
            data_att.loc[data_att["label"] == 0]["score"],
            hist=False,
            color=att_to_color[att],
            kde_kws={"linestyle": "--"},
        )

    # set label and font sizes
    plt.xlabel("Cosine Similarity Score", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # create legend
    color_legend = plt.legend(fontsize=fontsize)
    solid_line = Line2D([0], [0], color="black", linestyle="-")
    dash_line = Line2D([0], [0], color="black", linestyle="--")
    plt.legend([solid_line, dash_line], ["intra", "inter"], fontsize=fontsize,
               loc=2)
    plt.gca().add_artist(color_legend)

    # handle log scale
    if log_scale:
        title = "Score Distribution Log Scale"
        plt.semilogy()
        plt.ylim([10 ** (-5), 10])
    else:
        title = "Score Distribution"

    # set title
    plt.title(title, fontsize=fontsize)

    # save figure
    if save_figure_path is not None:
        plt.savefig(save_figure_path)


def overlapped_score_distribution(data, log_scale=False, save_figure_path=None):
    """
    Plot the score distribution of the cosine similarity score of impostor pairs
    (different people) and Genuine pair (same people) the plots are separated by
    ethnicity-gender attribute of the first person of each pair. Curves of
    different ethnicity-gender attribute are distinguished by colors.

    The final plot is saved to 'save_figure_path'

    Parameters
    ----------
    data:       pandas.DataFrame that contains column 'p1', 'p2', 'att1',
                'att2', 'score', and 'label'
                'p1' and 'p2' are the pair of images. 'att1' and 'att2' are the
                abbreviated attribute (ethnicity-gender) of 'p1' and 'p2'
                respectively. 'score' is the cosine similarity score between
                'p1' and 'p2', 'label' is a binary indicating whether 'p1' and
                'p2' are the same person
    log_scale:          boolean indicating whether to use log scale on y axis
    save_figure_path:   path to save the resulting score distribution plot. will
                        not save is the value is None
    """
    # set figure size
    plt.figure(figsize=(20, 10))

    # set color scheme and font size
    att_to_color = {
        "AM": "blue",
        "AF": "orange",
        "IM": "green",
        "IF": "red",
        "BM": "Purple",
        "BF": "brown",
        "WM": "hotpink",
        "WF": "black",
    }
    fontsize = 14

    # plot distribution for each ethnicity-gender attribute
    for att in [f"{e}{g}" for e in ["A", "I", "B", "W"] for g in ["M", "F"]]:
        data_att = data.loc[data["a1"] == att]

        # plot intra score
        sns.distplot(
            data_att.loc[data_att["label"] == 1]["score"],
            hist=False,
            label=att,
            color=att_to_color[att],
        )
        # plot inter score
        sns.distplot(
            data_att.loc[data_att["label"] == 0]["score"],
            hist=False,
            color=att_to_color[att],
            kde_kws={"linestyle": "--"},
        )

    # set label and font sizes
    plt.xlabel("Cosine Similarity Score", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    # create legend
    color_legend = plt.legend(fontsize=fontsize)
    solid_line = Line2D([0], [0], color="black", linestyle="-")
    dash_line = Line2D([0], [0], color="black", linestyle="--")
    plt.legend([solid_line, dash_line], ["intra", "inter"], fontsize=fontsize,
               loc=2)
    plt.gca().add_artist(color_legend)

    # handle log scale
    if log_scale:
        title = "Score Distribution Log Scale"
        plt.semilogy()
        plt.ylim([10 ** (-5), 10])
    else:
        title = "Score Distribution"

    # set title
    plt.title(title, fontsize=fontsize)

    # save figure
    if save_figure_path is not None:
        plt.savefig(save_figure_path)


def det_plot(data, group_by, plot_title, save_figure_path=None):
    """
    Plot Detection Error Tradeoff (DET) curves. Each curve use data separated by
    column 'group_by' in 'data'.

    Each DET curve is created by varying the threshold of the cosine similarity
    score which locates in 'score' column of 'data'. The final plot has title
    'plot_title' and is saved to 'save_figure_path'

    Parameters
    ----------
    data:               pandas.DataFrame that contains column 'p1', 'p2',
                        'score', and 'label'. 'p1' and 'p2' are the
                        pair of images. 'score' is the cosine similarity score
                        between 'p1' and 'p2'
                        'label' is a binary indicating whether 'p1' and 'p2'
                        are the same person
    group_by:           the column in 'data' that we want to use as a separator
    plot_title:         the title of the plot
    save_figure_path:   path to save the resulting det plot. will not save is
                        the value is None
    """
    subgroups = data.groupby(group_by)
    li_subgroups = subgroups.groups

    fontsize = 12
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    for subgroup in li_subgroups:
        # for each subgroup
        df_subgroup = subgroups.get_group(subgroup)
        labels, scores = (
            df_subgroup["label"].values.astype(int),
            df_subgroup["score"].values,
        )
        fpr, fnr, thresholds = calculate_det_curves(labels, scores)
        ax = draw_det_curve(
            fpr, fnr, ax=ax, label=subgroup, fontsize=fontsize, title=plot_title
        )

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


def plot_confusion_matrix(data, save_figure_path=None):
    """
    Using the values from 'df', visualize confusion matrix by varying color
    intensity of each cell based on its value.

    The resulting plot is saved to 'save_figure_path'

    Parameters
    ---------
    data:               pandas.DataFrame that contains values of all cells in
                        the confusion matrix we want to plot
    save_figure_path:   path to save the resulting confusion matrix plot. will
                        not save is the value is None
    """
    # plot confusion matrix in heatmap format
    fig, ax = plt.subplots(figsize=(9, 9))

    # set color scheme and style
    cmap = colors.LinearSegmentedColormap.from_list(
        "nameofcolormap", ["w", "b"], gamma=2.0
    )
    style.use("seaborn-paper")  # sets the size of the charts
    sns.set_style({"xtick.bottom": True}, {"ytick.left": True})

    # plot heatmap using seaborn
    ax = sns.heatmap(
        data,
        annot=True,
        linewidths=0.1,
        square=True,
        cmap=cmap,
        cbar_kws={"shrink": 0.7, "ticks": [0.0, 2.5, 5.0, 7.5, 10.0, 12.5]},
        linecolor="black",
        ax=ax,
        fmt=".2f",
        annot_kws={"size": 14},
        cbar=True,
    )

    # add the column names as labels, set fontsize and set title
    fontsize = 14
    ax.set_yticklabels(data.columns, rotation=0, fontsize=fontsize)
    ax.set_xticklabels(data.columns, fontsize=fontsize)
    ax.axhline(y=0, color="k", linewidth=2)
    ax.axhline(y=data.shape[1], color="k", linewidth=2)
    ax.axvline(x=0, color="k", linewidth=2)
    ax.axvline(x=data.shape[0], color="k", linewidth=2)
    ax.set_title("Rank 1 (%) Error", fontsize=fontsize)

    # save figure
    if save_figure_path is not None:
        plt.savefig(save_figure_path)


def confusion_matrix(im_paths, dir_embeddings, save_figure_path=None):
    """
    Plot rank-1 nearest neighbor confusion matrix. Rows and columns are
    different ethnicity-gender. The value in row x and column y is the error
    rate that each image of ethnicity-gender x and its rank-1 nearest neighbor
    of ethnicity-gender y is not the same person

    Parameters
    ----------
    im_paths:       path to the csv file that contains list of images of
                    interest. The csv must contain column 'path' that contains
                    relative paths to images
    dir_embeddings:     path to root directory that contains the embeddings.
    save_figure_path:       path to save the resulting confusion matrix plot.
                            will not save is the value is None
    """
    data = pd.read_csv(im_paths)
    image_list = data["path"].to_list()
    feature = load_features_from_image_list(image_list, dir_embeddings,
                                            ext_feat="npy")
    data = get_attribute_gender_ethnicity(data, "path")
    data["id"] = (
        data["path"].apply(lambda x:
                           "/".join(x.split("/")[:-1])).astype("category")
    )
    score_matrix = cosine_similarity(
        data["path"].apply(lambda x: feature[x][0]).to_list()
    )
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
    confusion_df = pd.DataFrame(confusion_npy, index=all_subgroup,
                                columns=all_subgroup)

    n_samples_per_subgroup = data["a"].count() / len(all_subgroup)
    confusion_percent_error_df = (confusion_df / n_samples_per_subgroup) * 100
    plot_confusion_matrix(confusion_percent_error_df, save_figure_path)


def create_bias_analysis_plots(
        im_pair_paths,
        im_paths,
        dir_embeddings,
        data=None,
        save_data=None,
        dir_output="results",
):
    """
    Using image pairs from 'image_pair_path', plot the following three plots.

    Violin plot - the distribution of the cosine similarity score of impostor
    pairs (different people) and Genuine pair (same people) the plots are
    separated by ethnicity-gender attribute of the first person of each pair.
    Overlapped Score Distribution plot - similar to violin plot, but overlap the
    curves of different ethnicity-gender attribute all on top of each other.
    DET plots - Detection Error Trade-off curves. Each DET curve is created by
    varying the threshold of the cosine similarity score between image pairs.

    Curves will be separated by three methods.
            i)      by gender
            ii)     by ethnicity
            iii)    by ethnicity-gender

    Using the list of image from 'image_list_path', plot the following plot.

    Confusion Matrix - rank-1 nearest neighbor confusion matrix when row and
    column are labeled by ethnicity-gender.

    Parameters
    ----------
    im_pair_paths:        path to csv file with all image pairs .
    im_paths:        path to csv file with list of images .
    dir_embeddings:     path to csv file with the embeddings. in the root
                            directories must exist subdirectories with name
                            {ethnicity}_{gender}s, each of which contain person
                            id subdirectories.
    data:         path to the saved processed dataframe that contain
                            attributes, person unique id, and scores
    save_data:    path to save intermediate processed data (with
                            attributes, person unique id, and scores). will not
                            save if the value is None.
    dir_output:        path to save the resulting figures.
    """
    if data is not None:
        print("load processed data")
        with open(data, "rb") as f:
            data_pair_df = pk.load(f)
    else:
        print(
            "get processed data (adding attribute, assigning unique person id, "
            "and calculating cosine similarity score)"
        )
        data_pair_df = load_image_pair_with_attribute_and_score(
            im_pair_paths, dir_embeddings
        )
        if save_data is not None:
            Path(os.path.dirname(save_data)).mkdir(parents=True, exist_ok=True)
            with open(save_data, "wb") as f:
                pk.dump(data_pair_df, f)

    # before saving figure, create nested directories if necessary
    Path(dir_output).mkdir(parents=True, exist_ok=True)

    violin_path = join(dir_output, "score_dist_violin.png")
    print(f"producing violin plot. result will be saved to {violin_path}")
    violin_plot(data_pair_df, save_figure_path=violin_path)

    over_dist_path = join(dir_output, "overlapped_score_dist.png")
    print(
        f"producing overlapped score distribution plot. result will be saved to"
        f" {over_dist_path}"
    )
    overlapped_score_distribution(
        data_pair_df, log_scale=False, save_figure_path=over_dist_path
    )

    log_over_dist_path = join(dir_output, "overlapped_log_scale_score_dist.png")
    print(
        f"producing overlapped score distribution plot on log scale. "
        f"result will be saved to {log_over_dist_path}"
    )
    overlapped_score_distribution(
        data_pair_df, log_scale=True, save_figure_path=log_over_dist_path
    )

    det_subgroup_path = join(dir_output, "det_subgroup.png")
    print(
        f"producing DET curve separated by ethnicity-gender. result will be "
        f"saved to {det_subgroup_path}"
    )
    det_plot(
        data_pair_df,
        "a1",
        "DET Curve Per Ethnicity and Gender",
        save_figure_path=det_subgroup_path,
    )

    det_gender_path = join(dir_output, "det_gender.png")
    print(
        f"producing DET curve separated by gender. result will be saved to "
        f"{det_gender_path}"
    )
    det_plot(
        data_pair_df, "g1", "DET Curve Per Gender",
        save_figure_path=det_gender_path
    )

    det_ethnicity_path = join(dir_output, "det_ethnicity.png")
    print(
        f"producing DET curve separated by ethnicity. result will be saved to "
        f"{det_ethnicity_path}"
    )
    det_plot(
        data_pair_df,
        "e1",
        "DET Curve Per Ethnicity",
        save_figure_path=det_ethnicity_path,
    )

    confusion_matrix_path = join(dir_output, "confusion_matrix.png")
    print(
        f"producing confusion matrix plot. result will be saved to "
        f"{confusion_matrix_path}"
    )
    confusion_matrix(
        im_paths, dir_embeddings, save_figure_path=confusion_matrix_path)


def clean_image_pair_and_image_list_csv(pair_paths, im_paths, dir_embeddings):
    """
    Clean image pair csv and image list csv by deleting the rows that contain a
    path to an image whose embedding does not exist in embedding_dir_path

    parameters
    ----------
    pair_paths:     path to csv file with image pairs
    im_paths:     path to csv file with list of images
    dir_embeddings:  path to csv file with the embeddings.
                    in the root directories must exist subdirectories with name
                    {ethnicity}_{gender}s, each with person id subdirectories

    returns
    -------
    image_pair_path:     path to csv file with updated image pairs
    image_list_path:     path to csv file with updated image list
    """
    check_exist = lambda rel_path: os.path.exists(
        os.path.join(dir_embeddings, replace_ext(rel_path))
    )
    # clean image pair csv
    image_pair = pd.read_csv(pair_paths)
    old_nrow = image_pair.shape[0]
    image_pair = image_pair[
        image_pair["p1"].map(check_exist) & image_pair["p2"].map(check_exist)
        ]
    new_nrow = image_pair.shape[0]
    print(
        f"For image pair csv, {old_nrow - new_nrow} rows out of {old_nrow} rows"
        f" were deleted ({100 * (1 - new_nrow / old_nrow):.2f}% of all rows)"
    )
    new_filename = "updated_" + os.path.basename(pair_paths)
    pair_paths = os.path.join(os.path.dirname(pair_paths), new_filename)
    image_pair.to_csv(pair_paths, index=False)

    # clean image list csv
    image_list = pd.read_csv(im_paths)
    old_nrow = image_list.shape[0]
    image_list = image_list[image_list["path"].map(check_exist)]
    new_nrow = image_list.shape[0]
    print(
        f"For image list csv, {old_nrow - new_nrow} rows out of {old_nrow} rows"
        f" were deleted ({100 * (1 - new_nrow / old_nrow):.2f}% of all rows)"
    )
    new_filename = "updated_" + os.path.basename(im_paths)
    im_paths = os.path.join(os.path.dirname(im_paths), new_filename)
    image_list.to_csv(im_paths, index=False)

    return pair_paths, im_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyse bias in face recognition by producing 3 plots to give "
            "more insight. These three plots are violin plot, DET curve, "
            "and confusion matrix plot"
        )
    )
    parser.add_argument(
        "-p",
        "--image_pair_path",
        type=str,
        action="store",
        required=True,
        help="path to the file that contain all image pairs",
    )
    parser.add_argument(
        "-l",
        "--image_list_path",
        type=str,
        action="store",
        required=True,
        help="path to the file that contains list of images",
    )
    parser.add_argument(
        "-e",
        "--embedding_path",
        type=str,
        action="store",
        required=True,
        help="path to the root directory that contains all the embeddings",
    )
    parser.add_argument(
        "-s",
        "--save_path",
        type=str,
        action="store",
        required=True,
        help="path to root directory to save figures and intermediate result",
    )
    parser.add_argument(
        "-d",
        "--processed_data",
        type=str,
        action="store",
        help="path to the dataframe containing attributes and scores",
    )
    parser.add_argument(
        "-c",
        "--clean_image_pair_list",
        action="store_true",
        help=(
            "specified if image pair and image list needs to be modified by "
            "deleting rows with images that we do not have face embedding for"
        ),
    )
    args = parser.parse_args()
    image_pair_path = args.image_pair_path
    image_list_path = args.image_list_path
    embedding_dir_path = args.embedding_path
    clean_image_pair_list = args.clean_image_pair_list

    if clean_image_pair_list:
        print(
            "cleaning image pair and image list csv: delete rows that contain "
            "image paths for which no embedding exists in directory"
        )
        image_pair_path, image_list_path = clean_image_pair_and_image_list_csv(
            image_pair_path, image_list_path, embedding_dir_path
        )
    save_figure_dir = os.path.join(args.save_path, "results")
    processed_data = args.processed_data
    if processed_data is not None:
        save_processed_data = None
    else:
        save_processed_data = os.path.join(args.save_path, "bfw-datatable.pkl")
    # run the whole pipeline
    create_bias_analysis_plots(
        image_pair_path,
        image_list_path,
        embedding_dir_path,
        processed_data,
        save_processed_data,
        save_figure_dir,
    )

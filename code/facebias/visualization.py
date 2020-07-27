"""
Functions for generating plots and visuals.
"""
import numpy as np
import seaborn as sns
from matplotlib import colors as colors, pyplot as plt, style as style, ticker as mtick
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import auc, roc_curve

from facebias.metrics import calculate_det_curves


def set_defaults(
    font={
        "font.family": "serif",
        "font.serif": "Times New Roman",
        "font.color": "darkred",
        "font.weight": "normal",
        "font.size": 16,
    },
    rc={"axes.facecolor": (0, 0, 0, 0)},
    style="white",
    gridstyle="whitegrid",
):
    sns.set(style=style, rc=rc)
    sns.set_style(gridstyle, font)


def box_plot(
    data,
    save_figure_path=None,
    fontsize=12,
    new_labels=("Imposter", "Genuine"),
    figsize=(13, 7),
):
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
    palette = {new_labels[0]: "orange", new_labels[1]: "lightblue"}
    data["Tag"] = data["label"]
    data.loc[data["label"] == 0, "Tag"] = new_labels[0]
    data.loc[data["label"] == 1, "Tag"] = new_labels[1]

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    sns.boxplot(
        x="a1",
        y="score",
        hue="Tag",
        data=data,
        linewidth=1.25,
        dodge=True,
        notch=True,
        palette=palette,
        ax=ax,
    )
    plt.xlabel("Subgroup", fontsize=fontsize)
    plt.ylabel("Score", fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=fontsize)
    plt.legend(loc="best", fontsize=fontsize)
    plt.title(
        "Score Distribution for Genuine and Imposter Pairs Across Subgroup",
        fontsize=fontsize,
    )

    plt.tight_layout()
    # save figure
    if save_figure_path is not None:
        plt.savefig(save_figure_path, transparent=True)


def draw_det_curve(
    fpr,
    fnr,
    ax=None,
    label=None,
    set_axis_log_x=True,
    set_axis_log_y=False,
    scale=100,
    title=None,
    label_x="FPR",
    label_y="FNR (%)",
    ticks_to_use_x=(1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-0),
    ticks_to_use_y=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40),
    fontsize=24,
):
    """
    Generate DET Curve (i.e., FNR vs FPR). It is assumed FPR and FNR is
    increasing and decreasing, respectfully.

    Parameters
    ----------
    fpr: list:  false positive rate
    fnr: list:  false negative rate
    ax: plt.Axes: <default=None>:   Axes object to plot on
    label:  <default=None>
    set_axis_log_x: <default=False>
    set_axis_log_y: <default=False>

    label_x: <default='FPR',>
    label_y: <default='FNR (%)',>
    scale: <default=100>
    title: <default=None>
    ticks_to_use_x: <default=ticks_to_use_x=(1e-4, 1e-3, 1e-2, 1e-1, 1e-0)>
    ticks_to_use_y: <default=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40)>
    fontsize:  <default=24>

    Returns Axes of figure:   plt.Axes()
    -------

    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(fpr, fnr * scale, label=label, linewidth=3)
    if set_axis_log_y:
        ax.set_yscale("log")
    if set_axis_log_x:
        ax.set_xscale("log")

    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticks(ticks_to_use_x)
    ax.set_yticks(scale * np.array(ticks_to_use_y))

    # add 10% to upper ylimit
    ax.set_ylim(0.00, scale * np.max(ticks_to_use_y))
    ax.set_xlim(np.min(ticks_to_use_x), np.max(ticks_to_use_x))
    ax.set_xlabel(label_x, fontsize=fontsize)
    ax.set_ylabel(label_y, fontsize=fontsize)

    ax.legend(loc="best")
    ax.set_title(title, fontsize=fontsize)

    return ax


def generate_roc(
    scores,
    labels,
    fpath="",
    calculate_auc=True,
    add_diag_line=False,
    color="darkorange",
    lw=2,
    label="ROC curve",
    title=None,
):
    """

    Parameters
    ----------
    scores: list    scores of the N pairs (len=N)
    labels: list    boolean labels of the N pairs (len=N)
    fpath:          file-path to save ROC; only saved if arg is passed in
    calculate_auc:  calculate AUC and display in legend of ROC
    add_diag_line:  add ROC curve for random (i.e., diagonal from (0,0) to (1,1)
    color:          color of plotted line
    lw:             Line width of plot
    label:          Legend Label
    title:          Axes title

    Returns Axes of figure:   plt.Axes()
    -------

    """
    fpr, tpr, _ = roc_curve(labels, scores)
    if calculate_auc:
        roc_auc = auc(fpr, tpr)
        label += f"area = {roc_auc}"

    fig, ax = plt.subplots(1)

    plt.plot(fpr, tpr, color=color, lw=lw, label=label)

    if add_diag_line:
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positive Rate")
    if title is not None:
        plt.title(title)
    plt.legend(loc="best")
    if fpath is not None:
        plt.savefig(fpath)

    return ax


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
    plt.legend([solid_line, dash_line], ["intra", "inter"], fontsize=fontsize, loc=2)
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

"""
Functions for generating plots and visuals.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter
from sklearn.metrics import auc, roc_curve


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
    title="Receiver operating characteristic",
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
    # plt.title(title)
    plt.legend(loc="best")
    if fpath is not None:
        plt.savefig(fpath)

    return ax

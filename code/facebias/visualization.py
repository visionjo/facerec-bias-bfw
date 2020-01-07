import seaborn as sns
import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.pyplot as plt


def draw_det_curve(fpr,
                   fnr,
                   ax=None,
                   label=None,
                   set_axis_log_x=True,
                   set_axis_log_y=False,
                   scale=100,
                   title=None,
                   label_x='FPR',
                   label_y='FNR (%)',
                   ticks_to_use_x=(1e-4, 1e-3, 1e-2, 1e-1, 1e-0),
                   ticks_to_use_y=(0.01, 0.03, 0.05, 0.10, 0.20, 0.30, 0.40)
                   ):
    """
    Generate DET Curve (i.e., FNR vs FPR). It is assumed FPR and FNR is increasing and decreasing, respectfully.

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

    sns.lineplot(fpr, fnr * scale, label=label, linewidth=3, ax=ax)
    if set_axis_log_y:
        ax.set_yscale('log')
    if set_axis_log_x:
        ax.set_xscale('log')

    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.get_yaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticks(ticks_to_use_x)
    ax.set_yticks(scale * np.array(ticks_to_use_y))

    # add 10% to upper ylimit
    ax.set_ylim(0.00, scale * np.max(ticks_to_use_y))
    ax.set_xlim(np.min(ticks_to_use_x), np.max(ticks_to_use_x))
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)

    ax.legend(loc='best')
    ax.set_title(title)

    return ax

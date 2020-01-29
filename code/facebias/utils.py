import sys

import numpy as np


def add_package_path(path_package=f"../"):
    if path_package not in sys.path:
        sys.path.append(path_package)


def eval_acc(threshold, predicts):
    """    """

    y_predict = (predicts["score"] > threshold).astype(int)

    return (predicts["label"] == y_predict).mean()


def find_best_threshold(thresholds, predicts, function=eval_acc, find_max=True):
    """
    Determine best threshold as the largest threshold that yields top accuracy.
    Note, tie goes to larger threshold.

    :param find_max:    If true, find largest
    :param function:    Function to base case on, i.e., how to score
    :param thresholds:  threshold values to calculate accuracy with respect to.
    :param predicts:    predictions [p1, p2, score, label], where score and
                        label (index 2 and 3) are used.

    :return:            Threshold value that yielded best accuracy (same type as
                        threshold[threshold.argmax()]).
    """
    assert "label" in predicts
    assert "score" in predicts

    op = np.greater_equal if find_max else np.less_equal
    predicts["label"] = predicts["label"].astype(int)
    best_threshold = best_score = 0
    for threshold in thresholds:

        score = function(threshold, predicts)
        if op(score, best_score):
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


def replace_ext(path):
    """
    Replace the extention (.jpg or .png) of path to .npy

    parameters
    ----------
    path:   path that we want to replace the extension to .npy

    returns
    -------
    path:   new path whose extension is already changed to .npy

    """

    if ".jpg" in path:
        path = path.replace(".jpg", ".npy")
    elif ".png" in path:
        path = path.replace(".png", ".npy")
    return path

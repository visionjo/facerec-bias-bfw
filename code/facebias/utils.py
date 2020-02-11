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
    last_score = -1
    counter = 0
    for threshold in thresholds:

        score = function(threshold, predicts)
        if score < last_score:
            counter += 1
        else:
            counter = 0
        if counter > 5:
            break
        if op(score, best_score):
            best_score = score
            best_threshold = threshold

        last_score = score
    return best_threshold, best_score


def get_acc_per_threshold(thresholds, predicts):
    """
    Calculate accuracies for a list of thresholds (see eval_acc())
    :param thresholds: threshold values to calculate accuracy with respect to.
    :param predicts:   predictions [p1, p2, score, label], where score and label
                        (index 2 and 3) are used.
    :return:    array of accuracies with same size and order as thresholds (ie
                results for each element of thresholds).
    """
    accs = np.zeros(len(thresholds))
    best_threshold = 0
    best_acc = 0
    for i, threshold in enumerate(thresholds):
        accs[i] = eval_acc(threshold, predicts)
        best_threshold = threshold if accs[i] > best_acc else best_threshold
        best_acc = accs[i] if accs[i] > best_acc else best_acc

    return accs, best_acc, best_threshold


def replace_ext(path):
    """
    Replace the extension (.jpg or .png) of path to .npy

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
